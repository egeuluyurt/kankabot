"""
Kanka Sinyal Botu v4.0
======================
Alpaca Markets + yfinance/ta + ApeWisdom + Finnhub + Telegram
"""

import os
import re
import sys
import time
import asyncio
import logging
import argparse
import threading
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import requests
import yfinance as yf
import pandas as pd
import ta
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    ReplaceOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, OrderType, QueryOrderStatus

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

from regime import detect_regime, MarketRegime
from sizing import calculate_position_size
from alternative_data import (
    get_insider_sentiment,
    get_economic_calendar,
    get_llm_sentiment_analysis,
    get_fred_macro_data,
)

from telegram import Update
from telegram.error import Conflict
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ─── Loglama ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─── Konfigürasyon ────────────────────────────────────────────────────────────
load_dotenv("config.env")

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
PAPER_TRADING     = os.getenv("PAPER_TRADING", "true").lower() == "true"
FINNHUB_API_KEY   = os.getenv("FINNHUB_API_KEY", "")
TG_TOKEN          = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID        = os.getenv("TELEGRAM_CHAT_ID", "")
TICKERS           = [t.strip() for t in os.getenv("TICKERS", "NVDA,AMD,QQQ,SMH,AAPL,MSFT,TSLA,PLTR,SPY,UNH").split(",")]
BUY_THRESHOLD     = float(os.getenv("BUY_THRESHOLD", "70"))
SELL_THRESHOLD    = float(os.getenv("SELL_THRESHOLD", "35"))
POSITION_PCT      = float(os.getenv("POSITION_PCT", "5"))
MAX_POSITIONS     = int(os.getenv("MAX_POSITIONS", "3"))
ATR_SL_MULT       = float(os.getenv("ATR_SL_MULT", "1.5"))
ATR_TP_MULT       = float(os.getenv("ATR_TP_MULT", "3.0"))
SCAN_INTERVAL     = int(os.getenv("SCAN_INTERVAL_MINUTES", "60"))

TRAILING_TRIGGER_PCT = 5.0   # Kâr >= %5 → SL'yi giriş fiyatına taşı (breakeven)
TIME_STOP_DAYS       = 5     # 5 günden uzun açık kalan pozisyonu kapat
EARLY_EXIT_PCT       = -7.0  # Zarar >= %-7 → SL beklenmeden piyasadan çık

# ─── Global durum ─────────────────────────────────────────────────────────────
BOT_PAUSED       = False
CRITICAL_DATA_OK = True   # False olursa tüm alımlar durur
vader      = SentimentIntensityAnalyzer()
ml_model   = None   # joblib ile yüklenen LGBMClassifier (main'de doldurulur)

class InverseVarianceWeighter:
    def __init__(self, window: int = 20):
        self.window     = window
        self._tech: list = []
        self._ml:   list = []

    def weighted_score(self, tech: float, ml: float) -> float:
        self._tech.append(tech)
        self._ml.append(ml)
        if len(self._tech) > self.window:
            self._tech.pop(0)
            self._ml.pop(0)

        if len(self._tech) < 5:
            w_t, w_m = 0.4, 0.6   # yeterli veri yoksa sabit fallback
        else:
            var_t = max(pd.Series(self._tech).var(), 1e-6)
            var_m = max(pd.Series(self._ml).var(),   1e-6)
            inv_t = 1.0 / var_t
            inv_m = 1.0 / var_m
            total = inv_t + inv_m
            w_t   = inv_t / total
            w_m   = inv_m / total

        final = tech * w_t + ml * w_m
        log.info(
            f"IV Ağırlık → Tech: {w_t:.2f}, ML: {w_m:.2f}, "
            f"Final: {final:.1f}"
        )
        return round(final, 1)

iv_weighter = InverseVarianceWeighter(window=20)

# ─── Finansal VADER ön işleme sözlüğü ────────────────────────────────────────
VADER_REPLACE = {
    r"\bputs\b":        "terrible",
    r"\bput option\b":  "terrible",
    r"\bshort\b":       "bearish bad",
    r"\bcalls\b":       "excellent",
    r"\bcall option\b": "excellent",
    r"\blong\b":        "bullish good",
    r"\bmoon\b":        "excellent amazing",
    r"\byolo\b":        "risky gamble",
    r"\bsqueeze\b":     "explosive surge",
    r"\bbag\b":         "losing terrible",
    r"\brug\b":         "scam terrible",
    r"\bbankruptcy\b":  "disaster terrible",
}

def preprocess_text(text: str) -> str:
    """Finansal terimleri VADER için nötralize eder."""
    text = text.lower()
    for pattern, replacement in VADER_REPLACE.items():
        text = re.sub(pattern, replacement, text)
    return text

def vader_to100(compound: float) -> float:
    """VADER compound [-1,1] → [0,100]"""
    return (compound + 1) / 2 * 100

# ─── Tenacity retry dekoratörü ────────────────────────────────────────────────
def _is_rate_limit(exc: Exception) -> bool:
    """HTTP 429 kontrolü."""
    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and exc.response.status_code == 429
    return False

retry_on_rate_limit = retry(
    retry=retry_if_exception_type(requests.HTTPError),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)

# ─── Telegram (senkron fallback) ──────────────────────────────────────────────
def tg_send(text: str) -> None:
    """Telegram'a senkron mesaj gönderir. Hata olursa loglar."""
    if not TG_TOKEN or not TG_CHAT_ID:
        log.warning("Telegram: TG_TOKEN veya TG_CHAT_ID tanımlı değil, mesaj atlandı.")
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if not resp.ok:
            log.warning(f"Telegram API hatası {resp.status_code}: {resp.text}")
        else:
            log.info(f"Telegram mesajı gönderildi (chat_id={TG_CHAT_ID})")
    except Exception as e:
        log.warning(f"Telegram gönderme hatası: {e}")

# ─── Alpaca istemcisi ─────────────────────────────────────────────────────────
class AlpacaEngine:
    def __init__(self):
        self.client = TradingClient(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            paper=PAPER_TRADING,
        )
        self.data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        self.mod = "PAPER" if PAPER_TRADING else "LIVE"

    @retry_on_rate_limit
    def get_account(self):
        return self.client.get_account()

    @retry_on_rate_limit
    def get_positions(self):
        return self.client.get_all_positions()

    def get_mid_price(self, symbol: str) -> Optional[float]:
        """Gerçek zamanlı bid/ask ortasını döndürür. Başarısız olursa None."""
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.data_client.get_stock_latest_quote(req)
            bid = float(quote[symbol].bid_price)
            ask = float(quote[symbol].ask_price)
            if bid > 0 and ask > 0:
                return round((bid + ask) / 2, 2)
        except Exception as e:
            log.warning(f"{symbol} mid-price alınamadı: {e}")
        return None

    @retry_on_rate_limit
    def place_bracket_buy(self, symbol: str, notional: float, price: float, sl_price: float, tp_price: float):
        """
        Bracket limit order — giriş fiyatı mid-point (bid/ask ortası).
        Alpaca bracket order notional desteklemediği için qty hesaplanır.
        """
        qty = max(1, int(notional / price))  # Bracket order tam sayı gerektiriyor
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=round(price, 2),
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=round(sl_price, 2)),
            take_profit=TakeProfitRequest(limit_price=round(tp_price, 2)),
        )
        return self.client.submit_order(req)

    def replace_order(self, order_id: str, new_limit: float) -> bool:
        """
        Açık limit emrinin giriş fiyatını günceller.
        422 'order parameters are not changed' hatasını tolere eder.
        """
        try:
            req = ReplaceOrderRequest(limit_price=round(new_limit, 2))
            self.client.replace_order_by_id(order_id, req)
            return True
        except Exception as e:
            err_str = str(e)
            if "422" in err_str or "order parameters are not changed" in err_str.lower():
                log.info(f"Emir güncellenmedi (parametreler değişmedi): {order_id}")
                return True  # Hata değil, parametre aynı
            log.error(f"Emir güncelleme hatası ({order_id}): {e}")
            return False

    def cancel_all_orders(self) -> int:
        """Tüm açık emirleri iptal eder. İptal edilen emir sayısını döndürür."""
        try:
            cancelled = self.client.cancel_orders()
            count = len(cancelled) if cancelled else 0
            log.info(f"Kapanış öncesi {count} açık emir iptal edildi.")
            return count
        except Exception as e:
            log.error(f"Emir iptali hatası: {e}")
            return 0

    def close_position(self, symbol: str) -> bool:
        """Pozisyonu piyasa fiyatından kapatır; bağlı SL/TP emirleri iptal edilir."""
        try:
            self.client.close_position(symbol)
            log.info(f"{symbol}: Pozisyon kapatıldı (close_position API)")
            return True
        except Exception as e:
            log.error(f"{symbol} pozisyon kapatma hatası: {e}")
            return False

    def get_open_sell_orders(self, symbol: str) -> list:
        """Bir sembol için açık satış emirlerini döndürür."""
        try:
            req = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                symbols=[symbol],
            )
            orders = self.client.get_orders(req)
            return [o for o in orders if o.side == OrderSide.SELL]
        except Exception as e:
            log.warning(f"{symbol} açık emirler alınamadı: {e}")
            return []

    def update_stop_price(self, order_id: str, new_stop: float) -> bool:
        """Açık stop emrinin tetikleme fiyatını günceller."""
        try:
            req = ReplaceOrderRequest(stop_price=round(new_stop, 2))
            self.client.replace_order_by_id(order_id, req)
            return True
        except Exception as e:
            err_str = str(e)
            if "422" in err_str or "order parameters are not changed" in err_str.lower():
                return True
            log.error(f"Stop fiyatı güncelleme hatası ({order_id}): {e}")
            return False

    @retry_on_rate_limit
    def place_sell(self, symbol: str, notional: float):
        """Mevcut pozisyonu piyasa fiyatından satar."""
        req = MarketOrderRequest(
            symbol=symbol,
            notional=round(notional, 2),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        return self.client.submit_order(req)


# ─── Borsa saatleri ───────────────────────────────────────────────────────────
def is_market_hours() -> bool:
    """NYSE açık mı? (Hafta içi 09:30–16:00 ET)"""
    ny = datetime.now(ZoneInfo("America/New_York"))
    if ny.weekday() >= 5:  # Cumartesi=5, Pazar=6
        return False
    open_time  = ny.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_time = ny.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_time <= ny <= close_time

def _is_near_close() -> bool:
    """NYSE kapanışına 5 dakika veya daha az kaldı mı? (weekday 15:55–16:00 ET)"""
    ny = datetime.now(ZoneInfo("America/New_York"))
    if ny.weekday() >= 5:
        return False
    close_time = ny.replace(hour=16, minute=0, second=0, microsecond=0)
    seconds_to_close = (close_time - ny).total_seconds()
    return 0 <= seconds_to_close <= 300  # Kapanışa 0–5 dakika kaldı

# ─── Teknik Analiz ────────────────────────────────────────────────────────────
def _download_fix(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    yf.Ticker().history() ile veri indirir — MultiIndex sorunu yok.
    Sütun isimleri küçük harfe çevrilir.
    """
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        return df
    df.columns = [c.lower() for c in df.columns]
    return df

def get_technical_score(ticker: str) -> dict:
    """
    Günlük (1D) + Saatlik (1H) MTF teknik analiz.
    Döndürür: {'tech_score', 'daily_score', 'rsi_score', 'macd_score',
                'rsi_val', 'macd_dir', 'price', 'ema200', 'atr'}
    """
    result = {
        "tech_score": 50.0, "daily_score": 50.0,
        "rsi_score": 50.0,  "macd_score": 50.0,
        "rsi_val": 50.0,    "macd_dir": "—",
        "price": 0.0,       "ema200": 0.0,
        "atr": None,
        "regime": MarketRegime.UNKNOWN,
        "adx": 25.0,
        "hurst": 0.5,
    }
    try:
        # ── Günlük filtre ─────────────────────────────────────────────────
        daily = _download_fix(ticker, "2y", "1d")
        if daily.empty or len(daily) < 200:
            log.warning(f"{ticker}: Yetersiz günlük veri ({len(daily)} bar)")
            return result

        # NaN satırları temizle
        close = daily["close"].ffill().dropna()
        log.info(f"{ticker}: {len(close)} günlük bar yüklendi")

        # pandas ewm ile EMA — min_periods=0 ile her zaman değer döner
        ema50_series  = close.ewm(span=50,  adjust=False, min_periods=0).mean()
        ema200_series = close.ewm(span=200, adjust=False, min_periods=0).mean()

        price    = float(close.iloc[-1])
        ema50_v  = float(ema50_series.iloc[-1])
        ema200_v = float(ema200_series.iloc[-1])
        log.info(f"{ticker}: Fiyat={price:.2f} EMA50={ema50_v:.2f} EMA200={ema200_v:.2f}")
        result["price"]  = price
        result["ema200"] = ema200_v

        if price > ema50_v and ema50_v > ema200_v:
            daily_score = 100
        elif price > ema200_v:
            daily_score = 70
        elif price < ema50_v and ema50_v < ema200_v:
            daily_score = 0
        elif price < ema200_v:
            daily_score = 30
        else:
            daily_score = 50
        result["daily_score"] = daily_score

        # ── Piyasa rejimi (ADX + Hurst) ───────────────────────────────────
        regime, regime_det = detect_regime(daily)
        result["regime"] = regime
        result["adx"]    = regime_det["adx"]
        result["hurst"]  = regime_det["hurst"]
        log.info(
            f"{ticker} Rejim: {regime} | "
            f"ADX={regime_det['adx']:.1f} | Hurst={regime_det['hurst']:.3f}"
        )

        # ── Saatlik tetikleyici ───────────────────────────────────────────
        hourly = _download_fix(ticker, "60d", "1h")
        if hourly.empty or len(hourly) < 30:
            log.warning(f"{ticker}: Yetersiz saatlik veri")
            result["tech_score"] = daily_score * 0.40 + 50 * 0.30 + 50 * 0.30
            return result

        rsi_series = ta.momentum.rsi(hourly["close"], window=14)
        macd_ind   = ta.trend.MACD(hourly["close"], window_fast=12, window_slow=26, window_sign=9)
        macd_line  = macd_ind.macd()
        macd_sig   = macd_ind.macd_signal()
        macd_hist  = macd_ind.macd_diff()

        # ATR (daily üzerinden hesapla)
        daily_atr = ta.volatility.AverageTrueRange(
            daily["high"], daily["low"], daily["close"], window=14
        ).average_true_range()
        if daily_atr is not None and not daily_atr.dropna().empty:
            result["atr"] = float(daily_atr.iloc[-1])

        if rsi_series is None or rsi_series.dropna().empty:
            rsi_score = 50.0
            rsi_val   = 50.0
        else:
            rsi_val  = float(rsi_series.iloc[-1])
            rsi_prev = float(rsi_series.iloc[-2]) if len(rsi_series) >= 2 else rsi_val
            result["rsi_val"] = rsi_val

            if rsi_val >= 70:
                rsi_score = 15
            elif rsi_val >= 55:
                rsi_score = 60 + (rsi_val - 55) * 2
            elif rsi_val <= 30 and rsi_val > rsi_prev:
                rsi_score = 95
            elif rsi_val <= 30:
                rsi_score = 85
            elif rsi_val <= 40 and rsi_val > rsi_prev:
                rsi_score = 75
            else:
                rsi_score = 40 + rsi_val * 0.3

        if macd_line is None or macd_line.dropna().empty:
            macd_score = 45.0
            macd_dir   = "—"
        else:
            macd_v  = float(macd_line.iloc[-1])
            sig_v   = float(macd_sig.iloc[-1])
            hist_v  = float(macd_hist.iloc[-1])
            hist_p  = float(macd_hist.iloc[-2]) if len(macd_hist) >= 2 else hist_v

            hist_growing = abs(hist_v) > abs(hist_p)

            if macd_v > sig_v and hist_v > 0 and hist_growing:
                macd_score = 100
                macd_dir   = "↑↑ güçlü"
            elif macd_v > sig_v and hist_v > 0:
                macd_score = 75
                macd_dir   = "↑ yükseliş"
            elif macd_v > sig_v and hist_v < 0:
                macd_score = 55
                macd_dir   = "↗ zayıf"
            elif macd_v < sig_v and hist_v < 0 and hist_growing:
                macd_score = 0
                macd_dir   = "↓↓ güçlü düşüş"
            elif macd_v < sig_v and hist_v < 0:
                macd_score = 25
                macd_dir   = "↓ düşüş"
            else:
                macd_score = 45
                macd_dir   = "— kararsız"

        result["rsi_score"]  = rsi_score
        result["macd_score"] = macd_score
        result["macd_dir"]   = macd_dir

        # ── MTF çelişki cezası ────────────────────────────────────────────
        daily_bull  = price > ema200_v
        daily_bear  = not daily_bull
        hourly_bull = rsi_val > 60 and macd_score > 60
        hourly_bear = rsi_val < 40 and macd_score < 40

        # ── Rejime göre ağırlıklı tech_score ─────────────────────────────
        # TREND   → trend takip indikatörleri (EMA/MACD) ön planda
        # RANGING → aşırı alım/satım indikatörü (RSI) ön planda
        # UNKNOWN → dengeli orta yol
        _regime = result.get("regime", MarketRegime.UNKNOWN)
        if _regime == MarketRegime.TRENDING:
            # daily(EMA)=0.50, macd=0.35, rsi=0.15
            tech_score = daily_score * 0.50 + macd_score * 0.35 + rsi_score * 0.15
        elif _regime == MarketRegime.RANGING:
            # rsi=0.55, daily(EMA)=0.30, macd=0.15
            tech_score = daily_score * 0.30 + rsi_score * 0.55 + macd_score * 0.15
        else:
            # UNKNOWN — eski dengeli ağırlıklar
            tech_score = daily_score * 0.40 + rsi_score * 0.30 + macd_score * 0.30

        if daily_bull and hourly_bear:
            tech_score *= 0.85
        elif daily_bear and hourly_bull:
            tech_score *= 0.85

        log.info(
            f"{ticker} tech_score={tech_score:.1f} "
            f"(rejim={_regime}, daily={daily_score:.0f}, "
            f"rsi={rsi_score:.0f}, macd={macd_score:.0f})"
        )

        result["tech_score"] = tech_score
        result["daily"]      = daily   # get_ml_score için ham günlük veri

    except Exception as e:
        log.error(f"{ticker} teknik analiz hatası: {e}")

    return result


# ─── ApeWisdom Reddit Sentiment ───────────────────────────────────────────────
@retry_on_rate_limit
def get_apewisdom_score(ticker: str) -> float:
    """
    ApeWisdom API üzerinden Reddit sentiment skoru döndürür.
    API key gerektirmez. Bulunamazsa 50 (nötr) döner.
    """
    try:
        url = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        for item in results:
            if item.get("ticker", "").upper() == ticker.upper():
                mentions      = float(item.get("mentions", 0))
                rank          = int(item.get("rank", 999))
                rank_24h_ago  = int(item.get("rank_24h_ago", rank))

                mentions_score  = min(mentions / 50.0 * 100, 100)
                rank_change     = rank_24h_ago - rank  # pozitif = yükselen ilgi
                momentum_bonus  = min(rank_change * 2, 20)
                reddit_score    = min(mentions_score + momentum_bonus, 100)

                log.info(
                    f"{ticker} ApeWisdom: mentions={mentions:.0f}, "
                    f"rank={rank} (önceki {rank_24h_ago}), "
                    f"skor={reddit_score:.1f}"
                )
                return reddit_score

        log.info(f"{ticker} ApeWisdom'da bulunamadı, nötr skor (50)")
        return 50.0

    except Exception as e:
        log.warning(f"{ticker} ApeWisdom hatası: {e}")
        return 50.0


# ─── Finnhub Sentiment ────────────────────────────────────────────────────────
@retry_on_rate_limit
def get_finnhub_score(ticker: str) -> float:
    """
    Önce Finnhub /news-sentiment endpoint'ini dener.
    Başarısız olursa /company-news + VADER ile fallback yapar.
    """
    base = "https://finnhub.io/api/v1"
    headers = {"X-Finnhub-Token": FINNHUB_API_KEY}

    if not FINNHUB_API_KEY:
        return 50.0

    # Birincil: haber sentiment
    try:
        url  = f"{base}/news-sentiment?symbol={ticker}"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "bullishPercent" in data:
            score = float(data["bullishPercent"]) * 100
            log.info(f"{ticker} Finnhub sentiment: {score:.1f}")
            return score
    except Exception as e:
        log.warning(f"{ticker} Finnhub /news-sentiment hatası: {e}, fallback deneniyor...")

    # Fallback: VADER analizi
    try:
        from datetime import timedelta
        today = datetime.now(ZoneInfo("America/New_York"))
        week_ago = today - timedelta(days=7)
        url = (
            f"{base}/company-news?symbol={ticker}"
            f"&from={week_ago.strftime('%Y-%m-%d')}"
            f"&to={today.strftime('%Y-%m-%d')}"
        )
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        news = resp.json()

        if not news:
            return 50.0

        scores = []
        for article in news[:20]:
            headline = article.get("headline", "")
            summary  = article.get("summary", "")
            text     = preprocess_text(f"{headline} {summary}")
            compound = vader.polarity_scores(text)["compound"]
            scores.append(vader_to100(compound))

        score = sum(scores) / len(scores) if scores else 50.0
        log.info(f"{ticker} Finnhub VADER fallback: {score:.1f} ({len(scores)} haber)")
        return score

    except Exception as e:
        log.warning(f"{ticker} Finnhub fallback hatası: {e}")
        return 50.0


# ─── Bileşik Skor & Sinyal ────────────────────────────────────────────────────
def composite_score(
    tech: float,
    reddit: float,
    finnhub: float,
    insider: float = 50.0,
    llm_score: Optional[float] = None,
) -> tuple[float, float, float]:
    """
    (final_score, tech_score, sentiment_score) döndürür.

    Ağırlıklar:
      Teknik       : %50
      Sentiment    : %20  (reddit + finnhub)
      Alternatif   : %30  (insider + LLM)

    LLM skoru yoksa alternatif = insider (tek başına %30).
    """
    sentiment = reddit * 0.50 + finnhub * 0.50

    # Alternatif veri: insider + LLM (LLM opsiyonel)
    if llm_score is not None:
        alt_score = insider * 0.50 + llm_score * 0.50
    else:
        alt_score = insider

    final = tech * 0.50 + sentiment * 0.20 + alt_score * 0.30
    return final, tech, sentiment

# ─── ML Skor ──────────────────────────────────────────────────────────────────
ML_FEATURES = [
    "price_ema50_ratio", "price_ema200_ratio",
    "rsi", "macd", "macd_hist", "macd_above_signal",
    "atr", "atr_pct", "bb_width", "adx", "hurst",
    "daily_return", "log_return", "volume_zscore",
    "vix", "fed_rate", "cpi",
]

def get_ml_score(df: pd.DataFrame, macro_data: dict) -> float:
    """
    Günlük OHLCV df üzerinden indikatörleri hesaplar, en son satırı alır,
    ml_model.predict_proba() ile yükseliş olasılığını 0–100 skoruna çevirir.
    Model yüklü değilse veya hata olursa 50 (nötr) döner.
    """
    global ml_model
    if ml_model is None or df is None or df.empty:
        return 50.0

    try:
        df = df.copy()
        close = df["close"].ffill()

        ema50  = close.ewm(span=50,  adjust=False, min_periods=0).mean()
        ema200 = close.ewm(span=200, adjust=False, min_periods=0).mean()

        macd_ind = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
        atr_ind  = ta.volatility.AverageTrueRange(df["high"], df["low"], close, window=14)
        bb_ind   = ta.volatility.BollingerBands(close, window=20)
        adx_ind  = ta.trend.ADXIndicator(df["high"], df["low"], close, window=14)

        atr_val = float(atr_ind.average_true_range().iloc[-1])

        # Hurst: son 60 günlük pencerede hesapla
        from regime import calculate_hurst
        try:
            hurst = calculate_hurst(close.iloc[-60:]) if len(close) >= 20 else 0.5
        except Exception:
            hurst = 0.5

        vol_mean = df["volume"].rolling(20).mean().iloc[-1]
        vol_std  = df["volume"].rolling(20).std().iloc[-1]
        vol_std  = vol_std if vol_std and vol_std > 0 else 1.0

        row = {
            "price_ema50_ratio":  float(close.iloc[-1] / ema50.iloc[-1]),
            "price_ema200_ratio": float(close.iloc[-1] / ema200.iloc[-1]),
            "rsi":                float(ta.momentum.rsi(close, window=14).iloc[-1]),
            "macd":               float(macd_ind.macd().iloc[-1]),
            "macd_hist":          float(macd_ind.macd_diff().iloc[-1]),
            "macd_above_signal":  int(macd_ind.macd().iloc[-1] > macd_ind.macd_signal().iloc[-1]),
            "atr":                atr_val,
            "atr_pct":            atr_val / float(close.iloc[-1]),
            "bb_width":           float((bb_ind.bollinger_hband().iloc[-1] - bb_ind.bollinger_lband().iloc[-1]) / close.iloc[-1]),
            "adx":                float(adx_ind.adx().iloc[-1]),
            "hurst":              hurst,
            "daily_return":       float(close.pct_change().iloc[-1]),
            "log_return":         float(np.log(close.iloc[-1] / close.iloc[-2])) if len(close) >= 2 else 0.0,
            "volume_zscore":      float((df["volume"].iloc[-1] - vol_mean) / vol_std),
            "vix":                float(macro_data.get("vix", 20.0)),
            "fed_rate":           float(macro_data.get("rate", 5.0)),
            "cpi":                float(macro_data.get("cpi", 3.0)),
        }

        X     = pd.DataFrame([[row[f] for f in ML_FEATURES]], columns=ML_FEATURES)
        proba = float(ml_model.predict_proba(X)[0][1])
        score = round(proba * 100, 1)
        return score

    except Exception as e:
        log.warning(f"ML skor hesaplama hatası: {e}")
        return 50.0


def confidence_level(tech: float, sentiment: float) -> str:
    """İki skor aynı yönde ise YÜKSEK, aksi hâlde ORTA."""
    if (tech >= 60 and sentiment >= 60) or (tech <= 40 and sentiment <= 40):
        return "YÜKSEK"
    return "ORTA"

def get_dynamic_thresholds(vix: float) -> tuple:
    """
    VIX seviyesine göre alım/satım eşiklerini dinamik ayarlar.
    Düşük volatilitede eşikler daralır (daha fazla sinyal).
    Krizde eşikler genişler (sahte sinyallerden korunur).
    """
    if vix < 15:
        return 62.0, 42.0    # Düşük vol: dar bant
    elif vix <= 25:
        return 70.0, 35.0    # Normal: mevcut değerler
    elif vix <= 35:
        return 78.0, 28.0    # Yüksek vol: genişlet
    else:
        return 85.0, 22.0    # Kriz: maksimum genişlik

def signal_label(final: float, confidence: str, vix: float = 20.0) -> str:
    buy_thr, sell_thr = get_dynamic_thresholds(vix)
    if final >= 80 and confidence == "YÜKSEK":
        return "💪 GÜÇLÜ AL"
    if final >= buy_thr:
        return "✅ AL"
    if final <= 20 and confidence == "YÜKSEK":
        return "🔴 GÜÇLÜ SAT"
    if final <= sell_thr:
        return "🔴 SAT"
    return "⏸ BEKLE"


# ─── Emir Yönetimi ────────────────────────────────────────────────────────────
def should_buy(
    ticker: str,
    positions: list,
    portfolio_value: float,
    buying_power: float,
    atr: Optional[float],
    regime: str = MarketRegime.UNKNOWN,
) -> tuple[bool, str]:
    """Alım koşullarını sırayla kontrol eder."""
    global BOT_PAUSED
    if BOT_PAUSED:
        return False, "Bot duraklatıldı"
    if not CRITICAL_DATA_OK:
        return False, "Dead Man's Switch aktif — kritik veri yok"
    if not is_market_hours():
        return False, "Borsa kapalı"
    if regime == MarketRegime.RANGING:
        return False, "Yatay piyasa (RANGING) — trend-takip sinyali geçersiz"
    active_symbols = [p.symbol for p in positions]
    if len(active_symbols) >= MAX_POSITIONS:
        return False, f"Pozisyon limiti ({MAX_POSITIONS})"
    if ticker in active_symbols:
        return False, "Zaten pozisyonda"
    position_size = portfolio_value * POSITION_PCT / 100
    if buying_power < position_size:
        return False, f"Yetersiz nakit (${buying_power:.0f} < ${position_size:.0f})"
    if atr is None or atr <= 0:
        return False, "ATR verisi yok"
    return True, "OK"


def place_bracket_buy(
    engine: AlpacaEngine,
    ticker: str,
    portfolio_value: float,
    price: float,
    atr: float,
    final_score: float = 65.0,
    macro_multiplier: float = 1.0,
) -> str:
    """Bracket limit-buy emri gönderir. Giriş fiyatı mid-point (bid/ask ortası)."""
    notional = calculate_position_size(
        portfolio_value=portfolio_value,
        final_score=final_score,
        atr=atr,
        price=price,
        tp_sl_ratio=ATR_TP_MULT / ATR_SL_MULT,
        macro_multiplier=macro_multiplier,
    )

    # Gerçek zamanlı mid-point al; başarısız olursa teknik analiz fiyatını kullan
    mid = engine.get_mid_price(ticker)
    entry_price = mid if mid else price
    log.info(f"{ticker} giriş fiyatı: mid={mid} → entry={entry_price:.2f}")

    sl_price = entry_price - ATR_SL_MULT * atr
    tp_price = entry_price + ATR_TP_MULT * atr

    try:
        order = engine.place_bracket_buy(ticker, notional, entry_price, sl_price, tp_price)
        msg = (
            f"✅ Bracket Limit Buy: {ticker} | "
            f"Limit=${entry_price:.2f} | "
            f"Notional=${notional:.0f} | "
            f"SL={sl_price:.2f} | TP={tp_price:.2f}"
        )
        log.info(msg)
        return msg
    except Exception as e:
        msg = f"❌ Bracket Buy hatası ({ticker}): {e}"
        log.error(msg)
        return msg


def place_sell(engine: AlpacaEngine, ticker: str, portfolio_value: float) -> str:
    """Mevcut pozisyonu satar."""
    notional = portfolio_value * POSITION_PCT / 100
    try:
        order = engine.place_sell(ticker, notional)
        msg = f"🔴 Sell: {ticker} | Notional=${notional:.0f}"
        log.info(msg)
        return msg
    except Exception as e:
        msg = f"❌ Sell hatası ({ticker}): {e}"
        log.error(msg)
        return msg


# ─── Pozisyon Yönetimi ────────────────────────────────────────────────────────
def morning_report(engine: AlpacaEngine) -> None:
    """NYSE açılışı (09:30–09:44 ET) penceresinde açık pozisyon özetini gönderir."""
    ny = datetime.now(ZoneInfo("America/New_York"))
    if not (ny.hour == 9 and 30 <= ny.minute <= 44):
        return
    try:
        positions = engine.get_positions()
        if not positions:
            tg_send("🌅 <b>Sabah Raporu</b>\nAçık pozisyon yok.")
            return
        lines = [f"🌅 <b>Sabah Raporu — {ny.strftime('%Y-%m-%d')}</b>"]
        for p in positions:
            entry   = float(p.avg_entry_price)
            cur     = float(p.current_price)
            pnl     = float(p.unrealized_pl)
            pnl_pct = float(p.unrealized_plpc) * 100
            market  = float(p.market_value)
            lines.append(
                f"\n<b>{p.symbol}</b>\n"
                f"  Giriş  : ${entry:.2f} → Şimdi: ${cur:.2f}\n"
                f"  Değer  : ${market:.2f}\n"
                f"  K/Z    : ${pnl:+.2f} (%{pnl_pct:+.1f})"
            )
        tg_send("\n".join(lines))
    except Exception as e:
        log.error(f"Sabah raporu hatası: {e}")


def manage_open_positions(engine: AlpacaEngine, positions: list) -> None:
    """
    Her taramada açık pozisyonları denetler:
      1. Erken çıkış  : Zarar >= EARLY_EXIT_PCT%     → piyasa fiyatından kapat
      2. Breakeven SL : Kâr   >= TRAILING_TRIGGER_PCT% → SL'yi giriş fiyatına taşı
      3. Zaman durağı : Pozisyon >= TIME_STOP_DAYS gün açık → kapat
    """
    now_ny = datetime.now(ZoneInfo("America/New_York"))

    for p in positions:
        symbol  = p.symbol
        entry   = float(p.avg_entry_price)
        cur     = float(p.current_price)
        pnl     = float(p.unrealized_pl)
        pnl_pct = float(p.unrealized_plpc) * 100
        market  = float(p.market_value)

        try:
            # ── 1. Erken çıkış ────────────────────────────────────────────────
            if pnl_pct <= EARLY_EXIT_PCT:
                log.warning(
                    f"{symbol}: Erken çıkış — zarar %{pnl_pct:.1f} "
                    f"(eşik %{EARLY_EXIT_PCT})"
                )
                if engine.close_position(symbol):
                    tg_send(
                        f"🚨 <b>Erken Çıkış — {symbol}</b>\n"
                        f"Zarar %{pnl_pct:.1f} eşiği aştı (eşik %{EARLY_EXIT_PCT})\n"
                        f"Giriş: ${entry:.2f} | Şimdi: ${cur:.2f}\n"
                        f"Piyasa fiyatından kapatıldı."
                    )
                continue  # Diğer kontrolleri atla

            # ── Açık satış emirlerini tek seferde çek (2 ve 3 için) ───────────
            sell_orders = engine.get_open_sell_orders(symbol)

            # ── 2. Breakeven trailing stop ────────────────────────────────────
            if pnl_pct >= TRAILING_TRIGGER_PCT:
                sl_orders = [o for o in sell_orders if o.order_type == OrderType.STOP]
                if sl_orders:
                    sl_order     = sl_orders[0]
                    current_stop = float(sl_order.stop_price) if sl_order.stop_price else 0.0
                    if current_stop < entry:
                        if engine.update_stop_price(sl_order.id, entry):
                            log.info(f"{symbol}: Breakeven SL güncellendi → ${entry:.2f}")
                            tg_send(
                                f"🔒 <b>Breakeven Stop — {symbol}</b>\n"
                                f"Kâr %{pnl_pct:.1f} → SL giriş fiyatına taşındı\n"
                                f"Yeni SL: ${entry:.2f}"
                            )

            # ── 3. Zaman durağı ───────────────────────────────────────────────
            if sell_orders:
                oldest    = min(sell_orders, key=lambda o: o.created_at)
                tz_aware  = oldest.created_at.astimezone(ZoneInfo("America/New_York"))
                days_held = (now_ny - tz_aware).days
                if days_held >= TIME_STOP_DAYS:
                    log.warning(f"{symbol}: Zaman durağı — {days_held} gün açık")
                    if engine.close_position(symbol):
                        tg_send(
                            f"⏱ <b>Zaman Durağı — {symbol}</b>\n"
                            f"{days_held} gündür açık, TP/SL tetiklenmedi\n"
                            f"Giriş: ${entry:.2f} | Şimdi: ${cur:.2f}\n"
                            f"K/Z: ${pnl:+.2f} (%{pnl_pct:+.1f})\n"
                            f"Piyasa fiyatından kapatıldı."
                        )

        except Exception as e:
            log.error(f"{symbol} pozisyon yönetimi hatası: {e}")


# ─── Ana Tarama Döngüsü ───────────────────────────────────────────────────────
def scan_once(engine: AlpacaEngine) -> None:
    """Tüm hisseleri bir kez tarar."""
    global BOT_PAUSED
    if BOT_PAUSED:
        log.info("Bot duraklatıldı, tarama atlandı.")
        return

    # ── Kapanış yakınsa emirleri iptal et, taramayı atla ─────────────────────
    if _is_near_close():
        log.info("NYSE kapanışına 5 dakika kaldı — açık emirler iptal ediliyor.")
        count = engine.cancel_all_orders()
        if count > 0:
            tg_send(f"⏰ NYSE kapanışına yakın: {count} açık limit emir iptal edildi.")
        log.info("Kapanış öncesi tarama atlandı.")
        return

    try:
        account        = engine.get_account()
        portfolio_val  = float(account.portfolio_value)
        buying_power   = float(account.buying_power)
        positions      = engine.get_positions()
    except Exception as e:
        log.error(f"Alpaca hesap bilgisi alınamadı: {e}")
        return

    active_tickers = {p.symbol for p in positions}
    log.info(
        f"Tarama başlıyor | Portföy: ${portfolio_val:.0f} | "
        f"Nakit: ${buying_power:.0f} | Açık pozisyon: {len(active_tickers)}"
    )

    # ── Sabah raporu (09:30–09:44 ET) ────────────────────────────────────────
    morning_report(engine)

    # ── Açık pozisyonları yönet ───────────────────────────────────────────────
    if positions and is_market_hours():
        manage_open_positions(engine, positions)
        # Erken çıkış / zaman durağı sonrası listeyi tazele
        positions      = engine.get_positions()
        active_tickers = {p.symbol for p in positions}

    # ── Makro veri — tarama başında bir kez çekilir ──────────────────────────
    macro_data       = get_fred_macro_data()
    if not CRITICAL_DATA_OK or not macro_data:
        log.error("Dead Man's Switch: Kritik veri eksik, tarama iptal edildi")
        tg_send("🛑 <b>Dead Man's Switch</b>\nMakro veri alınamadı — tüm işlemler durduruldu.")
        return
    vix              = macro_data["vix"]
    macro_multiplier = 1.0

    log.info(f"[MACRO] VIX: {vix:.1f}, Faiz: {macro_data['rate']:.2f}%, Enflasyon: {macro_data['cpi']:.1f}")

    if vix > 30:
        macro_multiplier = 0.5
        tg_send(
            f"⚠️ YÜKSEK VOLATİLİTE! VIX Endeksi {vix:.1f} seviyesinde. "
            f"Risk yönetimi gereği pozisyon büyüklükleri %50 azaltıldı."
        )
        log.warning(f"[MACRO] VIX={vix:.1f} > 30 — macro_multiplier=0.5 aktif")

    # ── Ekonomik takvim — tarama başında bir kez kontrol edilir ──────────────
    eco_risk, eco_event = get_economic_calendar()
    if eco_risk:
        log.warning(f"Ekonomik risk flag aktif: {eco_event}")
        tg_send(f"⚠️ <b>Ekonomik Olay Uyarısı</b>\n{eco_event}\nBugün yeni pozisyon açılmayacak.")

    for ticker in TICKERS:
        try:
            log.info(f"─── {ticker} analiz ediliyor...")

            tech_data  = get_technical_score(ticker)
            reddit_s   = get_apewisdom_score(ticker)
            finnhub_s  = get_finnhub_score(ticker)
            insider_s  = get_insider_sentiment(ticker)
            llm_s      = get_llm_sentiment_analysis(ticker)

            tech_s      = tech_data["tech_score"]
            _, _, sentiment_s = composite_score(tech_s, reddit_s, finnhub_s, insider_s, llm_s)

            # ── ML Skor ───────────────────────────────────────────────────
            ml_score = get_ml_score(tech_data.get("daily"), macro_data)
            final    = iv_weighter.weighted_score(tech_s, ml_score)

            log.info(
                f"[ML ANALİZ] {ticker} -> "
                f"Teknik: {tech_s:.1f}, ML: {ml_score:.1f}, Final: {final:.1f}"
            )

            conf   = confidence_level(tech_s, sentiment_s)
            signal = signal_label(final, conf, vix)

            log.info(
                f"{ticker} | Final={final:.1f} | Tech={tech_s:.1f} | "
                f"ML={ml_score:.1f} | Sentiment={sentiment_s:.1f} | Güven={conf} | {signal}"
            )

            action_msg = "İşlem yapılmadı"

            if "AL" in signal and is_market_hours():
                if eco_risk:
                    action_msg = f"Alım engellendi: Ekonomik olay ({eco_event})"
                    ok = False
                else:
                    ok, reason = should_buy(
                        ticker, positions, portfolio_val, buying_power,
                        tech_data["atr"], tech_data.get("regime", MarketRegime.UNKNOWN)
                    )
                    if not ok:
                        action_msg = f"Alım engellendi: {reason}"
                if ok:
                    notional = calculate_position_size(
                        portfolio_value=portfolio_val,
                        final_score=final,
                        atr=tech_data["atr"],
                        price=tech_data["price"],
                        tp_sl_ratio=ATR_TP_MULT / ATR_SL_MULT,
                        macro_multiplier=macro_multiplier,
                    )
                    if notional <= 0:
                        action_msg = "Alım engellendi: Kelly sıfır pozisyon önerdi"
                        log.info(f"{ticker}: {action_msg}")
                    else:
                        action_msg = place_bracket_buy(
                            engine, ticker, portfolio_val,
                            tech_data["price"], tech_data["atr"], final,
                            macro_multiplier
                        )
                        # Pozisyonu listeye ekle (cache için sahte nesne)
                        active_tickers.add(ticker)

            elif "SAT" in signal and ticker in active_tickers and is_market_hours():
                action_msg = place_sell(engine, ticker, portfolio_val)
                active_tickers.discard(ticker)

            # Telegram bildirimi (AL/SAT sinyallerinde)
            if signal != "⏸ BEKLE":
                _send_signal_message(
                    ticker, signal, final, conf, tech_data,
                    reddit_s, finnhub_s, sentiment_s,
                    insider_s, llm_s, ml_score, action_msg, engine.mod
                )

        except Exception as e:
            log.error(f"{ticker} tarama hatası: {e}")

        time.sleep(2)  # Rate limit koruması

    log.info("Tarama tamamlandı.")

def _send_signal_message(
    ticker, signal, final, conf, tech_data,
    reddit_s, finnhub_s, sentiment_s,
    insider_s, llm_s, ml_score, action_msg, mod
) -> None:
    """Telegram'a sinyal bildirimi gönderir (HTML formatı)."""
    now = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M ET")
    atr_val  = tech_data.get("atr") or 0
    price    = tech_data.get("price", 0)
    sl_price = price - ATR_SL_MULT * atr_val
    tp_price = price + ATR_TP_MULT * atr_val
    tech_s   = tech_data["tech_score"]

    ml_status = "🟢 Aktif" if ml_model is not None else "🔴 Yüklenmedi (nötr=50)"

    text = (
        f"<b>{signal} — {ticker}</b>\n"
        f"⏰ {now} | Mod: {mod}\n"
        f"\n"
        f"<b>📊 Final Skor:</b> {final:.1f} / 100\n"
        f"<b>🎯 Güven:</b> {conf}\n"
        f"\n"
        f"<b>🤖 ML Analiz ({ml_status}):</b>\n"
        f"  ML Tahmini (yükseliş iht.) : {ml_score:.1f}/100\n"
        f"  Teknik Skor                : {tech_s:.1f}/100\n"
        f"  Final = Teknik×0.4 + ML×0.6 : {tech_s*0.4:.1f} + {ml_score*0.6:.1f} = {final:.1f}\n"
        f"\n"
        f"<b>Teknik ({tech_s:.1f}):</b>\n"
        f"  Günlük filtre : {tech_data['daily_score']:.0f}\n"
        f"  RSI(14)       : {tech_data['rsi_val']:.1f}\n"
        f"  MACD yön      : {tech_data['macd_dir']}\n"
        f"  Rejim         : {tech_data.get('regime', '—')} (ADX={tech_data.get('adx', 0):.1f} / H={tech_data.get('hurst', 0):.2f})\n"
        f"  Fiyat         : ${price:.2f}\n"
        f"  EMA200        : ${tech_data['ema200']:.2f}\n"
        f"\n"
        f"<b>Risk (ATR={atr_val:.2f}):</b>\n"
        f"  Stop Loss     : ${sl_price:.2f}\n"
        f"  Take Profit   : ${tp_price:.2f}\n"
        f"\n"
        f"<b>Sentiment ({sentiment_s:.1f}):</b>\n"
        f"  Reddit (ApeWisdom) : {reddit_s:.1f}\n"
        f"  Finnhub            : {finnhub_s:.1f}\n"
        f"\n"
        f"<b>Alternatif Veri:</b>\n"
        f"  Insider            : {insider_s:.1f}\n"
        f"  LLM                : {f'{llm_s:.1f}' if llm_s is not None else 'devre dışı'}\n"
        f"\n"
        f"<b>İşlem:</b> {action_msg}"
    )
    tg_send(text)


# ─── Telegram Bot (Asenkron) ──────────────────────────────────────────────────
def start_telegram_bot(engine: AlpacaEngine) -> None:
    """
    Telegram komut dinleyicisini ayrı bir thread'de asyncio event loop ile başlatır.
    Ana trading döngüsünü bloklamaz.
    """
    if not TG_TOKEN:
        log.warning("TELEGRAM_BOT_TOKEN tanımlı değil, Telegram botu başlatılmadı.")
        return

    async def cmd_durum(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            account = engine.get_account()
            positions = engine.get_positions()
            text = (
                f"<b>📈 Bot Durumu</b>\n"
                f"Mod       : {engine.mod}\n"
                f"Durum     : {'⏸ Duraklatıldı' if BOT_PAUSED else '▶️ Çalışıyor'}\n"
                f"Portföy   : ${float(account.portfolio_value):.2f}\n"
                f"Nakit     : ${float(account.buying_power):.2f}\n"
                f"Pozisyon  : {len(positions)}/{MAX_POSITIONS}"
            )
        except Exception as e:
            text = f"Alpaca hatası: {e}"
        await update.message.reply_html(text)

    async def cmd_portfoy(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            positions = engine.get_positions()
            if not positions:
                await update.message.reply_text("Açık pozisyon yok.")
                return
            lines = ["<b>📂 Açık Pozisyonlar</b>"]
            for p in positions:
                entry  = float(p.avg_entry_price)
                market = float(p.market_value)
                pnl    = float(p.unrealized_pl)
                lines.append(
                    f"\n<b>{p.symbol}</b>\n"
                    f"  Giriş : ${entry:.2f}\n"
                    f"  Değer : ${market:.2f}\n"
                    f"  K/Z   : ${pnl:+.2f}"
                )
            await update.message.reply_html("\n".join(lines))
        except Exception as e:
            await update.message.reply_text(f"Hata: {e}")

    async def cmd_durdur(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        global BOT_PAUSED
        BOT_PAUSED = True
        log.info("Bot Telegram komutu ile duraklatıldı.")
        await update.message.reply_text("⏸ Bot duraklatıldı. /baslat ile devam edebilirsiniz.")

    async def cmd_baslat(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        global BOT_PAUSED
        BOT_PAUSED = False
        log.info("Bot Telegram komutu ile başlatıldı.")
        await update.message.reply_text("▶️ Bot yeniden başlatıldı.")

    def thread_main():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def on_error(update, context):
            if isinstance(context.error, Conflict):
                log.warning("Telegram 409 Conflict: başka bir instance çalışıyor, yeniden deneniyor...")
            else:
                log.error(f"Telegram hatası: {context.error}")

        app = ApplicationBuilder().token(TG_TOKEN).build()
        app.add_handler(CommandHandler("durum",   cmd_durum))
        app.add_handler(CommandHandler("portfoy", cmd_portfoy))
        app.add_handler(CommandHandler("durdur",  cmd_durdur))
        app.add_handler(CommandHandler("baslat",  cmd_baslat))
        app.add_error_handler(on_error)

        log.info("Telegram bot dinlemeye başladı.")
        app.run_polling(stop_signals=None, drop_pending_updates=True)

    t = threading.Thread(target=thread_main, daemon=True)
    t.start()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true",
                        help="Tek tarama yap ve çık (GitHub Actions modu)")
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("  Kanka Sinyal Botu v4.0 başlatılıyor...")
    log.info(f"  Mod: {'--once (GitHub Actions)' if args.once else 'sürekli (VPS)'}")
    log.info("=" * 55)

    # API key kontrolü
    for key, val in [("ALPACA_API_KEY", ALPACA_API_KEY), ("ALPACA_SECRET_KEY", ALPACA_SECRET_KEY)]:
        if val in ("", "BURAYA_YAZ"):
            log.error(f"{key} tanımlı değil.")
            sys.exit(1)

    # ── ML Modeli Yükle ───────────────────────────────────────────────────────
    global ml_model
    try:
        ml_model = joblib.load("kanka_model.joblib")
        log.info("✅ ML modeli yüklendi: kanka_model.joblib")
    except Exception as e:
        log.warning(f"⚠️ ML modeli yüklenemedi: {e} — nötr skor (50) kullanılacak")
        ml_model = None

    engine = AlpacaEngine()

    try:
        account = engine.get_account()
        log.info(
            f"Alpaca bağlantısı OK | Mod: {engine.mod} | "
            f"Portföy: ${float(account.portfolio_value):.2f}"
        )
    except Exception as e:
        log.error(f"Alpaca bağlantısı başarısız: {e}")
        sys.exit(1)

    # GitHub Actions modunda Telegram polling başlatılmaz
    # (tg_send() ile bildirimler yine de gider)
    if not args.once:
        start_telegram_bot(engine)
        tg_send(
            f"🚀 <b>Kanka Sinyal Botu v4.0 başlatıldı</b>\n"
            f"Mod: {engine.mod}\n"
            f"Hisseler: {', '.join(TICKERS)}\n"
            f"Tarama aralığı: {SCAN_INTERVAL} dakika"
        )

    if args.once:
        # ── GitHub Actions modu: tek tarama, çık ──────────────────────────
        log.info("GitHub Actions modu: tek tarama başlıyor...")
        try:
            scan_once(engine)
        except Exception as e:
            log.error(f"Tarama hatası: {e}")
            tg_send(f"⚠️ GitHub Actions tarama hatası: {e}")
            sys.exit(1)
        log.info("Tarama tamamlandı, çıkılıyor.")
    else:
        # ── VPS modu: sonsuz döngü ─────────────────────────────────────────
        log.info(f"Tarama aralığı: {SCAN_INTERVAL} dakika")
        try:
            while True:
                try:
                    scan_once(engine)
                except Exception as e:
                    log.error(f"Beklenmedik hata: {e}")
                    tg_send(f"⚠️ Bot hatası: {e}\n5 dakika bekleniyor...")
                    time.sleep(300)
                    continue

                log.info(f"Sonraki tarama {SCAN_INTERVAL} dakika sonra...")
                time.sleep(SCAN_INTERVAL * 60)

        except KeyboardInterrupt:
            log.info("Bot kullanıcı tarafından durduruldu.")
            tg_send("🛑 Kanka Sinyal Botu durduruldu.")


if __name__ == "__main__":
    main()
