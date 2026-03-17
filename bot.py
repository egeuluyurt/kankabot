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
    StopLossRequest,
    TakeProfitRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, OrderType

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
TICKERS           = [t.strip() for t in os.getenv("TICKERS", "AAPL,TSLA,NVDA,SPY,QQQ").split(",")]
BUY_THRESHOLD     = float(os.getenv("BUY_THRESHOLD", "65"))
SELL_THRESHOLD    = float(os.getenv("SELL_THRESHOLD", "35"))
POSITION_PCT      = float(os.getenv("POSITION_PCT", "5"))
MAX_POSITIONS     = int(os.getenv("MAX_POSITIONS", "3"))
ATR_SL_MULT       = float(os.getenv("ATR_SL_MULT", "1.5"))
ATR_TP_MULT       = float(os.getenv("ATR_TP_MULT", "3.0"))
SCAN_INTERVAL     = int(os.getenv("SCAN_INTERVAL_MINUTES", "60"))

# ─── Global durum ─────────────────────────────────────────────────────────────
BOT_PAUSED = False
vader = SentimentIntensityAnalyzer()

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
        self.mod = "PAPER" if PAPER_TRADING else "LIVE"

    @retry_on_rate_limit
    def get_account(self):
        return self.client.get_account()

    @retry_on_rate_limit
    def get_positions(self):
        return self.client.get_all_positions()

    @retry_on_rate_limit
    def place_bracket_buy(self, symbol: str, notional: float, price: float, sl_price: float, tp_price: float):
        """
        Bracket market order — borsa sunucusu SL/TP'yi yönetir.
        Alpaca bracket order notional desteklemediği için qty hesaplanır:
          qty = notional / güncel_fiyat (2 ondalık basamak)
        """
        qty = max(1, int(notional / price))  # Bracket order tam sayı gerektiriyor
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=round(sl_price, 2)),
            take_profit=TakeProfitRequest(limit_price=round(tp_price, 2)),
        )
        return self.client.submit_order(req)

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
        daily_bull = price > ema200_v
        daily_bear = not daily_bull
        hourly_bull = rsi_val > 60 and macd_score > 60
        hourly_bear = rsi_val < 40 and macd_score < 40

        tech_score = daily_score * 0.40 + rsi_score * 0.30 + macd_score * 0.30

        if daily_bull and hourly_bear:
            tech_score *= 0.85
        elif daily_bear and hourly_bull:
            tech_score *= 0.85

        result["tech_score"] = tech_score

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
def composite_score(tech: float, reddit: float, finnhub: float) -> tuple[float, float, float]:
    """(final_score, tech_score, sentiment_score) döndürür."""
    sentiment = reddit * 0.50 + finnhub * 0.50
    final     = tech * 0.60 + sentiment * 0.40
    return final, tech, sentiment

def confidence_level(tech: float, sentiment: float) -> str:
    """İki skor aynı yönde ise YÜKSEK, aksi hâlde ORTA."""
    if (tech >= 60 and sentiment >= 60) or (tech <= 40 and sentiment <= 40):
        return "YÜKSEK"
    return "ORTA"

def signal_label(final: float, confidence: str) -> str:
    if final >= 80 and confidence == "YÜKSEK":
        return "💪 GÜÇLÜ AL"
    if final >= BUY_THRESHOLD:
        return "✅ AL"
    if final <= 20 and confidence == "YÜKSEK":
        return "🔴 GÜÇLÜ SAT"
    if final <= SELL_THRESHOLD:
        return "🔴 SAT"
    return "⏸ BEKLE"


# ─── Emir Yönetimi ────────────────────────────────────────────────────────────
def should_buy(
    ticker: str,
    positions: list,
    portfolio_value: float,
    buying_power: float,
    atr: Optional[float],
) -> tuple[bool, str]:
    """Alım koşullarını sırayla kontrol eder."""
    global BOT_PAUSED
    if BOT_PAUSED:
        return False, "Bot duraklatıldı"
    if not is_market_hours():
        return False, "Borsa kapalı"
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
) -> str:
    """Bracket buy emri gönderir. Sonucu string olarak döndürür."""
    notional   = portfolio_value * POSITION_PCT / 100
    sl_price   = price - ATR_SL_MULT * atr
    tp_price   = price + ATR_TP_MULT * atr

    try:
        order = engine.place_bracket_buy(ticker, notional, price, sl_price, tp_price)
        msg = (
            f"✅ Bracket Buy: {ticker} | "
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


# ─── Ana Tarama Döngüsü ───────────────────────────────────────────────────────
def scan_once(engine: AlpacaEngine) -> None:
    """Tüm hisseleri bir kez tarar."""
    global BOT_PAUSED
    if BOT_PAUSED:
        log.info("Bot duraklatıldı, tarama atlandı.")
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

    for ticker in TICKERS:
        try:
            log.info(f"─── {ticker} analiz ediliyor...")

            tech_data  = get_technical_score(ticker)
            reddit_s   = get_apewisdom_score(ticker)
            finnhub_s  = get_finnhub_score(ticker)

            final, tech_s, sentiment_s = composite_score(
                tech_data["tech_score"], reddit_s, finnhub_s
            )
            conf   = confidence_level(tech_s, sentiment_s)
            signal = signal_label(final, conf)

            log.info(
                f"{ticker} | Final={final:.1f} | Tech={tech_s:.1f} | "
                f"Sentiment={sentiment_s:.1f} | Güven={conf} | {signal}"
            )

            action_msg = "İşlem yapılmadı"

            if "AL" in signal and is_market_hours():
                ok, reason = should_buy(
                    ticker, positions, portfolio_val, buying_power, tech_data["atr"]
                )
                if ok:
                    action_msg = place_bracket_buy(
                        engine, ticker, portfolio_val,
                        tech_data["price"], tech_data["atr"]
                    )
                    # Pozisyonu listeye ekle (cache için sahte nesne)
                    active_tickers.add(ticker)
                else:
                    action_msg = f"Alım engellendi: {reason}"

            elif "SAT" in signal and ticker in active_tickers and is_market_hours():
                action_msg = place_sell(engine, ticker, portfolio_val)
                active_tickers.discard(ticker)

            # Telegram bildirimi (AL/SAT sinyallerinde)
            if signal != "⏸ BEKLE":
                _send_signal_message(
                    ticker, signal, final, conf, tech_data,
                    reddit_s, finnhub_s, sentiment_s, action_msg, engine.mod
                )

        except Exception as e:
            log.error(f"{ticker} tarama hatası: {e}")

        time.sleep(2)  # Rate limit koruması

    log.info("Tarama tamamlandı.")

def _send_signal_message(
    ticker, signal, final, conf, tech_data,
    reddit_s, finnhub_s, sentiment_s, action_msg, mod
) -> None:
    """Telegram'a sinyal bildirimi gönderir (HTML formatı)."""
    now = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M ET")
    atr_val  = tech_data.get("atr") or 0
    price    = tech_data.get("price", 0)
    sl_price = price - ATR_SL_MULT * atr_val
    tp_price = price + ATR_TP_MULT * atr_val

    text = (
        f"<b>{signal} — {ticker}</b>\n"
        f"⏰ {now} | Mod: {mod}\n"
        f"\n"
        f"<b>📊 Bileşik Skor:</b> {final:.1f} / 100\n"
        f"<b>🎯 Güven:</b> {conf}\n"
        f"\n"
        f"<b>Teknik ({tech_data['tech_score']:.1f}):</b>\n"
        f"  Günlük filtre : {tech_data['daily_score']:.0f}\n"
        f"  RSI(14)       : {tech_data['rsi_val']:.1f}\n"
        f"  MACD yön      : {tech_data['macd_dir']}\n"
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
