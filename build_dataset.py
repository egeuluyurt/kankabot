"""
Kanka Sinyal Botu v5.1 — Veri Fabrikası
=========================================
10 yıllık (2016–bugün) günlük OHLCV + teknik indikatörler + makro veriler.

Çıktı: kanka_training_data.csv
  - OHLCV, EMA50, EMA200, RSI, MACD, ATR, ADX, Hurst
  - Makro: VIX (VIXCLS), Faiz (DFF), Enflasyon (CPIAUCSL)
  - TARGET: 5 iş günü sonraki kapanış bugünden %2+ yüksekse 1, değilse 0

Çalıştırma:
  pip install fredapi
  FRED_API_KEY=xxx python build_dataset.py
"""

import os
import sys
import logging
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from dotenv import load_dotenv

from regime import calculate_adx, calculate_hurst

load_dotenv("config.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Parametreler ─────────────────────────────────────────────────────────────
TICKERS = ["NVDA", "AMD", "QQQ", "SMH", "AAPL", "MSFT", "TSLA", "PLTR", "SPY", "UNH"]
START_DATE   = "2016-01-01"
END_DATE     = datetime.today().strftime("%Y-%m-%d")
FORWARD_BARS = 5    # kaç iş günü sonrasına bakılacak
TARGET_PCT   = 2.0  # hedef yükseliş yüzdesi
OUTPUT_FILE  = "kanka_training_data.csv"

FRED_SERIES = {
    "vix":       "VIXCLS",    # VIX korku endeksi
    "fed_rate":  "DFF",       # Fed Funds Rate (günlük)
    "cpi":       "CPIAUCSL",  # CPI (aylık — merge_asof ile günlüğe eşlenir)
}


# ─── Makro Veri (FRED) ────────────────────────────────────────────────────────
def fetch_fred_series() -> pd.DataFrame:
    """
    fredapi ile VIXCLS, DFF, CPIAUCSL serilerini çeker.
    Döndürür: tarih indexli DataFrame (günlük frekans, ffill uygulanmış)
    """
    fred_key = os.getenv("FRED_API_KEY", "")
    if not fred_key:
        log.warning("FRED_API_KEY tanımlı değil — makro kolonlar NaN olacak")
        idx = pd.date_range(START_DATE, END_DATE, freq="B")
        return pd.DataFrame(index=idx, columns=list(FRED_SERIES.keys()))

    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_key)

        frames = {}
        for col, series_id in FRED_SERIES.items():
            log.info(f"FRED: {series_id} çekiliyor...")
            s = fred.get_series(series_id, observation_start=START_DATE,
                                observation_end=END_DATE)
            frames[col] = s.rename(col)

        macro = pd.concat(frames.values(), axis=1)
        macro.index = pd.to_datetime(macro.index)
        macro = macro.sort_index()

        # Günlük frekansa upsample et (merge_asof için hazırlık)
        daily_idx = pd.date_range(START_DATE, END_DATE, freq="D")
        macro = macro.reindex(daily_idx).ffill()

        log.info(f"FRED verisi: {len(macro)} satır, {macro.shape[1]} kolon")
        return macro

    except Exception as e:
        log.error(f"FRED verisi çekilemedi: {e}")
        idx = pd.date_range(START_DATE, END_DATE, freq="B")
        return pd.DataFrame(index=idx, columns=list(FRED_SERIES.keys()))


# ─── Tek Ticker İşleme ────────────────────────────────────────────────────────
def process_ticker(ticker: str, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Bir ticker için 10 yıllık özellik matrisi oluşturur.
    """
    log.info(f"{ticker} işleniyor...")

    # OHLCV indir
    t  = yf.Ticker(ticker)
    df = t.history(start=START_DATE, end=END_DATE, interval="1d", auto_adjust=True)
    if df.empty or len(df) < 200:
        log.warning(f"{ticker}: yetersiz veri ({len(df)} satır), atlandı")
        return pd.DataFrame()

    df.columns = [c.lower() for c in df.columns]
    df.index   = pd.to_datetime(df.index).tz_localize(None)
    df         = df[["open", "high", "low", "close", "volume"]].copy()

    close = df["close"]

    # ── Teknik indikatörler ───────────────────────────────────────────────────
    df["ema50"]  = close.ewm(span=50,  adjust=False, min_periods=0).mean()
    df["ema200"] = close.ewm(span=200, adjust=False, min_periods=0).mean()
    df["price_ema200_ratio"] = close / df["ema200"]

    df["rsi"] = ta.momentum.rsi(close, window=14)

    macd_ind     = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["macd"]      = macd_ind.macd()
    df["macd_hist"] = macd_ind.macd_diff()
    df["macd_above_signal"] = (macd_ind.macd() > macd_ind.macd_signal()).astype(int)

    atr_ind  = ta.volatility.AverageTrueRange(df["high"], df["low"], close, window=14)
    df["atr"]     = atr_ind.average_true_range()
    df["atr_pct"] = df["atr"] / close

    bb = ta.volatility.BollingerBands(close, window=20)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close

    # ── ADX + Hurst (rolling window) ─────────────────────────────────────────
    # Tüm satırlar üzerinde rolling 60 günlük pencere ile hesapla
    adx_vals   = []
    hurst_vals = []
    window     = 60

    for i in range(len(df)):
        start_i = max(0, i - window + 1)
        chunk   = df.iloc[start_i : i + 1]
        if len(chunk) < 20:
            adx_vals.append(np.nan)
            hurst_vals.append(np.nan)
        else:
            adx_vals.append(calculate_adx(chunk))
            hurst_vals.append(calculate_hurst(chunk["close"]))

    df["adx"]   = adx_vals
    df["hurst"] = hurst_vals

    # ── Günlük fiyat değişimi ─────────────────────────────────────────────────
    df["daily_return"] = close.pct_change()

    # ── TARGET: 5 iş günü sonrası %2+ yükseliş = 1 ───────────────────────────
    df["future_close"] = close.shift(-FORWARD_BARS)
    df["target"]       = ((df["future_close"] / close - 1) >= TARGET_PCT / 100).astype(int)

    # ── Makro veriyi merge_asof ile ekle ─────────────────────────────────────
    df = df.reset_index().rename(columns={"index": "date"})
    macro_reset = macro_df.reset_index().rename(columns={"index": "date"})
    macro_reset["date"] = pd.to_datetime(macro_reset["date"])
    df["date"] = pd.to_datetime(df["date"])

    df = pd.merge_asof(
        df.sort_values("date"),
        macro_reset.sort_values("date"),
        on="date",
        direction="backward",   # en yakın geçmiş veriyi al
    )

    df["ticker"] = ticker

    # ── Temizlik ──────────────────────────────────────────────────────────────
    # future_close NaN olan son FORWARD_BARS satırı düşür
    df = df.dropna(subset=["target", "rsi", "macd", "adx"])

    log.info(f"{ticker}: {len(df)} satır hazır")
    return df


# ─── Ana Akış ─────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 55)
    log.info("  Kanka Veri Fabrikası v5.1 başlıyor")
    log.info(f"  Periyot : {START_DATE} → {END_DATE}")
    log.info(f"  Tickers : {', '.join(TICKERS)}")
    log.info("=" * 55)

    # Makro veriyi bir kez çek
    macro_df = fetch_fred_series()

    all_frames = []
    for ticker in TICKERS:
        try:
            frame = process_ticker(ticker, macro_df)
            if not frame.empty:
                all_frames.append(frame)
        except Exception as e:
            log.error(f"{ticker} işleme hatası: {e}")

    if not all_frames:
        log.error("Hiç veri üretilemedi!")
        sys.exit(1)

    combined = pd.concat(all_frames, ignore_index=True)

    # Kolon sırası
    base_cols   = ["date", "ticker", "open", "high", "low", "close", "volume"]
    tech_cols   = ["ema50", "ema200", "price_ema200_ratio",
                   "rsi", "macd", "macd_hist", "macd_above_signal",
                   "atr", "atr_pct", "bb_width", "adx", "hurst", "daily_return"]
    macro_cols  = list(FRED_SERIES.keys())
    label_cols  = ["target"]
    ordered     = base_cols + tech_cols + macro_cols + label_cols
    existing    = [c for c in ordered if c in combined.columns]
    combined    = combined[existing]

    combined.to_csv(OUTPUT_FILE, index=False)

    log.info("=" * 55)
    log.info(f"  Çıktı dosyası : {OUTPUT_FILE}")
    log.info(f"  Toplam satır  : {len(combined):,}")
    log.info(f"  Kolon sayısı  : {len(combined.columns)}")
    log.info(f"  Target dağılımı:\n{combined['target'].value_counts().to_string()}")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
