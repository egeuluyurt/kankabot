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
import time
import logging
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from dotenv import load_dotenv

from regime import calculate_hurst

load_dotenv("config.env")


# ─── Triple-Barrier Etiketleme ────────────────────────────────────────────────
def triple_barrier_label(
    df: pd.DataFrame,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    max_bars: int = 5,
) -> list:
    """
    Her satır için gerçek trade senaryosunu simüle eder.
    +1 = TP bariyerine önce çarptı
    -1 = SL bariyerine önce çarptı
     0 = max_bars doldu (nötr)
    Ham fiyat DROP'tan ÖNCE çağrılmalıdır.
    """
    if "atr" not in df.columns:
        raise ValueError("ATR kolonu eksik")

    closes = df["close"].values if "close" in df.columns else None

    raw_closes = df["close"].values if "close" in df.columns else \
                 (df["price_ema200_ratio"].values * df["ema200"].values
                  if "ema200" in df.columns else None)

    highs = df["high"].values if "high" in df.columns else raw_closes
    lows  = df["low"].values  if "low"  in df.columns else raw_closes
    atrs  = df["atr"].values

    labels = []
    for i in range(len(df) - max_bars):
        atr_i = atrs[i]
        if pd.isna(atr_i) or atr_i <= 0:
            labels.append(None)
            continue

        entry = raw_closes[i] if raw_closes is not None else 0
        sl    = entry - sl_mult * atr_i
        tp    = entry + tp_mult * atr_i
        result = 0

        for j in range(1, max_bars + 1):
            lo = lows[i + j]
            hi = highs[i + j] if highs is not None else lo

            sl_hit = lo <= sl
            tp_hit = hi >= tp

            if sl_hit and tp_hit:
                result = -1
                break
            elif tp_hit:
                result = 1
                break
            elif sl_hit:
                result = -1
                break

        labels.append(result)

    labels += [None] * max_bars
    return labels


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
        macro.index = pd.to_datetime(macro.index).tz_localize(None)
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

    # ── Sütun standartlaştırma ────────────────────────────────────────────────
    df = df.reset_index()
    df.rename(columns={
        "Date": "date", "Datetime": "date",
        "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    }, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("date").reset_index(drop=True)

    # ── float64 zorla + NaN temizle (ta kütüphanesi için hazırlık) ──────────────
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df = df.reset_index(drop=True)

    df = df.set_index("date")  # teknik hesaplar için index'e al
    close = df["close"]

    # ── Teknik indikatörler (.values ile index çakışması engellenir) ──────────
    df["ema50"]  = close.ewm(span=50,  adjust=False, min_periods=0).mean().values
    df["ema200"] = close.ewm(span=200, adjust=False, min_periods=0).mean().values

    df["rsi"] = ta.momentum.rsi(close, window=14).values

    macd_ind     = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["macd"]      = macd_ind.macd().values
    df["macd_hist"] = macd_ind.macd_diff().values
    df["macd_above_signal"] = (macd_ind.macd() > macd_ind.macd_signal()).astype(int).values

    atr_ind  = ta.volatility.AverageTrueRange(df["high"], df["low"], close, window=14)
    df["atr"]     = atr_ind.average_true_range().values
    df["atr_pct"] = (df["atr"] / close).values

    bb = ta.volatility.BollingerBands(close, window=20)
    df["bb_width"] = ((bb.bollinger_hband() - bb.bollinger_lband()) / close).values

    # ── ADX (ta library, tüm seri) ────────────────────────────────────────────
    if len(df) >= 15:
        adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
        df["adx"] = adx_ind.adx().values
    else:
        df["adx"] = float("nan")

    # ── Hurst (rolling 60 günlük pencere) ────────────────────────────────────
    hurst_vals = []
    window     = 60

    for i in range(len(df)):
        start_i = max(0, i - window + 1)
        chunk   = df.iloc[start_i : i + 1]
        if len(chunk) < 20:
            hurst_vals.append(np.nan)
        else:
            try:
                hurst_vals.append(calculate_hurst(chunk["close"]))
            except Exception:
                hurst_vals.append(0.5)

    df["hurst"] = hurst_vals

    # ── Stationary & türetilmiş özellikler ────────────────────────────────────
    df["price_ema50_ratio"]  = df["close"] / df["ema50"]
    df["price_ema200_ratio"] = df["close"] / df["ema200"]
    df["volume_zscore"]      = (
        df["volume"] - df["volume"].rolling(20).mean()
    ) / df["volume"].rolling(20).std()
    df["daily_return"]       = df["close"].pct_change()
    df["log_return"]         = np.log(df["close"] / df["close"].shift(1))

    # ── Triple-Barrier Etiketleme (ham fiyat DROP'tan önce) ───────────────────
    df["target"] = triple_barrier_label(df, sl_mult=1.5, tp_mult=3.0, max_bars=5)
    df = df.dropna(subset=["target", "atr"])
    df["target"] = df["target"].astype(int)

    # ── Ham fiyat ve EMA kolonlarını kaldır (stationarity) ────────────────────
    df.drop(columns=["open", "high", "low", "close", "volume",
                      "ema50", "ema200"], inplace=True, errors="ignore")

    # ── Makro veriyi merge_asof ile ekle (Düzeltme 3: tarih format eşitleme) ──
    df = df.reset_index()                                  # date index → sütun
    df.rename(columns={"index": "date"}, inplace=True)     # yedek rename
    df["date"] = pd.to_datetime(df["date"]).dt.normalize() # tz ve saat kaldır
    df = df.sort_values("date").reset_index(drop=True)

    macro_reset = macro_df.reset_index()
    macro_reset.rename(columns={"index": "date"}, inplace=True)
    macro_reset["date"] = pd.to_datetime(macro_reset["date"]).dt.normalize()
    macro_reset = macro_reset.sort_values("date").reset_index(drop=True)

    df = pd.merge_asof(
        df,
        macro_reset,
        on="date",
        direction="backward",   # en yakın geçmiş veriyi al
    )

    df["ticker"] = ticker

    # ── Temizlik ──────────────────────────────────────────────────────────────
    # future_close NaN olan son FORWARD_BARS satırı düşür
    df = df.dropna(subset=["rsi", "macd", "adx"])

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
        time.sleep(2)

    if not all_frames:
        log.error("Hiç veri üretilemedi!")
        sys.exit(1)

    combined = pd.concat(all_frames, ignore_index=True)

    # Kolon sırası
    base_cols   = ["date", "ticker"]
    tech_cols   = ["price_ema50_ratio", "price_ema200_ratio",
                   "rsi", "macd", "macd_hist", "macd_above_signal",
                   "atr", "atr_pct", "bb_width", "adx", "hurst",
                   "daily_return", "log_return", "volume_zscore"]
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
