"""
Piyasa Rejimi Tespiti — Kanka Sinyal Botu v5.0
================================================
ADX + Hurst Exponent kombinasyonu ile trend/ranging filtresi.

Mantık:
  ADX > 25 ve Hurst > 0.55 → TREND   (güçlü, yön var)
  ADX < 20 ve Hurst < 0.45 → RANGING (yatay, yanlış sinyal riski)
  Çelişki varsa ADX'i esas al (daha hızlı tepki verir)
"""

import logging

import numpy as np
import pandas as pd
import ta

log = logging.getLogger(__name__)


class MarketRegime:
    TRENDING = "TREND"
    RANGING  = "RANGING"
    UNKNOWN  = "UNKNOWN"


def calculate_adx(daily_df: pd.DataFrame, period: int = 14) -> float:
    """
    ta kütüphanesi ile ADX hesaplar.
    Veri yetersizse veya hata olursa nötr değer (25.0) döner.
    """
    try:
        adx_ind    = ta.trend.ADXIndicator(
            daily_df["high"], daily_df["low"], daily_df["close"], window=period
        )
        adx_series = adx_ind.adx()
        if adx_series is None or adx_series.dropna().empty:
            return 25.0
        val = float(adx_series.iloc[-1])
        return val if not np.isnan(val) else 25.0
    except Exception as e:
        log.warning(f"ADX hesaplama hatası: {e}")
        return 25.0


def calculate_hurst(price_series: pd.Series, lags_range: int = 20) -> float:
    """
    Rescaled Range (R/S) analizi ile Hurst Exponent hesaplar.
    Dış kütüphane gerektirmez.

    H < 0.5 → mean-reverting / yatay piyasa
    H = 0.5 → rastgele yürüyüş
    H > 0.5 → trend / ısrarcı hareket
    """
    try:
        prices = price_series.dropna().values
        if len(prices) < 50:
            return 0.5  # Yetersiz veri → nötr

        lags = range(2, lags_range)
        tau  = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
        tau  = [max(t, 1e-10) for t in tau]  # sıfır varyans koruması

        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        val  = float(poly[0])
        return val if not np.isnan(val) else 0.5
    except Exception as e:
        log.warning(f"Hurst hesaplama hatası: {e}")
        return 0.5


def detect_regime(daily_df: pd.DataFrame) -> tuple:
    """
    Günlük OHLC verisi üzerinden piyasa rejimini tespit eder.

    Parametreler:
      daily_df : 'high', 'low', 'close' sütunları olan günlük DataFrame

    Döndürür:
      (regime: str, details: dict)
        regime  → MarketRegime.TRENDING | RANGING | UNKNOWN
        details → {'adx': float, 'hurst': float}
    """
    adx   = calculate_adx(daily_df)
    hurst = calculate_hurst(daily_df["close"])

    details = {"adx": round(adx, 2), "hurst": round(hurst, 3)}

    # İkisi de trend → kesin TREND
    if adx > 25 and hurst > 0.55:
        return MarketRegime.TRENDING, details

    # İkisi de ranging → kesin RANGING
    if adx < 20 and hurst < 0.45:
        return MarketRegime.RANGING, details

    # Çelişki → ADX'i esas al (daha hızlı tepki)
    if adx > 25:
        return MarketRegime.TRENDING, details
    if adx < 20:
        return MarketRegime.RANGING, details

    return MarketRegime.UNKNOWN, details
