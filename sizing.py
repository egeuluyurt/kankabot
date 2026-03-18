"""
Dinamik Pozisyon Boyutlandırma — Kanka Sinyal Botu v5.0
=========================================================
Fractional Kelly Criterion:
  Kelly% = W - (1 - W) / R
    W = kazanma olasılığı (backtesting win_rate)
    R = TP/SL oranı (ATR_TP_MULT / ATR_SL_MULT = 3.0/1.5 = 2.0)

1/4 Kelly: beklenen getiriyi ~%20 azaltır, varyansı %80 düşürür.
Sinyal gücü ve volatilite çarpanları ek kalibre sağlar.
"""

import logging

log = logging.getLogger(__name__)


def calculate_position_size(
    portfolio_value: float,
    final_score: float,
    atr: float,
    price: float,
    win_rate: float = 0.52,        # Backtesting win_rate; ölçüm yapıldıkça güncelle
    kelly_fraction: float = 0.25,  # 1/4 Kelly — güvenli başlangıç noktası
    tp_sl_ratio: float = 2.0,      # ATR_TP_MULT / ATR_SL_MULT (3.0 / 1.5)
    max_pct: float = 10.0,         # Portföyün max %10'u
    min_pct: float = 1.0,          # Portföyün min %1'i
    macro_multiplier: float = 1.0, # Makro risk çarpanı (örn. VIX>30 → 0.5)
) -> float:
    """
    Sinyal gücü + volatilite + makro risk bazlı dinamik pozisyon büyüklüğü.

    Parametreler:
      portfolio_value  : Alpaca account.portfolio_value
      final_score      : composite_score() çıktısı (0–100)
      atr              : Günlük ATR(14)
      price            : Anlık fiyat (mid-point veya teknik analiz)
      win_rate         : Geçmiş işlemlerden ölçülen kazanma oranı
      kelly_fraction   : Tam Kelly'nin kaçta biri kullanılacak (0.25 önerilen)
      tp_sl_ratio      : Take-profit / stop-loss mesafe oranı
      max_pct          : Portföyün maksimum yüzdesi
      min_pct          : Portföyün minimum yüzdesi
      macro_multiplier : Makro ortam çarpanı — VIX>30 ise 0.5, normal ise 1.0

    Döndürür:
      notional (float) — dolar cinsinden pozisyon büyüklüğü
    """
    # ATR veya fiyat verisi yoksa sabit %3 ile fallback
    if atr is None or atr <= 0 or price <= 0:
        notional = portfolio_value * 3.0 / 100 * macro_multiplier
        log.info(f"ATR/fiyat verisi yok — sabit %3 × macro={macro_multiplier:.1f}: ${notional:.0f}")
        return notional

    # ── Kelly formülü ────────────────────────────────────────────────────────
    raw_kelly = win_rate - (1 - win_rate) / tp_sl_ratio
    raw_kelly = max(0.0, raw_kelly)  # Negatif Kelly = işlem açma

    # ── Sinyal gücü çarpanı: final_score 65–100 → 0.1–1.0 ───────────────────
    signal_strength = (final_score - 65) / 35
    signal_strength = max(0.1, min(signal_strength, 1.0))

    # ── Volatilite cezası: ATR/fiyat > %3 ise pozisyonu küçült ──────────────
    atr_pct = atr / price
    vol_penalty = min(0.03 / atr_pct, 1.0) if atr_pct > 0.03 else 1.0

    # ── Efektif Kelly ────────────────────────────────────────────────────────
    effective_kelly = raw_kelly * kelly_fraction * signal_strength * vol_penalty

    # ── Portföy yüzdesine dönüştür ve sınırla ───────────────────────────────
    position_pct = effective_kelly * 100
    position_pct = max(min_pct, min(position_pct, max_pct))

    notional = portfolio_value * position_pct / 100

    # ── Makro çarpanı uygula ─────────────────────────────────────────────────
    notional = notional * macro_multiplier

    log.info(
        f"Kelly boyutlandırma: raw_kelly={raw_kelly:.3f} | "
        f"signal={signal_strength:.2f} | vol_penalty={vol_penalty:.2f} | "
        f"macro={macro_multiplier:.1f} | "
        f"effective={effective_kelly:.3f} → %{position_pct:.1f} × {macro_multiplier:.1f} = ${notional:.0f}"
    )
    return notional
