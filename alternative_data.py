"""
Alternatif Veri Kaynakları — Kanka Sinyal Botu v5.0
=====================================================
- get_insider_sentiment()      : Finnhub insider alım/satım dengesi
- get_economic_calendar()      : Yüksek etkili ekonomik olay riski
- get_llm_sentiment_analysis() : LLM tabanlı haber sentiment (opsiyonel)

Tüm fonksiyonlar hata durumunda nötr değer döner — bot çalışmaya devam eder.
Env var'lar fonksiyon çağrısında okunur (load_dotenv() sıralaması nedeniyle).
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

log = logging.getLogger(__name__)

# Yüksek etkili ekonomik olay anahtar kelimeleri
HIGH_IMPACT_KEYWORDS = [
    "Federal Reserve", "FOMC", "Fed Rate", "Interest Rate Decision",
    "CPI", "Consumer Price Index", "Non-Farm Payrolls", "NFP",
    "GDP", "Gross Domestic Product", "Unemployment Rate", "PCE",
    "Personal Consumption", "Producer Price", "PPI",
]

# ─── Tenacity retry (HTTP 429) ────────────────────────────────────────────────
def _is_rate_limit(exc: Exception) -> bool:
    return isinstance(exc, requests.HTTPError) and \
           exc.response is not None and exc.response.status_code == 429

_retry_alt = retry(
    retry=retry_if_exception_type(requests.HTTPError),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)


# ─── Insider Sentiment ────────────────────────────────────────────────────────
@_retry_alt
def get_insider_sentiment(ticker: str) -> float:
    """
    Finnhub /stock/insider-transactions üzerinden içeriden alım/satım dengesi.

    Döndürür: 0–100 skor
      > 60 → net içeriden alım (pozitif sinyal)
      = 50 → nötr / veri yok
      < 40 → net içeriden satış (negatif sinyal)
    """
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        return 50.0

    try:
        url  = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={ticker}"
        resp = requests.get(url, headers={"X-Finnhub-Token": api_key}, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])

        if not data:
            log.info(f"{ticker} insider: veri yok → nötr (50)")
            return 50.0

        buy_shares  = 0.0
        sell_shares = 0.0
        for t in data:
            ttype  = t.get("transactionType", "")
            shares = abs(float(t.get("share", 0) or 0))
            # "P" = Purchase, "S" = Sale (Finnhub formatı)
            if ttype == "P" or ttype.upper().startswith("P"):
                buy_shares += shares
            elif ttype == "S" or ttype.upper().startswith("S"):
                sell_shares += shares

        total = buy_shares + sell_shares
        if total == 0:
            return 50.0

        score = (buy_shares / total) * 100
        log.info(
            f"{ticker} Insider: alım={buy_shares:.0f} hisse, "
            f"satış={sell_shares:.0f} hisse → skor={score:.1f}"
        )
        return round(score, 1)

    except Exception as e:
        log.warning(f"{ticker} insider sentiment hatası: {e}")
        return 50.0


# ─── Ekonomik Takvim ──────────────────────────────────────────────────────────
def get_economic_calendar() -> tuple:
    """
    Bugün yüksek etkili ekonomik olay var mı? (Fed, CPI, NFP vb.)

    Döndürür: (risk_flag: bool, event_name: str)
      risk_flag=True  → yeni pozisyon açma!
      risk_flag=False → normal işlem
    """
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        return False, ""

    try:
        today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        url   = f"https://finnhub.io/api/v1/calendar/economic?from={today}&to={today}"
        resp  = requests.get(
            url, headers={"X-Finnhub-Token": api_key}, timeout=10
        )
        resp.raise_for_status()
        events = resp.json().get("economicCalendar", [])

        for event in events:
            name   = event.get("event", "")
            impact = event.get("impact", "low")
            if impact == "high" or any(
                kw.lower() in name.lower() for kw in HIGH_IMPACT_KEYWORDS
            ):
                log.warning(f"⚠️ Yüksek etkili ekonomik olay tespit edildi: {name}")
                return True, name

        return False, ""

    except Exception as e:
        log.warning(f"Ekonomik takvim kontrolü başarısız: {e}")
        return False, ""


# ─── LLM Sentiment ───────────────────────────────────────────────────────────
def get_llm_sentiment_analysis(ticker: str) -> Optional[float]:
    """
    Haber başlıklarını LLM ile analiz eder (opsiyonel).

    Öncelik sırası:
      1. GEMINI_API_KEY varsa → Google Gemini 1.5 Flash (ücretsiz tier mevcut)
      2. OPENAI_API_KEY varsa → OpenAI gpt-4o-mini
      3. İkisi de yoksa     → None (composite_score insider-only çalışır)

    Döndürür: 0–100 float veya None
    """
    gemini_key  = os.getenv("GEMINI_API_KEY", "")
    openai_key  = os.getenv("OPENAI_API_KEY", "")
    finnhub_key = os.getenv("FINNHUB_API_KEY", "")

    if not gemini_key and not openai_key:
        return None  # LLM devre dışı

    # Finnhub'dan son 7 günün haber başlıklarını çek
    headlines: list = []
    if finnhub_key:
        try:
            today    = datetime.now(ZoneInfo("America/New_York"))
            week_ago = today - timedelta(days=7)
            url = (
                f"https://finnhub.io/api/v1/company-news?symbol={ticker}"
                f"&from={week_ago.strftime('%Y-%m-%d')}"
                f"&to={today.strftime('%Y-%m-%d')}"
            )
            resp = requests.get(
                url, headers={"X-Finnhub-Token": finnhub_key}, timeout=10
            )
            resp.raise_for_status()
            news      = resp.json()
            headlines = [
                a.get("headline", "") for a in news[:10] if a.get("headline")
            ]
        except Exception as e:
            log.warning(f"{ticker} LLM için haber başlıkları çekilemedi: {e}")

    if not headlines:
        return None

    headlines_text = "\n".join(f"- {h}" for h in headlines)
    prompt = (
        f"{ticker} hissesi için aşağıdaki haber başlıklarını yatırımcı "
        f"duyarlılığı açısından analiz et. "
        f"0 (çok negatif) ile 100 (çok pozitif) arasında SADECE bir tam sayı "
        f"döndür, başka hiçbir şey yazma.\n\n{headlines_text}"
    )

    # ── 1. Gemini 1.5 Flash (öncelikli) ──────────────────────────────────────
    if gemini_key:
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models"
                f"/gemini-1.5-flash:generateContent?key={gemini_key}"
            )
            resp = requests.post(
                url,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=15,
            )
            resp.raise_for_status()
            raw   = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            score = float(raw)
            score = max(0.0, min(score, 100.0))
            log.info(f"{ticker} Gemini sentiment: {score:.0f} ({len(headlines)} haber)")
            return score
        except Exception as e:
            log.warning(f"{ticker} Gemini sentiment hatası: {e} — OpenAI'ya düşülüyor")

    # ── 2. OpenAI gpt-4o-mini (fallback) ─────────────────────────────────────
    if openai_key:
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 5,
                    "temperature": 0,
                },
                timeout=15,
            )
            resp.raise_for_status()
            raw   = resp.json()["choices"][0]["message"]["content"].strip()
            score = float(raw)
            score = max(0.0, min(score, 100.0))
            log.info(f"{ticker} OpenAI sentiment: {score:.0f} ({len(headlines)} haber)")
            return score
        except Exception as e:
            log.warning(f"{ticker} OpenAI sentiment hatası: {e}")

    return None
