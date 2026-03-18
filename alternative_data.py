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
from fredapi import Fred

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


# ─── FRED Makro Veri ─────────────────────────────────────────────────────────
def get_fred_macro_data() -> dict:
    """
    FRED API üzerinden makroekonomik göstergelerin en güncel değerlerini çeker.

    Seriler:
      VIXCLS   — VIX Volatilite Endeksi
      DFF      — Günlük Federal Fon Oranı (%)
      CPIAUCSL — Tüketici Fiyat Endeksi (Enflasyon)

    Döndürür: {'vix': float, 'rate': float, 'cpi': float}
    Hata durumunda varsayılan değerler: VIX=20, Faiz=5, Enflasyon=3
    """
    defaults = {"vix": 20.0, "rate": 5.0, "cpi": 3.0}
    fred_key = os.getenv("FRED_API_KEY", "")
    if not fred_key:
        log.info("[MACRO] FRED_API_KEY tanımlı değil — varsayılan değerler kullanılıyor")
        return defaults

    try:
        fred = Fred(api_key=fred_key)
        result = {}

        for col, series_id in [("vix", "VIXCLS"), ("rate", "DFF"), ("cpi", "CPIAUCSL")]:
            try:
                series = fred.get_series(series_id)
                value  = float(series.dropna().iloc[-1])
                result[col] = value
            except Exception as e:
                log.warning(f"[MACRO] {series_id} çekilemedi: {e} — varsayılan kullanılıyor")
                result[col] = defaults[col]

        log.info(
            f"[MACRO] VIX={result['vix']:.1f} | "
            f"Faiz={result['rate']:.2f}% | "
            f"Enflasyon={result['cpi']:.1f}"
        )
        return result

    except Exception as e:
        log.warning(f"[MACRO] FRED bağlantısı başarısız: {e} — varsayılan değerler kullanılıyor")
        return defaults


# ─── Insider Sentiment ────────────────────────────────────────────────────────
@_retry_alt
def get_insider_sentiment(ticker: str) -> float:
    """
    Finnhub /stock/insider-transactions — Cluster Buying mantığı.

    Sinyal gücü hacimden (share sayısı) değil, KAÇ FARKLI KİŞİNİN
    aynı yönde işlem yaptığından üretilir. Birden fazla yönetici aynı
    anda alım yapıyorsa bu güçlü bir 'cluster buying' sinyalidir.

    Skor:
      buyer_count  > seller_count  → >50 (cluster buy)
      buyer_count  < seller_count  → <50 (cluster sell)
      buyer_count == seller_count  → 50  (nötr)
      Veri yok                     → 50  (nötr)

    Döndürür: 0–100 float
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

        # Her benzersiz kişi (name) için işlem yönünü belirle
        buyer_names  = set()
        seller_names = set()
        for t in data:
            ttype = t.get("transactionType", "")
            name  = t.get("name", "").strip()
            if not name:
                continue
            if ttype == "P" or ttype.upper().startswith("P"):
                buyer_names.add(name)
            elif ttype == "S" or ttype.upper().startswith("S"):
                seller_names.add(name)

        buyer_count  = len(buyer_names)
        seller_count = len(seller_names)
        total        = buyer_count + seller_count

        if total == 0:
            return 50.0

        # Cluster oranı: alıcı kişi sayısının toplama oranı → 0–100
        score = (buyer_count / total) * 100

        log.info(
            f"{ticker} Insider Cluster: {buyer_count} alıcı kişi, "
            f"{seller_count} satıcı kişi → skor={score:.1f}"
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
