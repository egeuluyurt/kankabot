"""
╔══════════════════════════════════════════════════════════════╗
║         KANKA SİNYAL BOTU v3.0 — TAM ENTEGRASYON            ║
║  MTF Teknik Analiz (1H + 1D) + VADER Sentiment               ║
║  Reddit + Finnhub (Twitter devre dışı — kota sorunu)         ║
║  GitHub Actions ücretsiz bulut · Telegram Bildirim           ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, time, logging, json, requests
from datetime import datetime, date, timedelta
from pathlib import Path

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# ── Ortam değişkenleri ───────────────────────────────────────
load_dotenv("config.env")

REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USERNAME      = os.getenv("REDDIT_USERNAME", "")
REDDIT_PASSWORD      = os.getenv("REDDIT_PASSWORD", "")
FINNHUB_API_KEY      = os.getenv("FINNHUB_API_KEY", "")
TELEGRAM_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID", "")
TICKERS              = os.getenv("TICKERS", "AAPL,TSLA,NVDA,SPY,QQQ").split(",")
BUY_THRESHOLD        = float(os.getenv("BUY_THRESHOLD", 65))
SELL_THRESHOLD       = float(os.getenv("SELL_THRESHOLD", 35))

# ── Ağırlıklar ───────────────────────────────────────────────
W_TECHNICAL  = 0.60   # Teknik analiz toplam payı
W_SENTIMENT  = 0.40   # Sentiment toplam payı

# Teknik alt ağırlıklar (toplam 1.0)
W_DAILY_TREND  = 0.40  # Günlük EMA200 yönü (ana filtre)
W_HOURLY_RSI   = 0.30  # Saatlik RSI (taktiksel giriş)
W_HOURLY_MACD  = 0.30  # Saatlik MACD (momentum onayı)

# Sentiment alt ağırlıklar (Twitter yok → Reddit+Finnhub)
W_REDDIT   = 0.50
W_FINNHUB  = 0.50

# ── Loglama ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_log.txt", encoding="utf-8"),
    ],
)
log = logging.getLogger("KankaBot")

# ── VADER Sentiment Analyzer ─────────────────────────────────
vader = SentimentIntensityAnalyzer()


# ════════════════════════════════════════════════════════════
#  VADER SENTIMENT  (basit keyword yerine bağlam anlayan NLP)
# ════════════════════════════════════════════════════════════
def vader_score(text: str) -> float:
    """
    VADER compound skoru: -1.0 (çok negatif) → +1.0 (çok pozitif)
    'Not a good buy' → negatif olarak değerlendirilir ✓
    'BULLISH!!!'     → çok pozitif olarak değerlendirilir ✓
    """
    if not text or len(text.strip()) < 3:
        return 0.0
    return vader.polarity_scores(text)["compound"]


def compound_to_100(compound: float) -> float:
    """VADER -1..+1 → 0..100 dönüşümü."""
    return round((compound + 1) / 2 * 100, 1)


# ════════════════════════════════════════════════════════════
#  TEKNİK ANALİZ  —  MTF  (Saatlik + Günlük Kesişim)
# ════════════════════════════════════════════════════════════
def get_technical_score(ticker: str) -> dict:
    """
    ÇOK ZAMANLI DİLİM (MTF) ANALİZİ:

    Adım 1 — GÜNLÜK FİLTRE:
      Fiyat > EMA200(günlük) → Yükseliş bölgesi (alım sinyali mümkün)
      Fiyat < EMA200(günlük) → Düşüş bölgesi (satım sinyali mümkün)

    Adım 2 — SAATLİK TETİKLEYİCİ:
      RSI(14, 1H): 40'ın altından dönüş → Giriş sinyali
      MACD(12,26,9, 1H): Histogram > 0  → Momentum onayı

    Sadece iki zaman dilimi hemfikirse güçlü sinyal üretilir.
    """
    try:
        # ── Günlük veri (6 ay, EMA200 için yeterli) ──────────
        log.info(f"  [{ticker}] Günlük veri çekiliyor...")
        df_daily = yf.download(
            ticker, period="8mo", interval="1d",
            auto_adjust=True, progress=False
        )
        # Saatlik veri (60 gün, yfinance limiti: 730 gün)
        log.info(f"  [{ticker}] Saatlik veri çekiliyor...")
        df_hourly = yf.download(
            ticker, period="60d", interval="1h",
            auto_adjust=True, progress=False
        )

        if df_daily.empty or len(df_daily) < 50:
            log.warning(f"  [{ticker}] Yeterli günlük veri yok!")
            return _neutral_technical()
        if df_hourly.empty or len(df_hourly) < 20:
            log.warning(f"  [{ticker}] Yeterli saatlik veri yok!")
            return _neutral_technical()

        # MultiIndex kolon düzleştirme
        for df in [df_daily, df_hourly]:
            if hasattr(df.columns, "levels"):
                df.columns = df.columns.get_level_values(0)

        close_d = df_daily["Close"]
        close_h = df_hourly["Close"]
        price   = float(close_d.iloc[-1])

        # ──────────────────────────────────────────────────────
        # KATMAN 1: Günlük trend filtresi (EMA50 + EMA200)
        # ──────────────────────────────────────────────────────
        ema50_d  = float(ta.ema(close_d, length=50).iloc[-1])
        ema200_d = float(ta.ema(close_d, length=200).iloc[-1]) \
                   if len(close_d) >= 200 else ema50_d

        above_ema200 = price > ema200_d  # Ana yön filtresi

        if price > ema50_d > ema200_d:
            daily_trend_score = 100.0   # Güçlü boğa trendi
        elif price > ema200_d:
            daily_trend_score = 70.0    # Yükseliş bölgesi
        elif price < ema50_d < ema200_d:
            daily_trend_score = 0.0     # Güçlü ayı trendi
        elif price < ema200_d:
            daily_trend_score = 30.0    # Düşüş bölgesi
        else:
            daily_trend_score = 50.0    # Kararsız

        # ──────────────────────────────────────────────────────
        # KATMAN 2: Saatlik RSI tetikleyici
        # ──────────────────────────────────────────────────────
        rsi_h     = ta.rsi(close_h, length=14)
        rsi_val   = float(rsi_h.iloc[-1])
        rsi_prev  = float(rsi_h.iloc[-2])

        # Aşırı satımdan dönüş kontrolü (en güçlü giriş sinyali)
        rsi_recovering = rsi_val > rsi_prev and rsi_val < 50

        if rsi_val >= 70:
            hourly_rsi_score = 15.0         # Aşırı alım → sat baskısı
        elif rsi_val >= 55:
            hourly_rsi_score = 60.0 + (rsi_val - 55) * 2.0
        elif rsi_val <= 30:
            hourly_rsi_score = 85.0         # Aşırı satım → al fırsatı
            if rsi_recovering:
                hourly_rsi_score = 95.0     # Dipten dönüş → güçlü sinyal
        elif rsi_val <= 40 and rsi_recovering:
            hourly_rsi_score = 75.0         # 40'tan dönüş → giriş sinyali
        else:
            hourly_rsi_score = 40.0 + rsi_val * 0.3

        # ──────────────────────────────────────────────────────
        # KATMAN 3: Saatlik MACD momentum onayı
        # ──────────────────────────────────────────────────────
        macd_df   = ta.macd(close_h, fast=12, slow=26, signal=9)
        macd_line = float(macd_df["MACD_12_26_9"].iloc[-1])
        sig_line  = float(macd_df["MACDs_12_26_9"].iloc[-1])
        hist      = float(macd_df["MACDh_12_26_9"].iloc[-1])
        prev_hist = float(macd_df["MACDh_12_26_9"].iloc[-2])

        if macd_line > sig_line and hist > 0:
            hourly_macd_score = 100.0 if hist > prev_hist else 75.0
        elif macd_line > sig_line and hist < 0:
            hourly_macd_score = 55.0   # Pozitife geçiş yakın
        elif macd_line < sig_line and hist < 0:
            hourly_macd_score = 0.0 if abs(hist) > abs(prev_hist) else 25.0
        else:
            hourly_macd_score = 45.0

        # ──────────────────────────────────────────────────────
        # MTF KESİŞİM SKORU
        # ──────────────────────────────────────────────────────
        tech_score = (
            daily_trend_score  * W_DAILY_TREND +
            hourly_rsi_score   * W_HOURLY_RSI  +
            hourly_macd_score  * W_HOURLY_MACD
        )

        # MTF çelişki cezası: günlük ile saatlik zıt yönde ise skor zayıflatılır
        hourly_bullish = hourly_rsi_score > 60 and hourly_macd_score > 60
        hourly_bearish = hourly_rsi_score < 40 and hourly_macd_score < 40

        if above_ema200 and hourly_bearish:
            tech_score *= 0.85   # Günlük boğa ama saatlik düşüyor → dikkat
            log.info(f"  [{ticker}] MTF çelişki: günlük boğa / saatlik ayı → ceza uygulandı")
        elif not above_ema200 and hourly_bullish:
            tech_score *= 0.85   # Günlük ayı ama saatlik yukarı → dikkat
            log.info(f"  [{ticker}] MTF çelişki: günlük ayı / saatlik boğa → ceza uygulandı")

        log.info(
            f"  [{ticker}] Teknik: Günlük={daily_trend_score:.0f} "
            f"H-RSI={hourly_rsi_score:.0f} H-MACD={hourly_macd_score:.0f} "
            f"→ {tech_score:.1f}"
        )

        return {
            "score":             round(tech_score, 1),
            "daily_trend_score": round(daily_trend_score, 1),
            "hourly_rsi_score":  round(hourly_rsi_score, 1),
            "hourly_macd_score": round(hourly_macd_score, 1),
            "rsi_val":           round(rsi_val, 1),
            "above_ema200":      above_ema200,
            "price":             round(price, 2),
            "ema200_d":          round(ema200_d, 2),
            "source":            "yfinance 1.2.0 + pandas-ta",
        }

    except Exception as e:
        log.error(f"  [{ticker}] Teknik analiz hatası: {e}", exc_info=True)
        return _neutral_technical()


def _neutral_technical():
    return {
        "score": 50.0, "daily_trend_score": 50.0,
        "hourly_rsi_score": 50.0, "hourly_macd_score": 50.0,
        "rsi_val": 50.0, "above_ema200": None,
        "price": 0.0, "ema200_d": 0.0, "source": "hata",
    }


# ════════════════════════════════════════════════════════════
#  REDDIT SENTİMENT  (ApeWisdom API ile - KEY GEREKTİRMEZ)
# ════════════════════════════════════════════════════════════
def get_reddit_sentiment(ticker: str) -> dict:
    """
    ApeWisdom API kullanılarak tüm borsa subredditlerindeki (all-stocks)
    bahsedilme (mention) ve trend verisi çekilir.
    API Key gerektirmez, tamamen ücretsizdir.
    """
    try:
        # Genellikle AAPL, NVDA, TSLA gibi popüler hisseler ilk 2 sayfada (ilk 200) yer alır.
        # Bu yüzden ilk 2 sayfayı taramamız yeterlidir.
        for page in range(1, 3): 
            url = f"https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}"
            r = requests.get(url, timeout=10)
            
            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])
                
                for item in results:
                    if item.get("ticker") == ticker:
                        mentions = int(item.get("mentions", 0))
                        mentions_24h = int(item.get("mentions_24h_ago", 0))
                        
                        # Skor Hesaplama Mantığı (Momentum Odaklı):
                        # Hisse Reddit'te popülerse taban skor 55'ten başlar.
                        base_score = 55.0 
                        
                        if mentions > mentions_24h:
                            # Son 24 saatte ilgi artmışsa (Hype varsa) skoru yükselt (Max 95)
                            increase_ratio = (mentions - mentions_24h) / max(mentions_24h, 1)
                            bonus = min(40.0, increase_ratio * 25.0) 
                            final_score = base_score + bonus
                        else:
                            # İlgi azalıyorsa veya sabitse skoru düşür (Min 25)
                            decrease_ratio = (mentions_24h - mentions) / max(mentions_24h, 1)
                            penalty = min(30.0, decrease_ratio * 20.0)
                            final_score = base_score - penalty
                            
                        final_score = round(final_score, 1)
                        log.info(f"  [{ticker}] ApeWisdom: {mentions} bahsetme (24s önce: {mentions_24h}) → {final_score}")
                        
                        return {"score": final_score, "posts": mentions}
                        
        # Eğer 2 sayfa boyunca (en popüler 200 hisse arasında) yoksa nötr dön.
        log.info(f"  [{ticker}] ApeWisdom ilk 200'de yok, nötr (50.0) dönülüyor.")
        return {"score": 50.0, "posts": 0}

    except Exception as e:
        log.error(f"  [{ticker}] ApeWisdom çekme hatası: {e}")
        return {"score": 50.0, "posts": 0}


# ════════════════════════════════════════════════════════════
#  FİNNHUB SENTİMENT
# ════════════════════════════════════════════════════════════
def get_finnhub_sentiment(ticker: str) -> dict:
    """
    Birincil: Finnhub hazır sentiment endpoint (bullishPercent)
    Fallback:  Son 7 günün haberleri VADER ile analiz edilir
    """
    if not FINNHUB_API_KEY or FINNHUB_API_KEY == "buraya_yaz":
        log.warning(f"  [{ticker}] Finnhub API key girilmemiş, atlanıyor")
        return {"score": 50.0, "articles": 0}

    try:
        # Önce hazır sentiment endpoint'ini dene
        r = requests.get(
            "https://finnhub.io/api/v1/news-sentiment",
            params={"symbol": ticker, "token": FINNHUB_API_KEY},
            timeout=10,
        )

        if r.status_code == 200:
            data = r.json()
            bull_pct = data.get("sentiment", {}).get("bullishPercent", None)
            articles = data.get("buzz", {}).get("articlesInLastWeek", 0)

            if bull_pct is not None:
                score = round(bull_pct * 100, 1)
                log.info(f"  [{ticker}] Finnhub (hazır): {articles} haber → {score}")
                return {"score": score, "articles": articles}

        # Fallback: Haber başlıklarını VADER ile analiz et
        today    = date.today().strftime("%Y-%m-%d")
        week_ago = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")

        r2 = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": ticker, "from": week_ago,
                    "to": today, "token": FINNHUB_API_KEY},
            timeout=10,
        )

        if r2.status_code == 200:
            news = r2.json()
            if not news:
                return {"score": 50.0, "articles": 0}

            vader_scores = [vader_score(n.get("headline", ""))
                           for n in news[:30] if n.get("headline")]

            if vader_scores:
                avg = sum(vader_scores) / len(vader_scores)
                score = compound_to_100(avg)
                log.info(f"  [{ticker}] Finnhub (VADER fallback): {len(vader_scores)} haber → {score}")
                return {"score": score, "articles": len(vader_scores)}

        log.warning(f"  [{ticker}] Finnhub: {r.status_code} hatası")
        return {"score": 50.0, "articles": 0}

    except Exception as e:
        log.error(f"  [{ticker}] Finnhub hatası: {e}")
        return {"score": 50.0, "articles": 0}


# ════════════════════════════════════════════════════════════
#  BİRLEŞİK KARAR MOTORU
# ════════════════════════════════════════════════════════════
def calculate_final_score(tech: dict, reddit: dict, finnhub: dict) -> dict:
    """
    Teknik %60 + Sentiment %40
    MTF çelişki durumunda güven aralığı da hesaplanır.
    """
    sentiment_score = (
        reddit["score"]  * W_REDDIT  +
        finnhub["score"] * W_FINNHUB
    )
    final = round(
        tech["score"]    * W_TECHNICAL +
        sentiment_score  * W_SENTIMENT, 1
    )

    # Güven seviyesi: teknik ve sentiment aynı yönde mi?
    tech_bullish      = tech["score"]  >= 60
    sentiment_bullish = sentiment_score >= 60
    tech_bearish      = tech["score"]  <= 40
    sentiment_bearish = sentiment_score <= 40

    if (tech_bullish and sentiment_bullish) or (tech_bearish and sentiment_bearish):
        confidence = "YÜKSEK"      # İki katman hemfikir
    elif tech["above_ema200"] is None:
        confidence = "DÜŞÜK"       # Veri hatası
    else:
        confidence = "ORTA"        # Kısmi uyum

    return {
        "final":      final,
        "technical":  round(tech["score"], 1),
        "sentiment":  round(sentiment_score, 1),
        "confidence": confidence,
    }


def get_signal(scores: dict) -> str:
    score = scores["final"]
    conf  = scores["confidence"]

    if score >= 80 and conf == "YÜKSEK":  return "💪 GÜÇLÜ AL"
    if score >= BUY_THRESHOLD:            return "✅ AL"
    if score <= 20 and conf == "YÜKSEK":  return "🔴 GÜÇLÜ SAT"
    if score <= SELL_THRESHOLD:           return "🔴 SAT"
    return "⏸ BEKLE"


# ════════════════════════════════════════════════════════════
#  TELEGRAM
# ════════════════════════════════════════════════════════════
def send_telegram(text: str):
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "buraya_yaz":
        log.info(f"\n{'─'*55}\n{text}\n{'─'*55}")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if r.status_code == 200:
            log.info(f"  ✅ Telegram gönderildi")
        else:
            log.warning(f"  ⚠️  Telegram hatası: {r.status_code} {r.text}")
    except Exception as e:
        log.error(f"  Telegram exception: {e}")


def format_message(ticker: str, scores: dict, signal: str,
                   tech: dict, reddit: dict, finnhub: dict) -> str:
    now   = datetime.now().strftime("%d.%m.%Y %H:%M")
    icon  = "🟢" if "AL" in signal else ("🔴" if "SAT" in signal else "🟡")

    # Neden sinyal üretildi — insan anlaşılır açıklama
    if tech["above_ema200"]:
        trend_reason = f"✓ Fiyat EMA200 üzerinde (${tech['price']} > ${tech['ema200_d']})"
    else:
        trend_reason = f"✗ Fiyat EMA200 altında (${tech['price']} < ${tech['ema200_d']})"

    rsi_reason = (
        f"↗ RSI dipten döndü ({tech['rsi_val']})"
        if tech["hourly_rsi_score"] >= 75 else
        f"⚠️ RSI aşırı alım ({tech['rsi_val']})"
        if tech["hourly_rsi_score"] <= 20 else
        f"RSI nötr ({tech['rsi_val']})"
    )

    macd_reason = (
        "✓ Saatlik MACD pozitif momentum"
        if tech["hourly_macd_score"] >= 75 else
        "✗ Saatlik MACD negatif momentum"
        if tech["hourly_macd_score"] <= 25 else
        "MACD geçiş bölgesinde"
    )

    conf_icon = "🔒" if scores["confidence"] == "YÜKSEK" else \
                "🔓" if scores["confidence"] == "ORTA"   else "⚠️"

    return (
        f"🤖 <b>KANKA SİNYAL BOTU v3.0</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Hisse:</b> ${ticker}   ⏰ {now}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{icon} <b>SİNYAL: {signal}</b>\n"
        f"📊 <b>Bileşik Skor: {scores['final']}/100</b>\n"
        f"{conf_icon} <b>Güven: {scores['confidence']}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>📈 Teknik Analiz — MTF ({scores['technical']}/100)</b>\n"
        f"  {trend_reason}\n"
        f"  {rsi_reason}\n"
        f"  {macd_reason}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>💬 VADER Sentiment ({scores['sentiment']:.0f}/100)</b>\n"
        f"  Reddit:   {reddit['score']}/100  ({reddit.get('posts', 0)} metin)\n"
        f"  Finnhub:  {finnhub['score']}/100 ({finnhub.get('articles', 0)} haber)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 <b>Fiyat:</b> ${tech['price']}\n"
        f"⚡ <i>TradingView'da grafiki doğrula → Midas'ta işlem yap!</i>"
    )


# ════════════════════════════════════════════════════════════
#  SONUÇLARI KAYDET  (GitHub Actions ephemerality için)
# ════════════════════════════════════════════════════════════
def save_results(results: list):
    """
    Sinyal geçmişini signals_history.json dosyasına kaydeder.
    GitHub Actions her koşumda bu dosyayı repo'ya push eder
    (workflow dosyasında ayarlı), böylece geçmiş kaybolmaz.
    """
    history_file = Path("signals_history.json")
    history = []

    if history_file.exists():
        try:
            history = json.loads(history_file.read_text(encoding="utf-8"))
        except Exception:
            history = []

    for r in results:
        history.append({
            "timestamp": datetime.now().isoformat(),
            "ticker":    r["ticker"],
            "score":     r["scores"]["final"],
            "signal":    r["signal"],
            "technical": r["scores"]["technical"],
            "sentiment": r["scores"]["sentiment"],
            "confidence":r["scores"]["confidence"],
        })

    # Son 500 kaydı tut
    history = history[-500:]
    history_file.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    log.info(f"  💾 {len(history)} kayıt signals_history.json'a yazıldı")


# ════════════════════════════════════════════════════════════
#  ANA TARAMA
# ════════════════════════════════════════════════════════════
def scan():
    log.info(f"\n{'='*60}")
    log.info(f"  🔍 TARAMA: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    log.info(f"  Hisseler: {', '.join(TICKERS)}")
    log.info(f"{'='*60}")

    results = []

    for ticker in TICKERS:
        log.info(f"\n  ── {ticker} ──────────────────────────────────")

        tech    = get_technical_score(ticker)
        time.sleep(2)                           # yfinance rate limit koruması

        reddit  = get_reddit_sentiment(ticker)
        finnhub = get_finnhub_sentiment(ticker)
        time.sleep(1)                           # Finnhub rate limit koruması

        scores  = calculate_final_score(tech, reddit, finnhub)
        signal  = get_signal(scores)

        results.append({
            "ticker":  ticker,
            "scores":  scores,
            "signal":  signal,
            "tech":    tech,
            "reddit":  reddit,
            "finnhub": finnhub,
        })

        log.info(
            f"  → {ticker}: Teknik={scores['technical']} "
            f"Sentiment={scores['sentiment']:.0f} "
            f"TOPLAM={scores['final']} | {signal} | Güven:{scores['confidence']}"
        )

        # AL veya SAT → Telegram bildirimi
        if "AL" in signal or "SAT" in signal:
            msg = format_message(ticker, scores, signal, tech, reddit, finnhub)
            send_telegram(msg)

        time.sleep(1.5)

    # Özet tablo
    log.info(f"\n  {'─'*55}")
    log.info(f"  {'HİSSE':<8} {'SKOR':>6}  {'SİNYAL':<20}  GÜVEN")
    log.info(f"  {'─'*55}")
    for r in results:
        bar = "█" * int(r["scores"]["final"] / 10) + \
              "░" * (10 - int(r["scores"]["final"] / 10))
        log.info(
            f"  {r['ticker']:<8} [{bar}] {r['scores']['final']:>5}/100  "
            f"{r['signal']:<20}  {r['scores']['confidence']}"
        )
    log.info(f"  {'─'*55}")

    save_results(results)
    return results


# ════════════════════════════════════════════════════════════
#  GİRİŞ NOKTASI
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log.info("""
╔══════════════════════════════════════════════════════════════╗
║         KANKA SİNYAL BOTU v3.0 BAŞLADI                       ║
║  MTF Teknik (1H+1D) + VADER Sentiment → Telegram             ║
╚══════════════════════════════════════════════════════════════╝
    """)
    log.info(f"Hisseler  : {', '.join(TICKERS)}")
    log.info(f"AL eşiği  : >{BUY_THRESHOLD}  |  SAT eşiği: <{SELL_THRESHOLD}")
    log.info(f"Ağırlıklar: Teknik %{int(W_TECHNICAL*100)} / Sentiment %{int(W_SENTIMENT*100)}")
    log.info(f"Sentiment : Reddit %{int(W_REDDIT*100)} / Finnhub %{int(W_FINNHUB*100)}")
    log.info(f"VADER NLP : Aktif (keyword matching devre dışı)")

    # GitHub Actions'ta tek seferlik çalışır, loop sadece lokal için
    import sys
    if "--once" in sys.argv or os.getenv("GITHUB_ACTIONS"):
        scan()
    else:
        interval = int(os.getenv("SCAN_INTERVAL_MINUTES", 60))
        log.info(f"Tarama    : Her {interval} dakikada bir (Ctrl+C ile durdur)\n")
        while True:
            try:
                scan()
                log.info(f"\n  ⏳ Sonraki tarama {interval} dakika sonra...")
                time.sleep(interval * 60)
            except KeyboardInterrupt:
                log.info("\n  Bot durduruldu. 👋")
                break
            except Exception as e:
                log.error(f"\n  ❌ Beklenmedik hata: {e}")
                log.info("  5 dakika sonra tekrar deneniyor...")
                time.sleep(300)
