# 🚀 GitHub Actions Kurulum Rehberi — Tamamen Ücretsiz

## Neden GitHub Actions?

| | Railway | GitHub Actions |
|---|---|---|
| Ücretsiz süre | $5 kredi (biter) | **2.000 dk/ay (bitmez)** |
| Maliyet | Sürekli çalışır → tükenebilir | Sadece çalıştığında kullanır |
| Bu bot için | ~50dk/ay → ✅ ÜCRETSİZ |
| Veri kalıcılığı | ✅ | signals_history.json ile ✅ |

---

## Adım 1 — GitHub Hesabı ve Repo

1. **github.com** → Sign up (ücretsiz)
2. "New repository" → isim: `kanka-sinyal-botu` → Public → Create
3. Tüm bu dosyaları yükle:
   ```
   sinyal_botu.py
   requirements.txt
   config.env          ← Sadece template olarak (key'leri BOŞ bırak!)
   .github/
     workflows/
       bot.yml
   ```

⚠️ API key'lerini config.env'ye yazma! Bir sonraki adımda GitHub Secrets'a gireceksin.

---

## Adım 2 — GitHub Secrets (API Key'lerin Güvenli Saklama Yeri)

Repo sayfası → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Her birini ayrı ayrı ekle:

| Secret Adı | Değer |
|---|---|
| `REDDIT_CLIENT_ID` | reddit.com/prefs/apps'tan al |
| `REDDIT_CLIENT_SECRET` | reddit.com/prefs/apps'tan al |
| `REDDIT_USERNAME` | Reddit kullanıcı adın |
| `REDDIT_PASSWORD` | Reddit şifren |
| `FINNHUB_API_KEY` | finnhub.io dashboard'dan al |
| `TELEGRAM_BOT_TOKEN` | @BotFather'dan al |
| `TELEGRAM_CHAT_ID` | getUpdates ile al |
| `TICKERS` | `AAPL,TSLA,NVDA,SPY,QQQ` |
| `BUY_THRESHOLD` | `65` |
| `SELL_THRESHOLD` | `35` |

---

## Adım 3 — İlk Manuel Test

Repo → **Actions** sekmesi → **🤖 Kanka Sinyal Botu** → **Run workflow** → **Run workflow**

Logları izlemek için: Actions → En son workflow run → `run-bot` job'una tıkla

Şunu görmelisin:
```
╔══════════════════════════════════╗
║  KANKA SİNYAL BOTU v3.0 BAŞLADI ║
╚══════════════════════════════════╝
[10:00:01] AAPL taranıyor...
[10:00:03] Teknik: Günlük=100 H-RSI=72 H-MACD=75 → 82.1
[10:00:04] Reddit: 38 metin → compound=0.312 → 65.6
[10:00:05] Finnhub: 14 haber → 71.0
[10:00:05] → AAPL: Teknik=82.1 Sentiment=68.0 TOPLAM=74.9 | ✅ AL | Güven:YÜKSEK
```

---

## Adım 4 — Otomatik Çalışma Takvimi

`bot.yml` dosyasındaki cron ayarı:
```yaml
- cron: "0 1-19 * * 1-5"
```

Bu ayar:
- **Pazartesi – Cuma** (borsa günleri)
- **04:00 – 22:00 Türkiye saati** (UTC 01:00 – 19:00)
- **Her saat başı** çalışır

Aylık kullanım: 19 saat × 21 iş günü = **~400 dakika** (limit: 2.000 dk)

---

## Sinyal Geçmişi

Her koşumun ardından `signals_history.json` dosyası repoya otomatik push edilir.

Dosyada şunu görürsün:
```json
[
  {
    "timestamp": "2026-03-17T10:00:05",
    "ticker": "NVDA",
    "score": 81.3,
    "signal": "💪 GÜÇLÜ AL",
    "technical": 84.2,
    "sentiment": 76.5,
    "confidence": "YÜKSEK"
  }
]
```

---

## Telegram Mesaj Örneği

```
🤖 KANKA SİNYAL BOTU v3.0
━━━━━━━━━━━━━━━━━━━━━━━
📌 Hisse: $NVDA   ⏰ 17.03.2026 13:00
━━━━━━━━━━━━━━━━━━━━━━━
🟢 SİNYAL: 💪 GÜÇLÜ AL
📊 Bileşik Skor: 81/100
🔒 Güven: YÜKSEK
━━━━━━━━━━━━━━━━━━━━━━━
📈 Teknik Analiz — MTF (84/100)
  ✓ Fiyat EMA200 üzerinde ($924 > $841)
  ↗ RSI dipten döndü (38.2)
  ✓ Saatlik MACD pozitif momentum
━━━━━━━━━━━━━━━━━━━━━━━
💬 VADER Sentiment (76/100)
  Reddit:   78/100  (43 metin)
  Finnhub:  74/100  (12 haber)
━━━━━━━━━━━━━━━━━━━━━━━
💰 Fiyat: $924.50
⚡ TradingView'da grafiki doğrula → Midas'ta işlem yap!
```

---

## TradingView Çift Doğrulama

Bot sinyal verdiğinde TradingView'da Pine Script'e bak:

✅ Bot AL + Pine Script AL → Güvenli, işleme gir
⚠️ Bot AL + Pine Script BEKLE → Dikkatli ol, küçük pozisyon
❌ Bot AL + Pine Script SAT → BEKLE, çelişki var

---

## Sorun Giderme

**Actions çalışmıyor** → Repo → Settings → Actions → "Allow all actions" seçili mi?

**Secret bulunamıyor** → Secret adları büyük/küçük harf duyarlı, tam aynı yaz

**Reddit 401** → App type "script" seçilmiş olmalı

**yfinance 429** → v1.2.0 kullandığından emin ol (requirements.txt'te var)

**Telegram gelmiyor** → Bota en az 1 mesaj atmış olmalısın
