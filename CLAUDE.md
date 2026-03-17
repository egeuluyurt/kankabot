# Kanka Sinyal Botu v4.0 — Kalıcı Bellek & Mimari Kurallar

## Proje Özeti
Alpaca Markets üzerinde çalışan, yfinance + pandas-ta teknik analiz,
ApeWisdom Reddit sentiment ve Finnhub haber analizi kullanan asenkron
bir algoritmik trading sinyal botu.

---

## Dosya Yapısı

```
/opt/kanka-bot/
├── bot.py            — Ana trading + analiz mantığı
├── config.env        — API anahtarları ve parametreler (chmod 600)
├── requirements.txt  — Python bağımlılıkları
├── bot.log           — Çalışma logları (otomatik oluşur)
└── venv/             — Python sanal ortamı
/etc/systemd/system/kanka-bot.service
setup.sh              — Tek seferlik VPS kurulum scripti
```

---

## KRİTİK MİMARİ KURALLAR

### 1. Alpaca Emir Yapısı
- **qty parametresi ASLA kullanılmaz.** Tüm emirler `notional=X` (dolar bazlı).
- Emir türü her zaman `OrderType.MARKET`.
- Stop-loss ve take-profit için **bracket order** kullanılır:
  ```python
  order_class=OrderClass.BRACKET
  stop_loss=StopLossRequest(stop_price=entry - 1.5*ATR)
  take_profit=TakeProfitRequest(limit_price=entry + 3.0*ATR)
  ```
- Manuel SL/TP döngüsü **yasaktır** — borsa sunucusu yönetir.

### 2. yfinance MultiIndex Düzeltmesi
yfinance.download() çoklu ticker için MultiIndex döndürür. Her ticker **ayrı ayrı** indirilir:
```python
df = yf.download(ticker, period=X, interval=Y, auto_adjust=True, progress=False)
if hasattr(df.columns, 'levels'):
    df.columns = df.columns.get_level_values(0)
```
Bu düzeltme olmadan pandas-ta **KeyError** ile çöker.

### 3. Reddit Sentiment — PRAW Yasak
2026 itibarıyla Reddit, veri merkezi IP'lerinden PRAW isteklerini **403** ile engeller.
Bunun yerine **ApeWisdom** kullanılır (API key gerektirmez):
```
GET https://apewisdom.io/api/v1.0/filter/all-stocks/page/1
```

### 4. Rate Limit Koruması
Tüm dış ağ istekleri **tenacity** ile sarmalanır:
- HTTP 429 → Exponential Backoff: 2^n saniye, max 60sn, max 5 deneme
- Alpaca limiti: 200 istek/dakika
- Finnhub limiti: 60 istek/dakika
- ApeWisdom limiti: 20 istek/dakika

### 5. Telegram — Asenkron + Senkron Fallback
- Komut dinleyici **ayrı thread** içinde asyncio event loop ile çalışır.
- Ana trading döngüsü **bloklanmaz**.
- `tg_send()` fonksiyonu `requests.post` ile senkron fallback sağlar.

### 6. Güvenlik
- config.env: **chmod 600** — sadece sahip okuyabilir.
- API key'ler hiçbir zaman koda gömülmez; `python-dotenv` ile okunur.
- Bot **kankabot** kullanıcısı altında çalışır, **root değil**.

---

## Teknik Analiz Parametreleri

### MTF — Günlük (1D) Filtre
| Koşul | Skor |
|-------|------|
| Fiyat > EMA50 > EMA200 | 100 (güçlü boğa) |
| Fiyat > EMA200 | 70 (yükseliş) |
| Fiyat < EMA50 < EMA200 | 0 (güçlü ayı) |
| Fiyat < EMA200 | 30 (düşüş) |
| Diğer | 50 (kararsız) |

### MTF — Saatlik (1H) Tetikleyici
RSI(14) ve MACD(12,26,9) skorları hesaplanır.

**MTF Çelişki Cezası:** Günlük ≠ Saatlik yön → skor × 0.85

### Ağırlıklar
```
tech_score      = daily*0.40 + rsi*0.30 + macd*0.30
sentiment_score = reddit*0.50 + finnhub*0.50
final_score     = tech*0.60 + sentiment*0.40
```

### ATR Bazlı Risk
```
Stop Loss   = giriş - 1.5 × ATR(14)
Take Profit = giriş + 3.0 × ATR(14)
Pozisyon    = portföy × POSITION_PCT/100  (notional)
```

---

## Sinyal Eşikleri

| Sinyal | Koşul |
|--------|-------|
| 💪 GÜÇLÜ AL | final >= 80 VE güven=YÜKSEK |
| ✅ AL | final >= 65 |
| 🔴 GÜÇLÜ SAT | final <= 20 VE güven=YÜKSEK |
| 🔴 SAT | final <= 35 |
| ⏸ BEKLE | Diğer |

**Güven:** Her iki skor (tech+sentiment) >= 60 VEYA <= 40 → YÜKSEK

---

## Risk Yönetimi Kontrol Sırası (should_buy)
1. BOT_PAUSED → False
2. Borsa saatleri dışı → False
3. Pozisyon sayısı >= MAX_POSITIONS → False
4. Ticker zaten pozisyonda → False
5. buying_power < position_size → False
6. ATR hesaplanamadı → False
7. Hepsi geçtiyse → True

---

## Ortam Değişkenleri (config.env)

| Değişken | Açıklama | Varsayılan |
|----------|----------|------------|
| ALPACA_API_KEY | Alpaca API anahtarı | — |
| ALPACA_SECRET_KEY | Alpaca gizli anahtarı | — |
| PAPER_TRADING | Paper/Live mod | true |
| FINNHUB_API_KEY | Finnhub API anahtarı | — |
| TELEGRAM_BOT_TOKEN | Telegram bot tokeni | — |
| TELEGRAM_CHAT_ID | Telegram sohbet ID | — |
| TICKERS | Taranacak hisseler | AAPL,TSLA,NVDA,SPY,QQQ |
| BUY_THRESHOLD | Alım eşiği | 65 |
| SELL_THRESHOLD | Satış eşiği | 35 |
| POSITION_PCT | Portföy başına pozisyon % | 5 |
| MAX_POSITIONS | Aynı anda max pozisyon | 3 |
| ATR_SL_MULT | ATR stop loss çarpanı | 1.5 |
| ATR_TP_MULT | ATR take profit çarpanı | 3.0 |
| SCAN_INTERVAL_MINUTES | Tarama aralığı (dk) | 60 |

---

## Telegram Komutları
- `/durum` — Portföy özeti + bot durumu
- `/portfoy` — Açık pozisyonlar + SL/TP
- `/durdur` — Taramayı duraklat
- `/baslat` — Taramayı devam ettir

---

## Bağımlılıklar
```
alpaca-py==0.38.0
yfinance==1.2.0
pandas==2.2.3
pandas-ta==0.3.14b
vaderSentiment==3.3.2
python-telegram-bot==21.11.1
tenacity==9.0.0
requests==2.32.3
python-dotenv==1.0.1
```
(praw kaldırıldı — ApeWisdom kullanılıyor)

---

## Çalıştırma
```bash
# Geliştirme
source venv/bin/activate
python bot.py

# Servis
sudo systemctl start kanka-bot
sudo systemctl status kanka-bot
sudo journalctl -u kanka-bot -f
```

---

*Son güncelleme: 2026-03-17 | Kanka Sinyal Botu v4.0*
