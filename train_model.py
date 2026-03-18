"""
Kanka Sinyal Botu v5.1 — Model Eğitimi
========================================
kanka_training_data.csv üzerinde LightGBM sınıflandırıcısı eğitir.

Çıktı: kanka_model.joblib
  - LGBMClassifier: OHLCV + teknik indikatörler + makro → 5 gün içinde %2+ yükseliş tahmini

Çalıştırma:
  python train_model.py
"""

import logging
import sys

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DATA_FILE  = "kanka_training_data.csv"
MODEL_FILE = "kanka_model.joblib"

# ─── Veri Yükleme ─────────────────────────────────────────────────────────────
log.info(f"Veri yükleniyor: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)
log.info(f"Toplam satır: {len(df):,} | Kolonlar: {list(df.columns)}")

# ─── Özellik ve Hedef ─────────────────────────────────────────────────────────
X = df.drop(columns=["date", "ticker", "target"], errors="ignore")
y = df["target"]

log.info(f"Özellik sayısı: {X.shape[1]} | Hedef dağılımı:\n{y.value_counts().to_string()}")

# ─── Eğitim / Test Bölmesi ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=False  # zaman serisi — karıştırma
)
log.info(f"Eğitim: {len(X_train):,} satır | Test: {len(X_test):,} satır")

# ─── Model Eğitimi ────────────────────────────────────────────────────────────
log.info("LGBMClassifier eğitiliyor...")
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
model.fit(X_train, y_train)

# ─── Başarı Analizi ───────────────────────────────────────────────────────────
accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy (Test): {accuracy:.4f}  ({accuracy*100:.2f}%)\n")

feature_names = X.columns.tolist()
importances   = model.feature_importances_
top5_idx      = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)[:5]

log.info("En önemli 5 özellik:")
for rank, idx in enumerate(top5_idx, 1):
    log.info(f"  {rank}. {feature_names[idx]:<25} importance={importances[idx]:,.0f}")

# ─── Kaydet ───────────────────────────────────────────────────────────────────
joblib.dump(model, MODEL_FILE)
log.info(f"Model kaydedildi: {MODEL_FILE}")
