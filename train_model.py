"""
Kanka Sinyal Botu v5.1 — Model Eğitimi (Walk-Forward)
=======================================================
kanka_training_data.csv üzerinde TimeSeriesSplit ile 5-fold
walk-forward validation yaparak LightGBM eğitir.

Çıktı: kanka_model.joblib  (en yüksek macro F1'li fold'un modeli)

Çalıştırma:
  python train_model.py
"""

import logging
import sys

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DATA_FILE  = "kanka_training_data.csv"
MODEL_FILE = "kanka_model.joblib"

TARGET_COL = "target"
DROP_COLS  = ["date", "ticker", "target"]

# ─── Veri Yükleme ─────────────────────────────────────────────────────────────
log.info(f"Veri yükleniyor: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)
df.dropna(inplace=True)
log.info(f"Toplam satır: {len(df):,}")
log.info(f"Target dağılımı:\n{df[TARGET_COL].value_counts().to_string()}")

# ─── Özellik ve Hedef ─────────────────────────────────────────────────────────
feature_cols = [c for c in df.columns if c not in DROP_COLS]
X = df[feature_cols].values
y = df[TARGET_COL].values
log.info(f"Özellik sayısı: {len(feature_cols)} → {feature_cols}")

# ─── Walk-Forward Validation (5 Fold) ────────────────────────────────────────
tscv       = TimeSeriesSplit(n_splits=5)
scores     = []
best_model = None
best_f1    = -np.inf

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = LGBMClassifier(
        n_estimators      = 400,
        learning_rate     = 0.05,
        num_leaves        = 15,
        min_child_samples = 20,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        class_weight      = "balanced",
        random_state      = 42,
        verbose           = -1,
        n_jobs            = -1,
    )

    model.fit(
        X_train, y_train,
        eval_set  = [(X_val, y_val)],
        callbacks = [
            __import__("lightgbm").early_stopping(30, verbose=False),
            __import__("lightgbm").log_evaluation(0),
        ],
    )

    report   = classification_report(
        y_val, model.predict(X_val),
        output_dict=True, zero_division=0,
    )
    macro_f1 = report["macro avg"]["f1-score"]
    scores.append(macro_f1)
    log.info(f"Fold {fold + 1}/5 — Macro F1: {macro_f1:.3f}")

    if macro_f1 > best_f1:
        best_f1    = macro_f1
        best_model = model

log.info(f"Walk-Forward Ort F1: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
log.info(f"En iyi fold F1: {best_f1:.3f}")

# ─── En önemli 5 özellik ──────────────────────────────────────────────────────
importances = best_model.feature_importances_
top5_idx    = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)[:5]
log.info("En önemli 5 özellik:")
for rank, idx in enumerate(top5_idx, 1):
    log.info(f"  {rank}. {feature_cols[idx]:<25} importance={importances[idx]:,.0f}")

# ─── Kaydet ───────────────────────────────────────────────────────────────────
joblib.dump(best_model, MODEL_FILE)
log.info(f"Model kaydedildi: {MODEL_FILE}")

# ─── Drift tespiti için referans istatistikleri ───────────────────────────────
ref_stats = {}
for feat in feature_cols:
    col_data = df[feat].dropna()
    ref_stats[feat] = {
        "mean": float(col_data.mean()),
        "std":  float(col_data.std()),
        "p5":   float(col_data.quantile(0.05)),
        "p95":  float(col_data.quantile(0.95)),
    }
stats_payload = {"feature_cols": feature_cols, "stats": ref_stats}
joblib.dump(stats_payload, "kanka_model_stats.joblib")
log.info("Referans istatistikleri kaydedildi: kanka_model_stats.joblib")
