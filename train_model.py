import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import joblib
import json
import os

DATA_FILE  = "training_data.csv"
MODEL_FILE = "credit_model.joblib"
META_FILE  = "model_meta.json"

print("Loading training data...")
df = pd.read_csv(DATA_FILE)

# ── Feature Engineering ───────────────────────────────────────────────────────

# Encode categorical features
le_category = LabelEncoder()
le_state     = LabelEncoder()

df["category_encoded"] = le_category.fit_transform(df["category"])
df["state_encoded"]    = le_state.fit_transform(df["state"])

# Save encoders for inference
joblib.dump(le_category, "encoder_category.joblib")
joblib.dump(le_state,    "encoder_state.joblib")

# Feature set — these are the exact signals from Squad transaction data
FEATURES = [
    "avg_daily_transactions",   # how often they transact
    "avg_transaction_amount",   # average payment size
    "trade_days_per_week",      # consistency of trading activity
    "supplier_diversity",       # breadth of their trade network
    "payment_regularity",       # how consistent their payment patterns are
    "dispute_rate",             # how often payments are disputed
    "total_transactions",       # total volume of transactions
    "avg_weekly_volume",        # average weekly revenue
    "volume_growth_rate",       # whether their business is growing
    "months_active",            # how long they have been on the platform
    "category_encoded",         # trade category
    "state_encoded",            # location
]

TARGET = "credit_score"

X = df[FEATURES]
y = df[TARGET]

# ── Train / Test Split ────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")

# ── Train LightGBM ────────────────────────────────────────────────────────────

print("\nTraining LightGBM credit model...")

model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
)

# ── Evaluate ──────────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, 850)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"  MAE : {mae:.2f} points  (average error on credit score)")
print(f"  R²  : {r2:.4f}          (1.0 = perfect, >0.85 = strong)")

# ── Feature Importance ────────────────────────────────────────────────────────

importance = dict(zip(FEATURES, [int(x) for x in model.feature_importances_]))
importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

print(f"\nFeature Importance (top signals):")
for feat, imp in importance_sorted.items():
    bar = "█" * int(imp / max(importance_sorted.values()) * 30)
    print(f"  {feat:<30} {bar} {imp:.0f}")

# ── Save Model + Metadata ─────────────────────────────────────────────────────

joblib.dump(model, MODEL_FILE)

meta = {
    "features":         FEATURES,
    "target":           TARGET,
    "mae":              round(mae, 2),
    "r2":               round(r2, 4),
    "n_training":       len(X_train),
    "n_test":           len(X_test),
    "feature_importance": importance_sorted,
    "score_range":      {"min": 0, "max": 850},
    "tiers": {
        "high":     "650–850",
        "medium":   "400–649",
        "low":      "150–399",
        "unscored": "0–149",
    },
    "trained_at": pd.Timestamp.now().isoformat(),
}

with open(META_FILE, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nModel saved to {MODEL_FILE}")
print(f"Metadata saved to {META_FILE}")
print("\nTraining complete.")