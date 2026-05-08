from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import joblib
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="KasuwaConnect AI Service",
    description="Credit scoring and job matching for informal traders",
    version="1.0.0",
)

# ── Load model assets on startup ──────────────────────────────────────────────

print("Loading credit model...")
model           = joblib.load("credit_model.joblib")
encoder_category = joblib.load("encoder_category.joblib")
encoder_state    = joblib.load("encoder_state.joblib")

with open("model_meta.json") as f:
    model_meta = json.load(f)

print("Model loaded successfully.")

# ── Schemas ───────────────────────────────────────────────────────────────────

class CreditScoreRequest(BaseModel):
    trader_id:              str
    avg_daily_transactions: float = Field(default=0, ge=0)
    avg_transaction_amount: float = Field(default=0, ge=0)
    trade_days_per_week:    float = Field(default=0, ge=0, le=7)
    supplier_diversity:     int   = Field(default=0, ge=0)
    payment_regularity:     float = Field(default=0, ge=0, le=1)
    dispute_rate:           float = Field(default=0, ge=0, le=1)
    total_transactions:     int   = Field(default=0, ge=0)
    avg_weekly_volume:      float = Field(default=0, ge=0)
    volume_growth_rate:     float = Field(default=0)
    months_active:          int   = Field(default=0, ge=0)
    category:               str   = Field(default="other")
    state:                  str   = Field(default="Lagos")

class CreditScoreResponse(BaseModel):
    trader_id:    str
    credit_score: int
    credit_tier:  str
    confidence:   str
    breakdown:    dict
    explanation:  str

# ── Helper ────────────────────────────────────────────────────────────────────

def get_credit_tier(score: int) -> str:
    if score >= 650: return "high"
    if score >= 400: return "medium"
    if score >= 150: return "low"
    return "unscored"

def get_confidence(months_active: int, total_transactions: int) -> str:
    if months_active >= 6 and total_transactions >= 50:
        return "high"
    if months_active >= 2 and total_transactions >= 10:
        return "medium"
    return "low"

def encode_category(value: str) -> int:
    try:
        return int(encoder_category.transform([value])[0])
    except:
        return int(encoder_category.transform(["other"])[0])

def encode_state(value: str) -> int:
    try:
        return int(encoder_state.transform([value])[0])
    except:
        return int(encoder_state.transform(["Lagos"])[0])

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "service":   "KasuwaConnect AI",
        "model_r2":  model_meta["r2"],
        "model_mae": model_meta["mae"],
    }

@app.post("/score", response_model=CreditScoreResponse)
def score_trader(req: CreditScoreRequest):
    try:
        features = np.array([[
            req.avg_daily_transactions,
            req.avg_transaction_amount,
            req.trade_days_per_week,
            req.supplier_diversity,
            req.payment_regularity,
            req.dispute_rate,
            req.total_transactions,
            req.avg_weekly_volume,
            req.volume_growth_rate,
            req.months_active,
            encode_category(req.category),
            encode_state(req.state),
        ]])

        raw_score    = model.predict(features)[0]
        credit_score = int(np.clip(raw_score, 0, 850))
        credit_tier  = get_credit_tier(credit_score)
        confidence   = get_confidence(req.months_active, req.total_transactions)

        # Build human-readable explanation
        strong_signals = []
        weak_signals   = []

        if req.payment_regularity >= 0.7:
            strong_signals.append("consistent payment patterns")
        else:
            weak_signals.append("irregular payment activity")

        if req.trade_days_per_week >= 5:
            strong_signals.append("trades frequently")
        elif req.trade_days_per_week < 3:
            weak_signals.append("infrequent trading days")

        if req.avg_weekly_volume >= 20000:
            strong_signals.append("strong weekly revenue")
        elif req.avg_weekly_volume < 3000:
            weak_signals.append("low weekly volume")

        if req.supplier_diversity >= 5:
            strong_signals.append("diverse trade network")

        if req.dispute_rate > 0.10:
            weak_signals.append("elevated dispute rate")

        parts = []
        if strong_signals:
            parts.append("Positive signals: " + ", ".join(strong_signals))
        if weak_signals:
            parts.append("Areas for improvement: " + ", ".join(weak_signals))
        if not strong_signals and not weak_signals:
            parts.append("Insufficient transaction history to evaluate")

        explanation = ". ".join(parts) + "."

        return CreditScoreResponse(
            trader_id    = req.trader_id,
            credit_score = credit_score,
            credit_tier  = credit_tier,
            confidence   = confidence,
            breakdown    = {
                "payment_regularity":     req.payment_regularity,
                "trade_days_per_week":    req.trade_days_per_week,
                "avg_weekly_volume":      req.avg_weekly_volume,
                "total_transactions":     req.total_transactions,
                "supplier_diversity":     req.supplier_diversity,
                "dispute_rate":           req.dispute_rate,
            },
            explanation = explanation,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    return model_meta