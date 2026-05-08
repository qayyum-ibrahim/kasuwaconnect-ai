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

# ── Job Matching Schemas ──────────────────────────────────────────────────────

class JobSeekerProfile(BaseModel):
    seeker_id:            str
    skills:               list[str]       = Field(default=[])
    preferred_categories: list[str]       = Field(default=[])
    languages:            list[str]       = Field(default=["english"])
    experience_level:     str             = Field(default="none")
    state:                str             = Field(default="")
    market_location:      Optional[str]   = Field(default=None)

class JobListing(BaseModel):
    job_id:             str
    title:              str
    category:           str
    skills_required:    list[str]       = Field(default=[])
    languages_required: list[str]       = Field(default=["english"])
    experience_level:   str             = Field(default="none")
    pay_amount:         float           = Field(default=0)
    pay_frequency:      str             = Field(default="daily")
    state:              str             = Field(default="")
    market_location:    Optional[str]   = Field(default=None)
    trader_id:          Optional[str]   = Field(default=None)

class MatchRequest(BaseModel):
    seeker:       JobSeekerProfile
    jobs:         list[JobListing]
    top_n:        int = Field(default=5, ge=1, le=20)

class JobMatch(BaseModel):
    job_id:           str
    title:            str
    match_score:      float
    match_percentage: int
    matched_skills:   list[str]
    score_breakdown:  dict
    pay_amount:       float
    pay_frequency:    str
    state:            str
    market_location:  Optional[str]

class MatchResponse(BaseModel):
    seeker_id:      str
    total_jobs:     int
    matches_found:  int
    matches:        list[JobMatch]

# ── Matching Logic ────────────────────────────────────────────────────────────

EXPERIENCE_LEVELS = ["none", "beginner", "intermediate", "experienced"]

def score_match(seeker: JobSeekerProfile, job: JobListing) -> dict:
    """Score a single seeker-job pair. Returns breakdown and total."""

    # 1. Skills overlap (35%)
    seeker_skills  = set(s.lower().strip() for s in seeker.skills)
    job_skills     = set(s.lower().strip() for s in job.skills_required)
    matched_skills = list(seeker_skills & job_skills)

    if len(job_skills) == 0:
        skill_score = 1.0  # no specific skills required — open to all
    else:
        skill_score = len(matched_skills) / len(job_skills)

    # 2. Category preference (25%)
    category_score = 1.0 if job.category in seeker.preferred_categories else 0.3

    # 3. Location proximity (20%)
    if seeker.state.lower() == job.state.lower():
        if seeker.market_location and job.market_location:
            location_score = (
                1.0 if seeker.market_location.lower() == job.market_location.lower()
                else 0.7
            )
        else:
            location_score = 0.7
    else:
        location_score = 0.1  # different state — very low match

    # 4. Language match (15%)
    seeker_langs = set(l.lower() for l in seeker.languages)
    job_langs    = set(l.lower() for l in job.languages_required)
    lang_overlap = seeker_langs & job_langs

    if len(job_langs) == 0:
        language_score = 1.0
    else:
        language_score = len(lang_overlap) / len(job_langs)

    # 5. Experience fit (5%)
    seeker_exp_idx = EXPERIENCE_LEVELS.index(seeker.experience_level) \
        if seeker.experience_level in EXPERIENCE_LEVELS else 0
    job_exp_idx = EXPERIENCE_LEVELS.index(job.experience_level) \
        if job.experience_level in EXPERIENCE_LEVELS else 0

    exp_diff = seeker_exp_idx - job_exp_idx
    if exp_diff >= 0:
        experience_score = 1.0   # seeker meets or exceeds requirement
    elif exp_diff == -1:
        experience_score = 0.5   # one level below — possible stretch
    else:
        experience_score = 0.1   # too far below requirement

    # Weighted total
    total = (
        skill_score      * 0.35 +
        category_score   * 0.25 +
        location_score   * 0.20 +
        language_score   * 0.15 +
        experience_score * 0.05
    )

    return {
        "total":              round(total, 4),
        "matched_skills":     matched_skills,
        "breakdown": {
            "skills_score":     round(skill_score, 3),
            "category_score":   round(category_score, 3),
            "location_score":   round(location_score, 3),
            "language_score":   round(language_score, 3),
            "experience_score": round(experience_score, 3),
        }
    }

# ── Match Route ───────────────────────────────────────────────────────────────

@app.post("/match", response_model=MatchResponse)
def match_jobs(req: MatchRequest):
    try:
        if not req.jobs:
            return MatchResponse(
                seeker_id     = req.seeker.seeker_id,
                total_jobs    = 0,
                matches_found = 0,
                matches       = [],
            )

        scored = []
        for job in req.jobs:
            result = score_match(req.seeker, job)
            if result["total"] > 0.15:  # filter out near-zero matches
                scored.append({
                    "job_id":           job.job_id,
                    "title":            job.title,
                    "match_score":      result["total"],
                    "match_percentage": int(result["total"] * 100),
                    "matched_skills":   result["matched_skills"],
                    "score_breakdown":  result["breakdown"],
                    "pay_amount":       job.pay_amount,
                    "pay_frequency":    job.pay_frequency,
                    "state":            job.state,
                    "market_location":  job.market_location,
                })

        # Sort by match score descending
        scored.sort(key=lambda x: x["match_score"], reverse=True)
        top_matches = scored[:req.top_n]

        return MatchResponse(
            seeker_id     = req.seeker.seeker_id,
            total_jobs    = len(req.jobs),
            matches_found = len(top_matches),
            matches       = top_matches,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/model-info")
def model_info():
    return model_meta