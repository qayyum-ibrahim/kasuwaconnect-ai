import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

NUM_TRADERS = 500
OUTPUT_FILE = "training_data.csv"

TRADE_CATEGORIES = ["food", "clothing", "electronics", "artisan", "transport", "agriculture"]
STATES           = ["Lagos", "Anambra", "Kano", "Rivers", "Oyo", "Kaduna", "Enugu"]

def generate_trader_profile(trader_id):
    """Generate a realistic informal trader profile with transaction history."""

    category    = random.choice(TRADE_CATEGORIES)
    state       = random.choice(STATES)
    months_active = random.randint(1, 24)

    # --- Behaviour archetypes ---
    # High performers: consistent, high volume, diverse suppliers
    # Medium performers: moderate consistency, average volume
    # Low performers: irregular, low volume, few transactions
    # Ghost accounts: registered but barely active

    archetype = random.choices(
        ["high", "medium", "low", "ghost"],
        weights=[0.20, 0.35, 0.30, 0.15]
    )[0]

    if archetype == "high":
        avg_daily_transactions = round(random.uniform(4, 10), 2)
        avg_transaction_amount = round(random.uniform(5000, 50000), 2)
        trade_days_per_week    = round(random.uniform(5, 7), 2)
        supplier_diversity     = random.randint(5, 15)
        payment_regularity     = round(random.uniform(0.75, 1.0), 3)
        dispute_rate           = round(random.uniform(0.0, 0.05), 3)

    elif archetype == "medium":
        avg_daily_transactions = round(random.uniform(2, 5), 2)
        avg_transaction_amount = round(random.uniform(2000, 15000), 2)
        trade_days_per_week    = round(random.uniform(3, 6), 2)
        supplier_diversity     = random.randint(2, 7)
        payment_regularity     = round(random.uniform(0.5, 0.80), 3)
        dispute_rate           = round(random.uniform(0.02, 0.10), 3)

    elif archetype == "low":
        avg_daily_transactions = round(random.uniform(0.5, 2), 2)
        avg_transaction_amount = round(random.uniform(500, 5000), 2)
        trade_days_per_week    = round(random.uniform(1, 4), 2)
        supplier_diversity     = random.randint(1, 3)
        payment_regularity     = round(random.uniform(0.25, 0.55), 3)
        dispute_rate           = round(random.uniform(0.05, 0.20), 3)

    else:  # ghost
        avg_daily_transactions = round(random.uniform(0, 0.5), 2)
        avg_transaction_amount = round(random.uniform(100, 1000), 2)
        trade_days_per_week    = round(random.uniform(0, 1), 2)
        supplier_diversity     = random.randint(0, 2)
        payment_regularity     = round(random.uniform(0.0, 0.25), 3)
        dispute_rate           = round(random.uniform(0.10, 0.40), 3)

    # Derived features
    total_transactions = int(avg_daily_transactions * trade_days_per_week * 4.3 * months_active)
    total_volume_ngn   = round(total_transactions * avg_transaction_amount, 2)
    avg_weekly_volume  = round(total_volume_ngn / max(months_active * 4.3, 1), 2)

    # Volume growth — high performers trend upward
    volume_growth_rate = {
        "high":   round(random.uniform(0.05, 0.25), 3),
        "medium": round(random.uniform(-0.05, 0.10), 3),
        "low":    round(random.uniform(-0.15, 0.05), 3),
        "ghost":  round(random.uniform(-0.30, 0.0), 3),
    }[archetype]

    # --- Credit score calculation ---
    # Weighted scoring formula — mirrors what a real credit model learns
    score = 0
    score += min(payment_regularity * 300, 300)       # max 300 pts — most important signal
    score += min((total_transactions / 200) * 200, 200) # max 200 pts
    score += min((avg_weekly_volume / 50000) * 150, 150) # max 150 pts
    score += min((supplier_diversity / 10) * 100, 100)  # max 100 pts
    score += min(trade_days_per_week / 7 * 100, 100)    # max 100 pts
    score -= dispute_rate * 200                          # penalty

    # Add realistic noise
    score += random.gauss(0, 15)
    credit_score = int(np.clip(score, 0, 850))

    # Credit tier
    if credit_score >= 650:
        credit_tier = "high"
    elif credit_score >= 400:
        credit_tier = "medium"
    elif credit_score >= 150:
        credit_tier = "low"
    else:
        credit_tier = "unscored"

    return {
        "trader_id":              f"synthetic_{trader_id:04d}",
        "category":               category,
        "state":                  state,
        "months_active":          months_active,
        "archetype":              archetype,
        "avg_daily_transactions": avg_daily_transactions,
        "avg_transaction_amount": avg_transaction_amount,
        "trade_days_per_week":    trade_days_per_week,
        "supplier_diversity":     supplier_diversity,
        "payment_regularity":     payment_regularity,
        "dispute_rate":           dispute_rate,
        "total_transactions":     total_transactions,
        "total_volume_ngn":       total_volume_ngn,
        "avg_weekly_volume":      avg_weekly_volume,
        "volume_growth_rate":     volume_growth_rate,
        "credit_score":           credit_score,
        "credit_tier":            credit_tier,
    }

# Generate dataset
print("Generating synthetic trader dataset...")
traders = [generate_trader_profile(i) for i in range(NUM_TRADERS)]
df = pd.DataFrame(traders)

# Save to CSV
df.to_csv(OUTPUT_FILE, index=False)

# Summary
print(f"\nDataset saved to {OUTPUT_FILE}")
print(f"Total traders: {len(df)}")
print(f"\nCredit tier distribution:")
print(df["credit_tier"].value_counts())
print(f"\nArchetype distribution:")
print(df["archetype"].value_counts())
print(f"\nCredit score stats:")
print(df["credit_score"].describe())
print(f"\nSample row:")
print(df.iloc[0].to_dict())