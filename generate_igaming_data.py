"""
DA-1: iGaming KPI Analytics Suite
Dataset Generator - Produces realistic iGaming player + transaction data
Output: 3 CSV files ready for EDA and PostgreSQL loading
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os

fake = Faker()
np.random.seed(42)
random.seed(42)

# ── CONFIG ──────────────────────────────────────────────────────────────────
N_PLAYERS       = 50_000
START_DATE      = datetime(2023, 1, 1)
END_DATE        = datetime(2024, 12, 31)
DATE_RANGE_DAYS = (END_DATE - START_DATE).days

OUTPUT_DIR = "/home/claude/igaming_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── SEGMENT DEFINITIONS ─────────────────────────────────────────────────────
# Each segment has different behavioural profiles
SEGMENTS = {
    "VIP":       {"weight": 0.05, "avg_deposit": 800,  "deposit_std": 400,  "sessions_per_month": 20, "avg_bet": 50,  "rtp": 0.94, "bonus_rate": 0.30, "churn_monthly_prob": 0.03},
    "High":      {"weight": 0.15, "avg_deposit": 300,  "deposit_std": 150,  "sessions_per_month": 12, "avg_bet": 20,  "rtp": 0.93, "bonus_rate": 0.40, "churn_monthly_prob": 0.06},
    "Mid":       {"weight": 0.35, "avg_deposit": 100,  "deposit_std": 60,   "sessions_per_month": 6,  "avg_bet": 8,   "rtp": 0.92, "bonus_rate": 0.50, "churn_monthly_prob": 0.10},
    "Low":       {"weight": 0.30, "avg_deposit": 40,   "deposit_std": 20,   "sessions_per_month": 3,  "avg_bet": 3,   "rtp": 0.91, "bonus_rate": 0.60, "churn_monthly_prob": 0.18},
    "Dormant":   {"weight": 0.15, "avg_deposit": 20,   "deposit_std": 10,   "sessions_per_month": 1,  "avg_bet": 2,   "rtp": 0.90, "bonus_rate": 0.70, "churn_monthly_prob": 0.40},
}

COUNTRIES = {
    "Germany":     0.18, "UK": 0.16, "Sweden": 0.10, "Finland": 0.08,
    "Canada":      0.08, "Norway": 0.07, "Austria": 0.06, "Malta": 0.05,
    "Netherlands": 0.05, "Australia": 0.07, "New Zealand": 0.04, "Ireland": 0.06,
}

GAME_TYPES = ["Slots", "Live Casino", "Table Games", "Sports Betting", "Poker"]
GAME_WEIGHTS = [0.45, 0.25, 0.12, 0.13, 0.05]

PAYMENT_METHODS = ["Credit Card", "Bank Transfer", "E-Wallet", "Crypto", "Prepaid Card"]
PAYMENT_WEIGHTS = [0.30, 0.25, 0.28, 0.10, 0.07]

CHANNELS = ["Organic", "Paid Search", "Affiliate", "Email", "Social", "Referral"]
CHANNEL_WEIGHTS = [0.20, 0.22, 0.30, 0.12, 0.10, 0.06]

PROMO_TYPES = ["Welcome Bonus", "Free Spins", "Reload Bonus", "Cashback", "VIP Reward", "None"]
PROMO_WEIGHTS = [0.15, 0.20, 0.20, 0.15, 0.05, 0.25]

# ── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

def weighted_choice(options, weights):
    return random.choices(options, weights=weights, k=1)[0]

# ── 1. PLAYERS TABLE ────────────────────────────────────────────────────────
print("Generating players table...")

segment_list  = list(SEGMENTS.keys())
segment_weights = [SEGMENTS[s]["weight"] for s in segment_list]
country_list  = list(COUNTRIES.keys())
country_weights = list(COUNTRIES.values())

players = []
for i in range(N_PLAYERS):
    player_id   = f"PLR{100000 + i}"
    segment     = weighted_choice(segment_list, segment_weights)
    country     = weighted_choice(country_list, country_weights)
    reg_date    = random_date(START_DATE, END_DATE - timedelta(days=30))
    channel     = weighted_choice(CHANNELS, CHANNEL_WEIGHTS)
    age         = int(np.clip(np.random.normal(34, 10), 18, 70))
    gender      = random.choices(["Male", "Female", "Other"], weights=[0.62, 0.35, 0.03])[0]
    preferred_game = weighted_choice(GAME_TYPES, GAME_WEIGHTS)
    payment_method = weighted_choice(PAYMENT_METHODS, PAYMENT_WEIGHTS)

    # Churn date logic: based on segment's monthly churn prob
    profile = SEGMENTS[segment]
    months_active = (END_DATE - reg_date).days // 30
    churned = False
    churn_date = None
    for m in range(months_active):
        if random.random() < profile["churn_monthly_prob"]:
            churned = True
            churn_date = reg_date + timedelta(days=30 * (m + 1) + random.randint(0, 29))
            break

    is_active = not churned
    last_login = (churn_date - timedelta(days=random.randint(1, 30))) if churned else \
                 (END_DATE - timedelta(days=random.randint(0, 30)))

    players.append({
        "player_id":        player_id,
        "segment":          segment,
        "country":          country,
        "registration_date": reg_date.date(),
        "acquisition_channel": channel,
        "age":              age,
        "gender":           gender,
        "preferred_game":   preferred_game,
        "payment_method":   payment_method,
        "is_active":        is_active,
        "churn_date":       churn_date.date() if churn_date else None,
        "last_login_date":  last_login.date(),
    })

df_players = pd.DataFrame(players)
print(f"  Players: {len(df_players):,} rows")

# ── 2. TRANSACTIONS TABLE ───────────────────────────────────────────────────
print("Generating transactions table...")

transactions = []
txn_id = 1

for _, player in df_players.iterrows():
    profile     = SEGMENTS[player["segment"]]
    reg_date    = pd.to_datetime(player["registration_date"])
    end_active  = pd.to_datetime(player["churn_date"]) if pd.notna(player["churn_date"]) else END_DATE
    active_days = max((end_active - reg_date).days, 1)
    active_months = max(active_days // 30, 1)

    # Number of deposits this player made overall
    n_deposits = max(1, int(np.random.normal(active_months * 1.8, active_months * 0.5)))

    for _ in range(n_deposits):
        txn_date = reg_date + timedelta(days=random.randint(0, active_days))
        deposit_amt = max(5, round(np.random.normal(profile["avg_deposit"], profile["deposit_std"]), 2))
        promo_type  = weighted_choice(PROMO_TYPES, PROMO_WEIGHTS)
        bonus_amt   = round(deposit_amt * random.uniform(0.1, 0.5), 2) if promo_type != "None" else 0.0

        transactions.append({
            "transaction_id":   f"TXN{txn_id:08d}",
            "player_id":        player["player_id"],
            "transaction_date": txn_date.date(),
            "transaction_type": "Deposit",
            "amount":           deposit_amt,
            "bonus_amount":     bonus_amt,
            "promo_type":       promo_type,
            "payment_method":   player["payment_method"],
        })
        txn_id += 1

df_transactions = pd.DataFrame(transactions)
print(f"  Transactions: {len(df_transactions):,} rows")

# ── 3. SESSIONS TABLE ───────────────────────────────────────────────────────
print("Generating sessions table (this takes ~60 seconds)...")

sessions = []
session_id = 1

for _, player in df_players.iterrows():
    profile    = SEGMENTS[player["segment"]]
    reg_date   = pd.to_datetime(player["registration_date"])
    end_active = pd.to_datetime(player["churn_date"]) if pd.notna(player["churn_date"]) else END_DATE
    active_days = max((end_active - reg_date).days, 1)
    active_months = max(active_days // 30, 1)

    # Sessions per month decreases after churn signal starts
    n_sessions = int(np.random.poisson(profile["sessions_per_month"] * active_months))
    n_sessions = max(1, min(n_sessions, 500))  # cap to avoid outlier blowout

    for _ in range(n_sessions):
        sess_date    = reg_date + timedelta(days=random.randint(0, active_days))
        game_type    = weighted_choice(GAME_TYPES, GAME_WEIGHTS)
        duration_min = max(2, int(np.random.exponential(25)))

        # Bets placed in session
        n_bets = max(1, int(duration_min * random.uniform(0.5, 3)))
        total_bet = round(n_bets * max(0.5, np.random.normal(profile["avg_bet"], profile["avg_bet"] * 0.4)), 2)

        # Win amount based on RTP with variance
        rtp_actual  = np.clip(np.random.normal(profile["rtp"], 0.03), 0.80, 1.05)
        total_win   = round(total_bet * rtp_actual, 2)

        # GGR = bets - wins (can be negative if player wins big)
        ggr = round(total_bet - total_win, 2)

        # Bonus used in session (subset of players use bonus)
        bonus_used = round(total_bet * random.uniform(0.0, 0.15), 2) if random.random() < profile["bonus_rate"] else 0.0

        # NGR = GGR - bonus cost (simplified; excludes taxes/fees)
        ngr = round(ggr - bonus_used, 2)

        sessions.append({
            "session_id":       f"SES{session_id:09d}",
            "player_id":        player["player_id"],
            "session_date":     sess_date.date(),
            "game_type":        game_type,
            "duration_minutes": duration_min,
            "total_bet":        total_bet,
            "total_win":        total_win,
            "ggr":              ggr,
            "bonus_used":       bonus_used,
            "ngr":              ngr,
            "device":           random.choices(["Desktop", "Mobile", "Tablet"], weights=[0.35, 0.55, 0.10])[0],
        })
        session_id += 1

df_sessions = pd.DataFrame(sessions)
print(f"  Sessions: {len(df_sessions):,} rows")

# ── SAVE TO CSV ─────────────────────────────────────────────────────────────
print("\nSaving CSVs...")
df_players.to_csv(f"{OUTPUT_DIR}/players.csv", index=False)
df_transactions.to_csv(f"{OUTPUT_DIR}/transactions.csv", index=False)
df_sessions.to_csv(f"{OUTPUT_DIR}/sessions.csv", index=False)

# ── QUICK SUMMARY ───────────────────────────────────────────────────────────
print("\n=== DATASET SUMMARY ===")
print(f"players.csv        {len(df_players):>10,} rows")
print(f"transactions.csv   {len(df_transactions):>10,} rows")
print(f"sessions.csv       {len(df_sessions):>10,} rows")
print(f"\nTotal GGR:  €{df_sessions['ggr'].sum():>15,.2f}")
print(f"Total NGR:  €{df_sessions['ngr'].sum():>15,.2f}")
print(f"Total Bets: €{df_sessions['total_bet'].sum():>15,.2f}")
print(f"\nActive players:   {df_players['is_active'].sum():,}")
print(f"Churned players:  {(~df_players['is_active']).sum():,}")
print(f"\nSegment breakdown:")
print(df_players['segment'].value_counts().to_string())
print("\nDone. Files saved to /home/claude/igaming_data/")
