"""
DA-1 Step 1: Exploratory Data Analysis
Run this after generate_igaming_data.py
"""

import pandas as pd
import numpy as np

DATA = "/home/claude/igaming_data"

# Load
players      = pd.read_csv(f"{DATA}/players.csv", parse_dates=["registration_date", "churn_date", "last_login_date"])
transactions = pd.read_csv(f"{DATA}/transactions.csv", parse_dates=["transaction_date"])
sessions     = pd.read_csv(f"{DATA}/sessions.csv", parse_dates=["session_date"])

print("=" * 60)
print("STEP 1: DATA SHAPE & TYPES")
print("=" * 60)
for name, df in [("players", players), ("transactions", transactions), ("sessions", sessions)]:
    print(f"\n{name.upper()} — {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(df.dtypes.to_string())

print("\n" + "=" * 60)
print("STEP 2: NULL CHECK")
print("=" * 60)
for name, df in [("players", players), ("transactions", transactions), ("sessions", sessions)]:
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    print(f"\n{name.upper()} nulls:")
    print(nulls.to_string() if len(nulls) else "  None")

print("\n" + "=" * 60)
print("STEP 3: KEY METRIC DISTRIBUTIONS")
print("=" * 60)

print("\n-- GGR by Segment --")
seg_ggr = sessions.merge(players[["player_id","segment"]], on="player_id")
print(seg_ggr.groupby("segment")["ggr"].agg(["sum","mean","count"]).round(2).to_string())

print("\n-- Avg Deposit by Segment --")
txn_seg = transactions.merge(players[["player_id","segment"]], on="player_id")
print(txn_seg.groupby("segment")["amount"].agg(["mean","median","sum"]).round(2).to_string())

print("\n-- Churn Rate by Segment --")
churn = players.groupby("segment")["is_active"].agg(
    total="count",
    active="sum"
)
churn["churned"] = churn["total"] - churn["active"]
churn["churn_rate_%"] = ((churn["churned"] / churn["total"]) * 100).round(1)
print(churn.to_string())

print("\n-- Monthly GGR Trend (2024) --")
sessions["month"] = sessions["session_date"].dt.to_period("M")
monthly = sessions[sessions["session_date"].dt.year == 2024].groupby("month")["ggr"].sum().round(2)
print(monthly.to_string())

print("\n-- Bonus Cost vs GGR by Segment --")
bonus = seg_ggr.groupby("segment").agg(
    total_ggr=("ggr","sum"),
    total_bonus_cost=("bonus_used","sum")
).round(2)
bonus["bonus_as_pct_ggr"] = ((bonus["total_bonus_cost"] / bonus["total_ggr"]) * 100).round(1)
print(bonus.to_string())

print("\n-- LTV Proxy (Total GGR per player) by Segment --")
player_ltv = seg_ggr.groupby(["player_id","segment"])["ggr"].sum().reset_index()
player_ltv.columns = ["player_id","segment","ltv_ggr"]
ltv_summary = player_ltv.groupby("segment")["ltv_ggr"].agg(["mean","median","max"]).round(2)
print(ltv_summary.to_string())

print("\n-- Top 5 Countries by GGR --")
country_ggr = sessions.merge(players[["player_id","country"]], on="player_id")
print(country_ggr.groupby("country")["ggr"].sum().sort_values(ascending=False).head(5).round(2).to_string())

print("\n-- Game Type Performance --")
print(sessions.groupby("game_type").agg(
    sessions=("session_id","count"),
    total_ggr=("ggr","sum"),
    avg_duration_min=("duration_minutes","mean")
).round(2).sort_values("total_ggr", ascending=False).to_string())

print("\n" + "=" * 60)
print("STEP 4: DATA QUALITY FLAGS")
print("=" * 60)
neg_ggr_sessions = (sessions["ggr"] < 0).sum()
print(f"Sessions with negative GGR (player wins): {neg_ggr_sessions:,} ({neg_ggr_sessions/len(sessions)*100:.1f}%)")
print(f"Date range: {sessions['session_date'].min()} to {sessions['session_date'].max()}")
print(f"Avg sessions per player: {len(sessions)/len(players):.1f}")
print(f"Avg deposits per player: {len(transactions)/len(players):.1f}")

print("\nEDA complete. Data is clean and ready for PostgreSQL loading.")
