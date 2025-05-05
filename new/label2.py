import pandas as pd
import numpy as np

# ------------------------------
# CONFIGURATION
# ------------------------------
SECS_BUCKET_BINS = [-1, 30, 120, 300, 600, 1200, np.inf]
SECS_BUCKET_LABELS = ["0-30", "31-120", "121-300", "301-600", "601-1200", "1200+"]

SCORE_BUCKET_BINS = [-np.inf, -10, -3, 3, 10, np.inf]
SCORE_BUCKET_LABELS = ["<-10", "-10 to -3", "-3 to 3", "3 to 10", ">10"]

PLAY_BUCKET_BINS = [-1, 5, 15, 30, 60, np.inf]
PLAY_BUCKET_LABELS = ["0-5", "6-15", "16-30", "31-60", "60+"]

# ------------------------------
# LOAD DATA
# ------------------------------
csv_path = input("Enter path to your dataset CSV: ")
df = pd.read_csv(csv_path)
print("Dataset loaded. Processing...")

# ------------------------------
# BUCKETING (vectorized)
# ------------------------------
df["secs_bucket"] = pd.cut(df["secs_remaining"], bins=SECS_BUCKET_BINS, labels=SECS_BUCKET_LABELS).astype("category")
df["score_bucket"] = pd.cut(df["score_diff"], bins=SCORE_BUCKET_BINS, labels=SCORE_BUCKET_LABELS).astype("category")
df["play_bucket"] = pd.cut(df["play_length"], bins=PLAY_BUCKET_BINS, labels=PLAY_BUCKET_LABELS).astype("category")

# ------------------------------
# PACE (rolling mean on play_length by game)
# ------------------------------
df["pace_last_5"] = df.groupby("game_id")["play_length"].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)

# ------------------------------
# RUN LENGTH (streak of same scoring team)
# ------------------------------
df["run_length"] = 0

# Vectorized run length calculation
df["scoring_play_flag"] = df["scoring_play"].fillna(0).astype(int)
df["same_team"] = (df["action_team"] == df["action_team"].shift(1)) & (df["scoring_play_flag"] == 1)
df["run_id"] = (~df["same_team"]).cumsum()
df["run_length"] = df.groupby("run_id").cumcount() + 1

# Zero run length where not scoring plays
df.loc[df["scoring_play_flag"] == 0, "run_length"] = 0

# ------------------------------
# MOMENTUM SCORE (simple formula)
# ------------------------------
df["momentum_score"] = df["run_length"] + df["pace_last_5"]

# ------------------------------
# MOMENTUM LABEL (simple threshold logic)
# ------------------------------
df["momentum_label"] = ((df["momentum_score"] >= 2) | (df["pace_last_5"] >= 3)).astype(int)

# ------------------------------
# SAVE LABELED FILE
# ------------------------------
out_path = csv_path.replace(".csv", "_with_momentum.csv")
df.to_csv(out_path, index=False)

print(f"✅ Saved labeled dataset to {out_path}")
