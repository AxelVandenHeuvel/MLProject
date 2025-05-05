import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# ------------------------------
# CONFIG - Update these columns based on your model
# ------------------------------
categorical_columns = [
    'home', 'away', 'half', 'description', 'action_team', 'scoring_play', 'foul',
    'shot_team', 'shot_outcome', 'shooter', 'three_pt', 'free_throw', 'possession_before', 'possession_after'
]

numerical_columns = [
    'game_id', 'play_id', 'secs_remaining', 'home_score', 'away_score', 'score_diff', 'play_length'
]

# ------------------------------
# Load dataset
# ------------------------------

csv_path = input("Enter path to your dataset CSV: ")
df = pd.read_csv(csv_path)
print("Dataset loaded. Processing...")

# ------------------------------
# Build category encoders
# ------------------------------
category_encoders = {}

for col in categorical_columns:
    df[col] = df[col].fillna("MISSING").astype("category")

    if "MISSING" not in df[col].cat.categories:
        df[col] = df[col].cat.add_categories(["MISSING"])

    category_encoders[col] = dict(zip(df[col].cat.categories, range(len(df[col].cat.categories))))
    df[col] = df[col].cat.codes

# Save category encoders
with open("category_encoders.pkl", "wb") as f:
    pickle.dump(category_encoders, f)

print("category_encoders.pkl saved.")

# ------------------------------
# Build category sizes
# ------------------------------
category_sizes = {col: len(enc) for col, enc in category_encoders.items()}

with open("category_sizes.pkl", "wb") as f:
    pickle.dump(category_sizes, f)

print("category_sizes.pkl saved.")

# ------------------------------
# Build scaler for numeric columns
# ------------------------------
scaler = StandardScaler()
scaler.fit(df[numerical_columns])

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("scaler.pkl saved.")

print("✅ All files generated successfully!")
