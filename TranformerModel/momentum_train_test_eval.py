import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from momentum_model import MomentumTransformer

# ------------------------------
# Helper function to add points_next_min_diff
# ------------------------------

def add_points_next_min_diff(df):
    df = df.copy()

    results = []

    for game_id, game_df in df.groupby("game_id"):
        game_df = game_df.sort_values("secs_remaining").reset_index(drop=True)

        game_df["lookup_secs"] = game_df["secs_remaining"] - 60

        lookup = game_df[["secs_remaining", "score_diff"]].rename(columns={
            "secs_remaining": "lookup_secs",
            "score_diff": "start_score_diff"
        })

        game_df = pd.merge_asof(
            game_df,
            lookup,
            on="lookup_secs",
            direction="backward"
        )

        # Enforce strict 60 sec window
        game_df["time_gap"] = game_df["secs_remaining"] - game_df["lookup_secs"]
        game_df["start_score_diff"] = game_df["start_score_diff"].where(game_df["time_gap"] <= 60,
                                                                        game_df["score_diff"])

        game_df["points_next_min_diff"] = game_df["score_diff"] - game_df["start_score_diff"]

        results.append(game_df)

    df_result = pd.concat(results).sort_index()

    return df_result


# ------------------------------
# Dataset class
# ------------------------------

class PlayDataset(Dataset):
    def __init__(self, X_cat, X_num, y):
        self.X_cat = X_cat if isinstance(X_cat, torch.Tensor) else torch.tensor(X_cat, dtype=torch.long)
        self.X_num = X_num if isinstance(X_num, torch.Tensor) else torch.tensor(X_num, dtype=torch.float32)
        self.y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]

# ------------------------------
# Load dataset
# ------------------------------

path = input("Enter path to your dataset CSV: ")
df = pd.read_csv(path)
df.drop(columns=['date','time_remaining_half', 'home_time_out_remaining', 'win_prob', 'naive_win_prob', 'away_time_out_remaining','home_favored_by','total_line','referees','arena_location','arena','attendance','secs_remaining_absolute'], inplace=True)
df = add_points_next_min_diff(df)

print("Points next min diff created and dataset loaded.")

# ------------------------------
# Load category sizes
# ------------------------------

with open("category_sizes.pkl", "rb") as f:
    category_sizes = pickle.load(f)

# ------------------------------
# Process dataframe
# ------------------------------

model = MomentumTransformer(category_sizes, num_numeric=len(MomentumTransformer.numerical_columns))
X_cat, X_num, y = model.process_dataframe(df, make_labels=True)

# ------------------------------
# Train-test split
# ------------------------------

X_cat_train, X_cat_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_cat, X_num, y, test_size=0.2, random_state=42
)

train_dataset = PlayDataset(X_cat_train, X_num_train, y_train)
test_dataset = PlayDataset(X_cat_test, X_num_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# ------------------------------
# Train Model
# ------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MomentumTransformer(category_sizes, num_numeric=len(MomentumTransformer.numerical_columns)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Starting training...")

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for cat, num, label in tqdm(train_loader, desc=f"Epoch {epoch + 1} Progress"):
        cat, num, label = cat.to(device), num.to(device), label.to(device)
        preds, _ = model(cat, num)
        loss = criterion(preds, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

# ------------------------------
# Evaluate on Test Set
# ------------------------------

model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for cat, num, label in test_loader:
        cat, num = cat.to(device), num.to(device)
        logits, _ = model(cat, num)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(label.numpy())
        y_pred.extend(preds.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("---------- Test Results ----------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# ------------------------------
# Confusion Matrix
# ------------------------------

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Run", "Run"])
disp.plot(cmap='Blues')
plt.title("Momentum Transformer - Test Set Confusion Matrix")
plt.show()
