import pickle

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import kagglehub
from momentum_model import MomentumTransformer

# Download latest version
path = kagglehub.dataset_download("robbypeery/college-basketball-pbp-23-24")

print("Path to dataset files:", path)
df = pd.read_csv(path + '/Colorado_pbp.csv')

df.drop(columns=['date', 'time_remaining_half', 'home_time_out_remaining', 'win_prob', 'naive_win_prob',
                 'away_time_out_remaining', 'home_favored_by', 'total_line', 'referees', 'arena_location', 'arena',
                 'attendance', 'secs_remaining_absolute'], inplace=True)

df['points_last_min_diff'] = 0

# Iterate through the DataFrame row by row
for idx, row in df.iterrows():
    current_half = row['half']
    current_secs = row['secs_remaining']

    # Filter for plays within the last 60 seconds
    mask = (df['secs_remaining'] >= current_secs - 60) & (df['secs_remaining'] <= current_secs)
    recent_plays = df[mask]

    # Sum points scored by home and away in that window
    curr_points = row['score_diff']
    start_points = recent_plays.head(1)['score_diff'].values[0]
    diff = int(curr_points) - int(start_points)

    df.loc[idx, 'points_next_min_diff'] = diff

print("Column created")
categorical_columns = [
    'home', 'away', 'half', 'description', 'action_team', 'scoring_play', 'foul',
    'shot_team', 'shot_outcome', 'shooter', 'three_pt', 'free_throw', 'possession_before', 'possession_after'
]
numerical_columns = [
    'game_id', 'play_id', 'secs_remaining', 'home_score', 'away_score', 'score_diff', 'play_length'
]

df_encoded = df.copy()
with open("category_sizes.pkl", "rb") as f:
    category_sizes = pickle.load(f)

with open("category_encoders.pkl", "rb") as f:
    category_encoders = pickle.load(f)

for col in categorical_columns:
    df_encoded[col] = df_encoded[col].map(category_encoders[col]).fillna(category_encoders[col]['MISSING']).astype(int)

sequence_length = 5
X_cat, X_num, y = [], [], []

for i in range(len(df_encoded) - sequence_length):
    window = df_encoded.iloc[i:i + sequence_length]
    next_diff = df_encoded.iloc[i + sequence_length]["points_next_min_diff"]
    label = 1 if abs(next_diff) >= 3 else 0

    X_cat.append(window[categorical_columns].values)
    X_num.append(window[numerical_columns].values)
    y.append(label)

X_cat = np.array(X_cat)
X_num = np.array(X_num)
scaler = StandardScaler()
X_num = scaler.fit_transform(X_num.reshape(-1, X_num.shape[-1])).reshape(X_num.shape)
y = np.array(y)


class PlayDataset(Dataset):
    def __init__(self, X_cat, X_num, y):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


dataset = PlayDataset(X_cat, X_num, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = MomentumTransformer(category_sizes, num_numeric=len(numerical_columns)).cuda()

try:
    model.load_state_dict(torch.load("momentum_transformer.pth"))
    model.eval()
    print("Model Found")
except FileNotFoundError:
    print("No model found. Training from scratch...")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(15):
        tqdm.write(f"Epoch {epoch + 1}")
        running_loss = 0.0
        for cat, num, label in tqdm(dataloader, desc=f"Epoch {epoch + 1} Progress"):
            cat, num, label = cat.cuda(), num.cuda(), label.cuda()
            preds, _ = model(cat, num)
            loss = criterion(preds, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "momentum_transformer.pth")
    print("Model saved after training.")

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("scaler.pkl saved.")