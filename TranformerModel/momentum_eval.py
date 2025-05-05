import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from momentum_model import MomentumTransformer, load_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ------------------------------
# Load Dataset
# ------------------------------

path = input("Enter path to your dataset CSV: ")
df = pd.read_csv(path)

# ------------------------------
# Load model
# ------------------------------

model = load_model()
device = next(model.parameters()).device

# ------------------------------
# Prepare Data
# ------------------------------
df = pd.read_csv(path)

# Add points_next_min_diff calculation
df['points_next_min_diff'] = 0

for idx, row in df.iterrows():
    current_secs = row['secs_remaining']

    mask = (df['secs_remaining'] >= current_secs - 60) & (df['secs_remaining'] <= current_secs)
    recent_plays = df[mask]

    curr_points = row['score_diff']
    start_points = recent_plays.head(1)['score_diff'].values[0]
    diff = int(curr_points) - int(start_points)

    df.loc[idx, 'points_next_min_diff'] = diff

print("points_next_min_diff created!")

X_cat, X_num, y_true = model.process_dataframe(df, make_labels=True)

dataset = torch.utils.data.TensorDataset(X_cat, X_num, y_true)
dataloader = DataLoader(dataset, batch_size=64)

y_pred = []
y_true_all = []

model.eval()
with torch.no_grad():
    for cat, num, label in tqdm(dataloader, desc="Running Evaluation"):
        cat = cat.to(device)
        num = num.to(device)
        label = label.to(device)

        logits, _ = model(cat, num)
        pred_class = torch.argmax(logits, dim=1)

        y_true_all.extend(label.cpu().numpy())
        y_pred.extend(pred_class.cpu().numpy())

y_true_all = np.array(y_true_all)
y_pred = np.array(y_pred)

# ------------------------------
# Metrics
# ------------------------------

accuracy = accuracy_score(y_true_all, y_pred)
precision = precision_score(y_true_all, y_pred)
recall = recall_score(y_true_all, y_pred)
f1 = f1_score(y_true_all, y_pred)

print("---------- Evaluation Results ----------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# ------------------------------
# Confusion Matrix
# ------------------------------

cm = confusion_matrix(y_true_all, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Run", "Run"])
disp.plot(cmap='Blues')
plt.title("Momentum Prediction - Confusion Matrix")
plt.show()
