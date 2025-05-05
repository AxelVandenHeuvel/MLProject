import pickle
from collections import Counter
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay, precision_recall_curve, fbeta_score, PrecisionRecallDisplay, auc, roc_curve
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
from momentum_model import MomentumTransformer


# ------------------------------
# Helper function to add points_next_min_diff
# ------------------------------
def cumulative_gain_curve(y_true, y_probas, pos_label=1):
    y_true = np.asarray(y_true)
    y_probas = np.asarray(y_probas)

    # Sort by predicted probabilities
    sorted_indices = np.argsort(y_probas)[::-1]
    y_true_sorted = y_true[sorted_indices]

    total_positives = np.sum(y_true == pos_label)
    cumulative_true_positives = np.cumsum(y_true_sorted == pos_label)

    percentages = np.arange(1, len(y_true) + 1) / len(y_true)
    gains = cumulative_true_positives / total_positives

    return percentages, gains

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
if path == "":
    path = "C:\\Users\\zacha\\.cache\\kagglehub\\datasets\\robbypeery\\college-basketball-pbp-23-24\\versions\\18\\Colorado_pbp.csv"
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

class_sample_count = np.array([ (y_train==t).sum() for t in [0,1] ])
weights = 1. / class_sample_count
samples_weights = np.array([weights[t] for t in y_train])
sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=32)

# ------------------------------
# Train Model
# ------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2, reduction='mean'):
        super().__init__()
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        targets = targets.to(inputs.device)

        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)

        if isinstance(self.alpha, torch.Tensor):
            # If alpha is per class → get correct alpha for each target
            alpha_t = self.alpha.to(inputs.device)[targets]
        else:
            # If scalar
            alpha_t = self.alpha

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


# weights = torch.tensor([0.25, 2.7], device=device)  # hand tuned
# criterion = nn.CrossEntropyLoss(weight=weights)
class_counts = Counter(y_train.numpy())
total = sum(class_counts.values())

alpha_run = total / class_counts[1]
alpha_no_run = total / class_counts[0]

alpha = torch.tensor([alpha_no_run, alpha_run], device=device)

criterion = FocalLoss(alpha=alpha, gamma=4)

print("Starting training...")

for epoch in range(25):
    model.train()
    running_loss = 0.0
    for cat, num, label in tqdm(train_loader, desc=f"Epoch {epoch + 1} Progress"):
        cat = cat.to(device)
        num = num.to(device)
        label = label.to(device)

        preds, _ = model(cat, num)

        loss = criterion(preds, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
    if avg_loss < 0.01:
        break

# --------------------
# Get probabilities for positive class
# --------------------
model.eval()
probs = []
y_true = []

with torch.no_grad():
    for cat, num, label in test_loader:
        cat, num = cat.to(device), num.to(device)
        logits, _ = model(cat, num)
        prob = torch.softmax(logits, dim=1)[:, 1]  # probability of "Run"
        probs.extend(prob.cpu().numpy())
        y_true.extend(label.numpy())

probs = np.array(probs)
y_true = np.array(y_true)

# --------------------
# Calculate Precision-Recall curve and PR AUC
# --------------------
precision, recall, thresholds = precision_recall_curve(y_true, probs)
pr_auc = auc(recall, precision)

print(f"PR AUC: {pr_auc:.4f}")

# --------------------
# Find best threshold for F2 Score (favoring recall)
# --------------------
f2_scores = []

for t in thresholds:
    y_pred = (probs >= t).astype(int)
    f2 = fbeta_score(y_true, y_pred, beta=1)
    f2_scores.append(f2)

best_threshold = thresholds[np.argmax(f2_scores)]
print(f"Optimal F2 Threshold: {best_threshold:.2f}")

# --------------------
# Final predictions using best threshold
# --------------------
y_pred = (probs >= best_threshold).astype(int)

accuracy = accuracy_score(y_true, y_pred)
precision_final = precision_score(y_true, y_pred)
recall_final = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
f2 = fbeta_score(y_true, y_pred, beta=2)

print("---------- Test Results (F2 Optimized) ----------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision_final:.4f}")
print(f"Recall:    {recall_final:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"F2 Score:  {f2:.4f}")

# --------------------
# Confusion Matrix
# --------------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Run", "Run"])
disp.plot(cmap='Blues')
plt.title("Momentum Transformer - Test Set Confusion Matrix (F2 Optimized)")
plt.show()

# --------------------
# Precision-Recall Curve
# --------------------
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.title("Precision-Recall Curve")
plt.show()


cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues')
plt.title("Normalized Confusion Matrix")
plt.show()


prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curve")
plt.show()

plt.hist([probs[y_true==0], probs[y_true==1]], bins=50, label=['No Run', 'Run'], stacked=True)
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Prediction Distribution by True Class")
plt.legend()
plt.show()

precision, recall, thresholds = precision_recall_curve(y_true, probs)

plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold")
plt.legend()
plt.show()

plt.hist([probs[y_true == 0], probs[y_true == 1]], bins=50, label=["No Run", "Run"], alpha=0.7)
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Prediction Score Distribution")
plt.legend()
plt.show()

percentages, gains = cumulative_gain_curve(y_true, probs)

plt.plot(percentages, gains, label="Model")
plt.plot([0, 1], [0, 1], linestyle='--', label="Baseline (Random)")
plt.xlabel("Percentage of Samples")
plt.ylabel("Percentage of Runs Found")
plt.title("Cumulative Gain Curve")
plt.legend()
plt.show()