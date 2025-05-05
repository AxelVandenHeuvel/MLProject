import pandas as pd
import numpy as np
import pickle

from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, precision_recall_curve, PrecisionRecallDisplay
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ------------------------------
# Load labeled data
# ------------------------------

csv_path = input("Enter labeled momentum CSV path: ")
df = pd.read_csv(csv_path)

print("Loaded labeled data.")

# ------------------------------
# Create Classification Labels
# ------------------------------

# Binary momentum label: if momentum_score >= 2 or pace_last_5 >= 3
# df["momentum_label"] = ((df["momentum_score"] >= 2) | (df["pace_last_5"] >= 3)).astype(int)

print(df["momentum_label"].value_counts())

# ------------------------------
# Define categorical + numerical columns
# ------------------------------

categorical_columns = ["home", "away", "half", "description", "action_team", "scoring_play", "foul",
                       "shot_team", "shot_outcome", "shooter", "three_pt", "free_throw",
                       "possession_before", "possession_after"]

numerical_columns = ["game_id", "play_id", "secs_remaining", "home_score", "away_score", "score_diff",
                     "play_length", "pace_last_5", "last_5_play_points"]

# ------------------------------
# Encode categorical columns
# ------------------------------

category_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = df[col].astype(str)  # Ensure uniform type
    df[col] = le.fit_transform(df[col])
    category_encoders[col] = le

with open("momentum_category_encoders.pkl", "wb") as f:
    pickle.dump(category_encoders, f)

print("Saved category encoders.")

# ------------------------------
# Scale numerical columns
# ------------------------------

scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

with open("momentum_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Saved scaler.")

# ------------------------------
# Dataset + DataLoader
# ------------------------------

class MomentumDataset(Dataset):
    def __init__(self, X_cat, X_num, y):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]

X_cat = df[categorical_columns].values
X_num = df[numerical_columns].values
y = df["momentum_label"].values

X_cat_train, X_cat_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_cat, X_num, y, test_size=0.2, random_state=42
)

train_dataset = MomentumDataset(X_cat_train, X_num_train, y_train)
test_dataset = MomentumDataset(X_cat_test, X_num_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# ------------------------------
# Simple MLP Model
# ------------------------------

class MomentumMLP(nn.Module):
    def __init__(self, cat_sizes, num_numeric, hidden_size=64):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, hidden_size) for size in cat_sizes])
        self.num_proj = nn.Linear(num_numeric, hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x_cat, x_num):
        emb = sum(emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings))
        num = self.num_proj(x_num)
        x = emb + num
        return self.fc(x)

cat_sizes = [df[col].nunique() for col in categorical_columns]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MomentumMLP(cat_sizes, len(numerical_columns)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ------------------------------
# Train
# ------------------------------

print("Starting training...")

for epoch in range(25):
    model.train()
    total_loss = 0
    for cat, num, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        cat, num, label = cat.to(device), num.to(device), label.to(device)

        optimizer.zero_grad()
        logits = model(cat, num)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    tqdm.write(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

# ------------------------------
# Evaluate on Test Set
# ------------------------------

model.eval()

all_probs = []
all_y_true = []

with torch.no_grad():
    for cat, num, label in test_loader:
        cat, num = cat.to(device), num.to(device)
        logits = model(cat, num)

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)

        all_y_true.extend(label.numpy())

# Convert to numpy arrays
probs = np.array(all_probs)
y_true = np.array(all_y_true)

# Now you can calculate all your metrics safely:
accuracy = accuracy_score(y_true, probs >= 0.5)
precision = precision_score(y_true, probs >= 0.5)
recall = recall_score(y_true, probs >= 0.5)
f1 = f1_score(y_true, probs >= 0.5)
roc_auc = roc_auc_score(y_true, probs)

print("---------- Test Results ----------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")


# ------------------------------
# Save Model
# ------------------------------

torch.save(model.state_dict(), "momentum_classifier.pth")
print("Saved momentum_classifier.pth ✅")

y_pred = (probs >= 0.5).astype(int)


cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Momentum", "Momentum"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# ----------------------------
# 2. ROC Curve
# ----------------------------

fpr, tpr, roc_thresholds = roc_curve(y_true, probs)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
roc_display.plot()
plt.title("ROC Curve")
plt.savefig("roc_curve.png")
plt.close()

# ----------------------------
# 3. Precision-Recall Curve
# ----------------------------

precision, recall, pr_thresholds = precision_recall_curve(y_true, probs)
pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
pr_display.plot()
plt.title("Precision-Recall Curve")
plt.savefig("precision_recall_curve.png")
plt.close()

# ----------------------------
# 4. F1 Score vs Threshold
# ----------------------------

f1_scores = []

for thresh in pr_thresholds:
    preds_thresh = (probs >= thresh).astype(int)
    f1 = f1_score(y_true, preds_thresh)
    f1_scores.append(f1)

plt.plot(pr_thresholds, f1_scores)
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 Score vs Threshold")
plt.savefig("f1_score_vs_threshold.png")
plt.close()

# ----------------------------
# 5. Prediction Probability Histogram
# ----------------------------

plt.hist(probs, bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Prediction Probability Distribution")
plt.savefig("probability_histogram.png")
plt.close()

# ----------------------------
# 6. Calibration Curve
# ----------------------------

prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve")
plt.savefig("calibration_curve.png")
plt.close()

print("✅ All graphs saved.")
