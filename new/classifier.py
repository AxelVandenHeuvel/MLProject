import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import calibration_curve
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ------------------------------
# Dataset Class
# ------------------------------

class PlayDataset(Dataset):
    def __init__(self, X_cat, X_num, y):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]

# ------------------------------
# Momentum Classifier Model
# ------------------------------

class MomentumClassifier(nn.Module):
    def __init__(self, category_sizes, num_numeric, d_model=32, num_classes=2):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            f"embed_{col}": nn.Embedding(size, d_model) for col, size in category_sizes.items()
        })
        self.embedding_keys = list(category_sizes.keys())
        self.numeric_proj = nn.Linear(num_numeric, d_model)
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, x_cat, x_num):
        emb = sum(self.embeddings[f"embed_{col}"](x_cat[:, i]) for i, col in enumerate(self.embedding_keys))
        num_proj = self.numeric_proj(x_num)
        x = emb + num_proj
        return self.output_layer(x)

# ------------------------------
# Temperature Scaling Module
# ------------------------------

class CalibratedMomentumClassifier(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, *args, **kwargs):
        logits = self.model(*args, **kwargs)
        return logits / self.temperature

    def set_temperature(self, valid_loader):
        self.eval()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for cat, num, label in train_loader:
                cat, num, label = cat.to(device), num.to(device), label.to(device)
                logits = model(cat, num)
                logits_list.append(logits)
                labels_list.append(label)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        nll_criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(logits / self.temperature, labels)
            optimizer.zero_grad()
            loss.backward()
            return loss

        optimizer.step(eval)
        return self.temperature.item()

# ------------------------------
# Load data
# ------------------------------

path = input("Enter labeled momentum CSV path: ")
df = pd.read_csv(path)
df = df.sort_values(['game_id', 'play_id'], ascending=[True,True]).reset_index(drop=True)
print("Loaded labeled data.")

# ------------------------------
# Encode categories
# ------------------------------

categorical_columns = [
    'home', 'away', 'half', 'description', 'action_team', 'scoring_play', 'foul',
    'shot_team', 'shot_outcome', 'shooter', 'three_pt', 'free_throw',
    'possession_before', 'possession_after'
]
numerical_columns = [
    'game_id', 'play_id', 'secs_remaining', 'home_score', 'away_score', 'score_diff', 'play_length'
]

category_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    category_encoders[col] = le

with open("category_encoders.pkl", "wb") as f:
    pickle.dump(category_encoders, f)

category_sizes = {col: df[col].nunique() for col in categorical_columns}

scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

X_cat = df[categorical_columns].values
X_num = df[numerical_columns].values
y = df["momentum_label"].values

X_cat_train, X_cat_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_cat, X_num, y, test_size=0.2, random_state=42
)

train_dataset = PlayDataset(X_cat_train, X_num_train, y_train)
test_dataset = PlayDataset(X_cat_test, X_num_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# ------------------------------
# Train model
# ------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MomentumClassifier(category_sizes, num_numeric=X_num.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Starting training...")

for epoch in range(25):
    model.train()
    running_loss = 0
    for cat, num, label in train_loader:
        cat, num, label = cat.to(device), num.to(device), label.to(device)
        logits = model(cat, num)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1} - Loss: {running_loss / len(train_loader):.4f}")

# ------------------------------
# Apply temperature scaling
# ------------------------------

model_with_temp = CalibratedMomentumClassifier(model).to(device)
temp = model_with_temp.set_temperature(test_loader)
print(f"Optimal temperature: {temp:.4f}")

# ------------------------------
# Evaluate calibrated model
# ------------------------------

model_with_temp.eval()
y_true, probs = [], []

with torch.no_grad():
    for cat, num, label in test_loader:
        cat, num = cat.to(device), num.to(device)
        logits = model_with_temp(cat, num)
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        probs.extend(prob)
        y_true.extend(label.numpy())

probs = np.array(probs)
y_true = np.array(y_true)
y_pred = (probs >= 0.5).astype(int)

print("---------- Test Results (Calibrated) ----------")
print("Accuracy: ", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("ROC AUC:", roc_auc_score(y_true, probs))

# ------------------------------
# Save calibrated model
# ------------------------------

torch.save(model_with_temp.state_dict(), "momentum_classifier_calibrated.pth")
print("Saved momentum_classifier_calibrated.pth ✅")

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

results_df = pd.DataFrame({
    "y_true": y_true,
    "prob": probs
})

results_df.to_csv("momentum_eval_results.csv", index=False)
print("✅ Saved momentum_eval_results.csv for visualizer.")