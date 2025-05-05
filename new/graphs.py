import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve, roc_auc_score,
    f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve

# Load results
results = pd.read_csv("momentum_eval_results.csv")
y_true = results["y_true"].values
probs = results["prob"].values
y_pred = (probs >= 0.5).astype(int)
print(1)
# Metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, probs)
print(2)
# Prepare for plots
precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, probs)
fpr, tpr, roc_thresholds = roc_curve(y_true, probs)
prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
print(3)
# Find optimal F1 threshold
best_f1, best_threshold = 0, 0
# for t in thresholds:
#     print(t)
#     pred = (probs >= t).astype(int)
#     p = precision_score(y_true, pred)
#     r = recall_score(y_true, pred)
#     f1 = (1 * p * r) / (1 * p + r + 1e-8)
#     if f1 > best_f1:
#         best_f1 = f1
best_threshold = .26
print(2)
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
# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle("Momentum Classifier Evaluation", fontsize=20)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["No Momentum", "Momentum"]).plot(ax=axs[0, 0], cmap="Blues", colorbar=False)
axs[0, 0].set_title("Confusion Matrix")

# Normalized Confusion Matrix
cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
ConfusionMatrixDisplay(cm_norm, display_labels=["No Momentum", "Momentum"]).plot(ax=axs[0, 1], cmap="Blues", colorbar=False)
axs[0, 1].set_title("Normalized Confusion Matrix")

# ROC Curve
axs[1, 0].plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
axs[1, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
axs[1, 0].set_title("ROC Curve")
axs[1, 0].set_xlabel("False Positive Rate")
axs[1, 0].set_ylabel("True Positive Rate")
axs[1, 0].legend()
axs[1, 0].grid()

# Calibration Curve
axs[1, 1].plot(prob_pred, prob_true, marker='o')
axs[1, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
axs[1, 1].set_title("Calibration Curve")
axs[1, 1].set_xlabel("Mean Predicted Probability")
axs[1, 1].set_ylabel("Fraction of Positives")
axs[1, 1].grid()
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

percentages, gains = cumulative_gain_curve(y_true, probs)

axs[2, 0].plot(percentages, gains, label="Model")
axs[2, 0].plot([0, 1], [0, 1], linestyle='--', label="Baseline (Random)")
axs[2, 0].set_xlabel("Percentage of Samples")
axs[2, 0].set_ylabel("Percentage of Runs Found")
axs[2, 0].set_title("Cumulative Gain Curve")
axs[2, 0].legend()
# Prediction Histogram
# axs[2, 0].hist(probs[y_true == 0], bins=20, alpha=0.5, label="No Momentum")
# axs[2, 0].hist(probs[y_true == 1], bins=20, alpha=0.5, label="Momentum")
# axs[2, 0].set_title("Prediction Probability Histogram")
# axs[2, 0].set_xlabel("Predicted Probability")
# axs[2, 0].set_ylabel("Count")
# axs[2, 0].legend()

# Precision Recall Curve
axs[2, 1].plot(recall_vals, precision_vals)
axs[2, 1].axvline(x=recall_vals[np.argmin(np.abs(thresholds - best_threshold))], color="red", linestyle="--")
axs[2, 1].set_title(f"Precision Recall (F1 Best @ {best_threshold:.2f})")
axs[2, 1].set_xlabel("Recall")
axs[2, 1].set_ylabel("Precision")



plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("momentum_model_eval_summary.png")
plt.show()

print("✅ Saved summary figure to momentum_model_eval_summary.png")
