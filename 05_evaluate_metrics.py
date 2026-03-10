import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

print("Starting Step 5: System Evaluation & Threshold Calibration...")

# Load results from Step 4
try:
    results = np.load("data/test_results.npy")
    true_labels = results[:, 0]
    predicted_scores = results[:, 1]
except:
    print("Run Step 4 first!")
    exit()

# Calculate FAR and FRR
fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
frr = 1 - tpr

eer_index = np.nanargmin(np.absolute((frr - fpr)))
eer_threshold = thresholds[eer_index]
eer = fpr[eer_index]

print(f"CALIBRATION COMPLETE: Equal Error Rate (EER) is {eer:.4f}")
print(f"OPTIMAL THRESHOLD: {eer_threshold:.4f}")

# Plotting
plt.figure(figsize=(12, 5))

# 1. ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.plot([0, 1],[0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Acceptance Rate (FAR)')
plt.ylabel('True Positive Rate (1 - FRR)')
plt.title('ROC Curve')
plt.legend()

# 2. Score Distribution
plt.subplot(1, 2, 2)
plt.hist([s for l, s in zip(true_labels, predicted_scores) if l == 1], alpha=0.6, label='Positive Pairs (Same)')
plt.hist([s for l, s in zip(true_labels, predicted_scores) if l == 0], alpha=0.6, label='Negative Pairs (Diff)')
plt.axvline(eer_threshold, color='red', linestyle='dashed', label=f'Threshold: {eer_threshold:.2f}')
plt.title('Cosine Similarity Distribution')
plt.legend()

plt.tight_layout()
plt.savefig("data/final_evaluation_report.png")
print("Saved final graphs to 'data/final_evaluation_report.png'.")
plt.show()
