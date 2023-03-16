import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the anomaly scores from your experiment
anomaly_scores = pd.read_csv("anomaly_scores.csv")

# Load the ground truth labels (if available)
# ground_truth = pd.read_csv("ground_truth.csv")

# Convert timestamp column to datetime format (if needed)
anomaly_scores['timestamp'] = pd.to_datetime(anomaly_scores['timestamp'])

# Plot anomaly scores over time
plt.figure(figsize=(16,4))
plt.plot(anomaly_scores['timestamp'], anomaly_scores['score'], label='Anomaly Score')
plt.xlabel('Time')
plt.ylabel('Anomaly Score')
plt.legend()
plt.show()

# Plot distribution of anomaly scores
plt.figure(figsize=(8,6))
sns.histplot(anomaly_scores['score'], kde=True)
plt.xlabel('Anomaly Score')
plt.ylabel('Count')
plt.show()

# Plot a scatter plot of anomaly scores against another feature (if available)
# plt.figure(figsize=(8,6))
# sns.scatterplot(data=anomaly_scores, x='feature_name', y='score')
# plt.xlabel('Feature Name')
# plt.ylabel('Anomaly Score')
# plt.show()

# Compute precision-recall curve and plot it (if ground truth is available)
# from sklearn.metrics import precision_recall_curve
# precision, recall, _ = precision_recall_curve(ground_truth['label'], anomaly_scores['score'])
# plt.figure(figsize=(8,6))
# plt.plot(recall, precision)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.show()

# Compute ROC curve and plot it (if ground truth is available)
# from sklearn.metrics import roc_curve, auc
# fpr, tpr, _ = roc_curve(ground_truth['label'], anomaly_scores['score'])
# roc_auc = auc(fpr, tpr)
# plt.figure(figsize=(8,6))
# plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic (ROC) curve')
# plt.legend(loc="lower right")
# plt.show()
