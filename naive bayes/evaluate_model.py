import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Load dataset
df = sns.load_dataset("titanic").dropna(subset=["sex", "class", "embarked", "survived"])

# 2) Define features and target
FEATURES = ["sex", "class", "embarked"]
TARGET = "survived"

X = df[FEATURES].copy()
y = df[TARGET].copy()

# 3) Encode categorical features
X_enc = pd.DataFrame()
for col in FEATURES:
    cat_series = pd.Categorical(X[col])
    X_enc[col] = cat_series.codes  # numerical codes

# 4) Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.3, random_state=42, stratify=y
)

# 5) Train Naive Bayes model
nb = CategoricalNB()
nb.fit(X_train, y_train)

# 6) Predict labels + probabilities
y_pred = nb.predict(X_test)
y_prob = nb.predict_proba(X_test)[:, 1]  # P(class=1)
y_true = y_test.to_numpy()

# 7) confusion matrix calculations
TP = np.sum((y_pred == 1) & (y_true == 1))
TN = np.sum((y_pred == 0) & (y_true == 0))
FP = np.sum((y_pred == 1) & (y_true == 0))
FN = np.sum((y_pred == 0) & (y_true == 1))

# 8) Metrics
accuracy = (TP + TN) / (TP + TN + FP + FN) # כמה מהתחזיות נכונות
precision = TP / (TP + FP) # כמה מהנוסעים שחזינו ששרדו באמת שרדו
tpr = TP / (TP + FN) # כמה מהנוסעים ששרדו המודל הצליח לזהות
fpr = FP / (FP + TN) # כמה מאלו שלא שרדו המודל סיווג שחזו

# 9) Print results
print("=== Confusion Matrix ===")
print(f"TP = {TP}")
print(f"TN = {TN}")
print(f"FP = {FP}")
print(f"FN = {FN}\n")

print("=== Metrics ===")
print(f"Accuracy  = {accuracy:.3f}")
print(f"Precision = {precision:.3f}")
print(f"TPR (Recall) = {tpr:.3f}")
print(f"FPR = {fpr:.3f}")

# 10) ROC + AUC (for plotting)
fpr_vals, tpr_vals, _ = roc_curve(y_true, y_prob)
auc = roc_auc_score(y_true, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr_vals, tpr_vals, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve (Titanic)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
