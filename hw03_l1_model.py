"""
IDS 570 Homework 3, Part II: L1 (lasso) model.

Reproduces the Week 09 tutorial classifier (step11_train_classifier.py):
load saved JSON -> TF-IDF -> logistic regression (L1 penalty instead of L2) -> evaluate.
"""
from pathlib import Path
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

MODEL_NAME = "L1"
DATA_DIR = Path("data")
TOP_N = 15

# 1. Load the SAME train/test files saved by the tutorial (Step 9)
with open(DATA_DIR / "train_core_vs_neg.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open(DATA_DIR / "test_core_vs_neg.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

X_train_texts = [t for (t, y) in train_data]
y_train = [y for (t, y) in train_data]
X_test_texts = [t for (t, y) in test_data]
y_test = [y for (t, y) in test_data]

print(f"=== {MODEL_NAME} model ===")
print("Train size:", len(X_train_texts))
print("Test size :", len(X_test_texts))

# 2. The SAME TF-IDF settings as the tutorial (fit on train only)
vectorizer = TfidfVectorizer(lowercase=True, min_df=5, max_df=0.9)
X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)
print("TF-IDF matrix shapes: train", X_train.shape, "test", X_test.shape)

# 3. The ONE key change: L1 penalty (l1_ratio=1) with a solver that supports L1 (liblinear).
#    random_state fixes the order liblinear visits the data in, so reruns give the same result.
clf = LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=2000, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# 4. Evaluation outputs
print("\n1. Confusion matrix (rows = actual NEG, CORE; columns = predicted NEG, CORE):")
print(confusion_matrix(y_test, y_pred))

print("\n2. Classification report:")
print(classification_report(y_test, y_pred, digits=3))

auc = roc_auc_score(y_test, y_prob)
print("3. ROC AUC:", round(auc, 4))

# 5. Sparsity diagnostic: how many words have a weight that is not exactly zero
coefs = clf.coef_[0]
n_total = coefs.shape[0]
n_nonzero = int(np.count_nonzero(coefs))
print(f"\n4. Non-zero coefficients: {n_nonzero} of {n_total} "
      f"({100 * n_nonzero / n_total:.2f}%)")

# 6. What the model learned: top positive and negative words
feature_names = vectorizer.get_feature_names_out()
order = np.argsort(coefs)  # smallest (most negative) to largest

print(f"\nTop {TOP_N} positive-weight words (most predictive of CORE = 1):")
for i in order[::-1][:TOP_N]:
    print(f"  {feature_names[i]:20s} {coefs[i]: .4f}")

neg_idx = [i for i in order if coefs[i] < 0][:TOP_N]
print(f"\nTop {TOP_N} negative-weight words (most predictive of NEG = 0); "
      f"{sum(coefs < 0)} words have a weight below zero:")
for i in neg_idx:
    print(f"  {feature_names[i]:20s} {coefs[i]: .4f}")

# 7. Save test-set predictions so the misclassified chunks can be read by a human
records = [
    {"id": k, "label": int(y_test[k]), "pred": int(y_pred[k]),
     "prob_core": round(float(y_prob[k]), 4), "text": X_test_texts[k]}
    for k in range(len(y_test))
]
out_path = DATA_DIR / f"hw03_{MODEL_NAME.lower()}_test_predictions.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False)
n_fp = sum(1 for r in records if r["label"] == 0 and r["pred"] == 1)
n_fn = sum(1 for r in records if r["label"] == 1 and r["pred"] == 0)
print(f"\nSaved test predictions to {out_path} "
      f"(false positives: {n_fp}, false negatives: {n_fn})")
