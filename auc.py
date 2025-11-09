# Import library
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("CTU-IoT-Malware-Capture-1-1conn.log.labeled.cleanned.csv")

# 1️⃣ Membuat dataset tidak seimbang (90% malicious, 10% benign)
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_classes=2,
    weights=[0.1, 0.9],   # 10% benign, 90% malicious
    random_state=42
)

# 2️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3️⃣ Model Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# 4️⃣ Prediksi
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 5️⃣ Evaluasi metrik
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Akurasi     : {acc:.4f}")
print(f"F1-Score    : {f1:.4f}")
print(f"ROC-AUC     : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))

# 6️⃣ Visualisasi Kurva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0,1], [0,1], color='red', linestyle='--')
plt.title('ROC Curve for Malicious vs Benign Classification')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
