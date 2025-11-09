# === Import Library ===
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# === Load Dataset ===
df = pd.read_csv("CTU-IoT-Malware-Capture-1-1conn.log.labeled.cleanned.csv")

# === Preprocessing ===
X = df.drop(columns=["label", "detailed-label", "ts", "uid", "id.orig_h", "id.resp_h"])
X = X.replace("-", 0)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# Target Encoding
label_mapping = {"Benign": 0, "Malicious": 1}
y = df["label"].map(label_mapping)

for col in X.columns:
    benign_vals = df[df["label"] == "Benign"][col]
    malicious_vals = df[df["label"] == "Malicious"][col]
    overlap = len(set(benign_vals).intersection(set(malicious_vals)))
    print(f"{col}: overlap = {overlap}")



# Train-Test Split (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# === Training Naive Bayes ===
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Prediksi
y_pred = nb_model.predict(X_test)

# === Evaluasi ===
acc = accuracy_score(y_test, y_pred)
print(f"\n=== Akurasi Naive Bayes ===\nAkurasi : {acc*100:.2f}%")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malicious"],
            yticklabels=["Benign", "Malicious"])
plt.title("Confusion Matrix - Naive Bayes")
plt.show()

# === Cross Validation (5-Fold) ===
cv_scores = cross_val_score(nb_model, X, y, cv=5)
print("\n=== Cross Validation (5-Fold) ===")
print("Akurasi per fold:", cv_scores)
print("Rata-rata akurasi:", cv_scores.mean())
print(df['label'].value_counts())
