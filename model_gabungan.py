# === Import Library ===
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from sklearn.model_selection import cross_val_score

# === Load Dataset ===
df = pd.read_csv("CTU-IoT-Malware-Capture-1-1conn.log.labeled.cleanned.csv")

# === Preprocessing ===
X = df.drop(columns=["label", "detailed-label", "ts", "uid", "id.orig_h", "id.resp_h"])
X = X.replace("-", 0)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# === Target Encoding (Manual Mapping) ===
# 0 = Benign (Negatif), 1 = Malicious (Positif)
label_mapping = {"Benign": 0, "Malicious": 1 , "Malicious C&C": 1}
y = df["label"].map(label_mapping)

#=== overlap features ===
for col in X.columns:
    benign_vals = df[df["label"] == "Benign"][col]
    malicious_vals = df[df["label"] == "Malicious"][col]
    malicious_vals = df[df["label"] == "Malicious C&C"][col]
    overlap = len(set(benign_vals).intersection(set(malicious_vals)))
    print(f"{col}: overlap = {overlap}")


#=== Train-Test Split (80% - 20%) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Cross-validation untuk kedua model
#== Naive Bayes Cross-validation ==
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
cv_scores_nb = cross_val_score(nb_model, X, y, cv=5)
#== Decision Tree Cross-validation ==
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
cv_scores_dt = cross_val_score(dt_model, X, y, cv=5)

print("Naive Bayes CV:", cv_scores_nb.mean())
print("Decision Tree CV:", cv_scores_dt.mean())

# === Model Naive Bayes ===
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
   
# === Model Decision Tree ===
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
# Cek overfitting
train_acc_dt = dt_model.score(X_train, y_train)
test_acc_dt = dt_model.score(X_test, y_test)

print("Decision Tree Train Accuracy:", train_acc_dt)
print("Decision Tree Test Accuracy :", test_acc_dt)

# === Ringkasan Akurasi ===
acc_nb = accuracy_score(y_test, y_pred_nb)
acc_dt = accuracy_score(y_test, y_pred_dt)

# Ambil classification report (tanpa f1-score)
report_nb = classification_report(
    y_test, y_pred_nb, target_names=["Benign", "Malicious"], output_dict=True
)

report_dt = classification_report(
    y_test, y_pred_dt, target_names=["Benign", "Malicious"], output_dict=True
)

# Buat DataFrame perbandingan Precision & Recall
summary_table = pd.DataFrame({
    "Naive Bayes (Precision)": [report_nb["Benign"]["precision"], report_nb["Malicious"]["precision"]],
    "Naive Bayes (Recall)": [report_nb["Benign"]["recall"], report_nb["Malicious"]["recall"]],
    "Naive Bayes (F1-Score)": [report_nb["Benign"]["f1-score"], report_nb["Malicious"]["f1-score"]],
    "Decision Tree (Precision)": [report_dt["Benign"]["precision"], report_dt["Malicious"]["precision"]],
    "Decision Tree (Recall)": [report_dt["Benign"]["recall"], report_dt["Malicious"]["recall"]],
    "Decision Tree (F1-Score)": [report_dt["Benign"]["f1-score"], report_dt["Malicious"]["f1-score"]]
}, index=["Benign", "Malicious"])

print("=== Perbandingan Precision, Recall & F1-Score ===")
print(summary_table)


# === Hasil Akurasi & Jumlah Prediksi ===
print("\n=== Ringkasan Akurasi ===")
print(f"Naive Bayes Accuracy : {acc_nb*100:.2f}%")
print(f"Decision Tree Accuracy : {acc_dt*100:.2f}%")
print("\nJumlah Prediksi Benar/Salah:")
print(f"Naive Bayes  -> Benar: {(y_pred_nb==y_test).sum()} | Salah: {(y_pred_nb!=y_test).sum()}")
print(f"Decision Tree -> Benar: {(y_pred_dt==y_test).sum()} | Salah: {(y_pred_dt!=y_test).sum()}")

# === Confusion Matrix ===
fig, axes = plt.subplots(1, 2, figsize=(12,5))

class_names = ["Benign", "Malicious"]

sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title("Confusion Matrix - Naive Bayes")

sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt="d", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title("Confusion Matrix - Decision Tree")


plt.show()

# === Visualisasi Pohon Keputusan ===
plt.figure(figsize=(20,10))
plot_tree(dt_model, feature_names=X.columns, class_names=["Benign", "Malicious"],
          filled=True, rounded=True, fontsize=8, max_depth=3)
plt.title("Decision Tree Visualization (Max Depth=3)")
plt.show()

# Ambil feature importance dari Decision Tree
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": dt_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

importances = pd.Series(dt_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=importances.index)
plt.title("Feature Importance - Decision Tree")
plt.show()

# === Hitung Positif (Malicious=1) & Negatif (Benign=0) ===
total_benign = (y == 0).sum()
total_malicious = (y == 1).sum()

print("\n=== Distribusi Data Asli ===")
print(f"Benign (Negatif/0)   : {total_benign}")
print(f"Malicious (Positif/1): {total_malicious}")
print(f"Total Data           : {len(y)}")

# === Prediksi Positif/Negatif dari tiap model ===
pred_nb_positive = (y_pred_nb == 1).sum()
pred_nb_negative = (y_pred_nb == 0).sum()

pred_dt_positive = (y_pred_dt == 1).sum()
pred_dt_negative = (y_pred_dt == 0).sum()

print("\n=== Distribusi Prediksi ===")
print("Naive Bayes  -> Positif (Malicious=1):", pred_nb_positive, "| Negatif (Benign=0):", pred_nb_negative)
print("Decision Tree -> Positif (Malicious=1):", pred_dt_positive, "| Negatif (Benign=0):", pred_dt_negative)
print("Distribusi label:\n", y.value_counts(normalize=True))
print(df['label'].value_counts())

#=== Cek Overfitting ===
print("Train Acc DT:", dt_model.score(X_train, y_train))
print("Test Acc DT :", dt_model.score(X_test, y_test))


# === Export Model & LabelEncoder ===
joblib.dump(nb_model, "naive_bayes_model.pkl")
joblib.dump(dt_model, "decision_tree_model.pkl")
joblib.dump(label_mapping, "label_encoder.pkl")   # << simpan encoder label

print("\nModel berhasil disimpan sebagai .pkl")
print("Ukuran naive_bayes_model.pkl :", os.path.getsize("naive_bayes_model.pkl"), "bytes")
print("Ukuran decision_tree_model.pkl:", os.path.getsize("decision_tree_model.pkl"), "bytes")
print("Ukuran label_encoder.pkl      :", os.path.getsize("label_encoder.pkl"), "bytes")