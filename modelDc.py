# === Import Library ===
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import Ridge, Lasso

# === Load Dataset ===
df = pd.read_csv("CTU-IoT-Malware-Capture-1-1conn.log.labeled.cleanned.csv")

# === Preprocessing ===
X = df.drop(columns=[
    "label", "detailed-label", "ts", "uid",  
    "id.orig_h", "id.resp_h",
])
X = X.replace("-", 0)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# === Target Encoding (Manual Mapping) ===
label_mapping = {"Benign": 0, "Malicious": 1}
y = df["label"].map(label_mapping)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Regularization (Ridge / Lasso) ===
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
ridge.fit(X_train, y_train)

# === Model Decision Tree (utama untuk evaluasi) ===
dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5
)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# === Evaluasi ===
acc_dt = accuracy_score(y_test, y_pred_dt)
report_dt = classification_report(
    y_test, y_pred_dt, target_names=["Benign", "Malicious"], output_dict=True
)

summary_table = pd.DataFrame({
    "Precision": [report_dt["Benign"]["precision"], report_dt["Malicious"]["precision"]],
    "Recall": [report_dt["Benign"]["recall"], report_dt["Malicious"]["recall"]],
    "F1-Score": [report_dt["Benign"]["f1-score"], report_dt["Malicious"]["f1-score"]]
}, index=["Benign", "Malicious"])

print("=== Ringkasan Akurasi ===")
print(f"Decision Tree Accuracy : {acc_dt*100:.2f}%")
print("\n=== Precision, Recall & F1-Score ===")
print(summary_table)

# === Hyperparameter Tuning ===
param_grid = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train, y_train)
print("\nBest Params Decision Tree:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

# === Confusion Matrix ===
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred_dt),
            annot=True, fmt="d", cmap="Greens",
            xticklabels=["Benign", "Malicious"],
            yticklabels=["Benign", "Malicious"])
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# === Visualisasi Pohon Keputusan (dibatasi hanya 3 level) ===
dt_viz = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_viz.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(
    dt_viz,
    feature_names=X.columns,
    class_names=["Benign", "Malicious"],
    filled=True,
    rounded=True,
    fontsize=10,
    impurity=False,   # sembunyikan nilai Gini
    proportion=False,
    precision=2
)
plt.title("Simplified Decision Tree (Max Depth = 3)")
plt.show()

# === Feature Importance ===
importances = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=importances.index)
plt.title("Feature Importance - Decision Tree")
plt.show()

# === Distribusi Label ===
total_benign = (y == 0).sum()
total_malicious = (y == 1).sum()

print("\n=== Distribusi Data Asli ===")
print(f"Benign (Negatif/0)   : {total_benign}")
print(f"Malicious (Positif/1): {total_malicious}")
print(f"Total Data           : {len(y)}")
