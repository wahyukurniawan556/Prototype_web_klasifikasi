import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree

# === Load Model ===
nb_model = joblib.load("naive_bayes_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
le = joblib.load("label_encoder.pkl")  # << load encoder label

st.title("Prototype Klasifikasi Malware Berdasarkan Aktivitas Jaringan")
st.write("Upload file `.csv` untuk klasifikasi dan evaluasi model.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # === Load data ===
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Awal")
    st.dataframe(df.head())

    # === Preprocessing ===
    X = df.drop(columns=["label","detailed-label","ts","uid","id.orig_h","id.resp_h"], errors="ignore")
    X = X.replace("-",0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    if "label" in df.columns:
        y = df["label"]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_
    else:
        y = None
        y_encoded = None
        class_names = ["Benign", "Malicious"]

    if st.button("🔎 Jalankan Klasifikasi"):
        # === Prediksi ===
        pred_nb = nb_model.predict(X)
        pred_dt = dt_model.predict(X)

        df_result = df.copy()
        df_result["Prediksi_NaiveBayes"] = pred_nb
        df_result["Prediksi_DecisionTree"] = pred_dt

        st.subheader("📊 Hasil Klasifikasi")
        st.dataframe(df_result.head())

        # === Evaluasi hanya jika ada label asli ===
        if y is not None:
            st.subheader("📈 Evaluasi Model")

            # === Akurasi ===
            acc_nb = accuracy_score(y_encoded, pred_nb)
            acc_dt = accuracy_score(y_encoded, pred_dt)

            # === Precision & Recall ===
            report_nb = classification_report(y_encoded, pred_nb, target_names=class_names, output_dict=True)
            report_dt = classification_report(y_encoded, pred_dt, target_names=class_names, output_dict=True)

            # === Tabel Evaluasi NB ===
            st.write("### 📌 Naive Bayes")
            df_nb = pd.DataFrame({
                "Precision": [report_nb[c]["precision"] for c in class_names],
                "Recall": [report_nb[c]["recall"] for c in class_names]
            }, index=class_names)
            df_nb.loc["Accuracy"] = [acc_nb, acc_nb]
            st.table(df_nb)

            # === Tabel Evaluasi DT ===
            st.write("### 📌 Decision Tree")
            df_dt = pd.DataFrame({
                "Precision": [report_dt[c]["precision"] for c in class_names],
                "Recall": [report_dt[c]["recall"] for c in class_names]
            }, index=class_names)
            df_dt.loc["Accuracy"] = [acc_dt, acc_dt]
            st.table(df_dt)

            # === Confusion Matrix NB ===
            st.write("### Confusion Matrix - Naive Bayes")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_encoded, pred_nb), annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names, ax=ax)
            st.pyplot(fig)

            # === Confusion Matrix DT ===
            st.write("### Confusion Matrix - Decision Tree")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_encoded, pred_dt), annot=True, fmt="d", cmap="Greens",
                        xticklabels=class_names, yticklabels=class_names, ax=ax)
            st.pyplot(fig)

            # === Pohon Keputusan ===
            st.write("### 🌳 Visualisasi Pohon Keputusan (max_depth=3)")
            fig, ax = plt.subplots(figsize=(12,6))
            plot_tree(dt_model, feature_names=X.columns, class_names=class_names,
                      filled=True, rounded=True, fontsize=8, max_depth=3, ax=ax)
            st.pyplot(fig)

                    # === Summary Akurasi NB vs DT ===
            st.write("## 📊 Perbandingan Akurasi Model")

            summary_acc = pd.DataFrame({
                "Model": ["Naive Bayes", "Decision Tree"],
                "Accuracy": [acc_nb, acc_dt]
            })

            st.table(summary_acc)

            # === Grafik Perbandingan Akurasi ===
            fig, ax = plt.subplots()
            sns.barplot(x="Model", y="Accuracy", data=summary_acc, palette="viridis", ax=ax)
            ax.set_ylim(0, 1)  # supaya skala dari 0 sampai 1
            ax.set_title("Perbandingan Akurasi Naive Bayes vs Decision Tree")
            for i, v in enumerate(summary_acc["Accuracy"]):
                ax.text(i, v + 0.01, f"{v*100:.2f}%", ha="center", fontsize=10)
            st.pyplot(fig)
               # Jika ada label asli di dataset, hitung akurasi juga
        if "label" in df.columns:
            y_true = le.transform(df["label"])
            acc_nb = accuracy_score(y_true, nb_model.predict(X))
            acc_dt = accuracy_score(y_true, dt_model.predict(X))

            # Tabel akurasi
            acc_table = pd.DataFrame({
                "Model": ["Naive Bayes", "Decision Tree"],
                "Akurasi": [f"{acc_nb*100:.2f}%", f"{acc_dt*100:.2f}%"]
            })
            st.subheader("📊 Perbandingan Akurasi Model")
            st.table(acc_table)
   # === Hitung Positif (Malicious) & Negatif (Benign) ===
            total_benign = (y == 0).sum()
            total_malicious = (y == 1).sum()

            print("\n=== Distribusi Data Asli ===")
            print(f"Benign (Negatif/0)   : {total_benign}")
            print(f"Malicious (Positif/1): {total_malicious}")
            print(f"Total Data           : {len(y)}")

            # === Prediksi Positif/Negatif dari tiap model ===
            pred_nb_positive = (pred_nb == 1).sum()
            pred_nb_negative = (pred_nb == 0).sum()

            pred_dt_positive = (pred_dt == 1).sum()
            pred_dt_negative = (pred_dt == 0).sum()

            print("\n=== Distribusi Prediksi ===")
            print("Naive Bayes  -> Positif :", pred_nb_positive, "| Negatif (Benign=0):", pred_nb_negative)
            print("Decision Tree -> Positif :", pred_dt_positive, "| Negatif (Benign=0):", pred_dt_negative)
        # === Statistik Prediksi ===
        st.subheader("📊 Statistik Prediksi")
        total_samples = len(df_result)
        nb_positive = (df_result["Prediksi_NaiveBayes"] == 1).sum()
        nb_negative = (df_result["Prediksi_NaiveBayes"] == 0).sum()
        dt_positive = (df_result["Prediksi_DecisionTree"] == 1).sum()
        dt_negative = (df_result["Prediksi_DecisionTree"] == 0).sum()
        stats_data = {
            "Model": ["Naive Bayes", "Decision Tree"],
            "Total Samples": [total_samples, total_samples],
            "Prediksi Positif (Malicious)": [nb_positive, dt_positive],
            "Prediksi Negatif (Benign)": [nb_negative, dt_negative]
        }
        stats_df = pd.DataFrame(stats_data)
        st.table(stats_df)    
        # === Download hasil klasifikasi ===
        csv = df_result.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Hasil CSV", csv, "hasil_klasifikasi.csv", "text/csv")
