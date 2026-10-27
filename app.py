# ============================================================
# STREAMLIT: PREDIKSI KEBOTAKAN
# Hanya MEMAKAI model yang sudah dilatih di Colab (tanpa latih ulang)
# Jalankan:  streamlit run app.py
# ============================================================
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

st.set_page_config(page_title="Prediksi Botak", page_icon="🧑‍🦲",
                   layout="wide")

# ------------------------------------------------------------
# 0. LOKASI FILE
# ------------------------------------------------------------
DIR = Path(__file__).parent / "hasil_prediksi_botak"
DIR_MODEL = DIR / "semua_model"
LABELS = {0: "Tidak Botak", 1: "Botak"}

PENJELASAN = {
    "LGBMClassifier": "Boosting berbasis pohon: pohon dibuat berurutan, tiap pohon baru memperbaiki kesalahan pohon sebelumnya. Cepat dan biasanya akurat.",
    "XGBClassifier": "Boosting pohon dengan regularisasi ketat. Sangat populer untuk data tabular.",
    "RandomForestClassifier": "Ratusan pohon dilatih pada sampel acak, lalu hasilnya divoting. Stabil dan tahan overfitting.",
    "LogisticRegression": "Model linear yang menghitung peluang kelas dari kombinasi bobot fitur. Mudah ditafsirkan.",
    "CalibratedClassifierCV": "Pembungkus yang mengkalibrasi probabilitas model dasar (regresi logistik) agar angka peluangnya lebih jujur.",
    "LinearSVC": "Mencari bidang pemisah terbaik dengan margin terbesar. Tidak punya probabilitas asli.",
    "BaggingClassifier": "Banyak decision tree dilatih pada sampel bootstrap lalu divoting. Mengurangi varians.",
    "LinearDiscriminantAnalysis": "Mencari kombinasi linear fitur yang paling memisahkan dua kelas (asumsi data kira-kira normal).",
    "AdaBoostClassifier": "Boosting yang memberi bobot lebih pada data yang sering salah ditebak di putaran sebelumnya.",
    "RidgeClassifierCV": "Klasifikasi linear dengan regularisasi L2; alpha dipilih otomatis lewat cross-validation.",
    "RidgeClassifier": "Klasifikasi linear dengan regularisasi L2. Cepat, tanpa probabilitas asli.",
    "SGDClassifier": "Model linear yang dilatih dengan stochastic gradient descent. Ringan untuk data besar.",
    "SVC": "Support Vector Machine kernel RBF; bisa menangkap pola tidak linear. Tanpa probabilitas asli.",
}


# ------------------------------------------------------------
# 1. LOAD SEMUA HASIL DARI COLAB (di-cache, hanya dibaca sekali)
# ------------------------------------------------------------
@st.cache_data
def load_json(name):
    with open(DIR / name, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(name):
    return pd.read_csv(DIR / name)


@st.cache_resource(show_spinner="Memuat model...")
def load_models():
    out, gagal = {}, {}
    for p in sorted(DIR_MODEL.glob("*.pkl")):
        try:
            with open(p, "rb") as f:
                out[p.stem] = pickle.load(f)
        except Exception as e:
            # Cadangan khusus XGBoost: bangun ulang dari file JSON
            json_path = DIR_MODEL / f"{p.stem}.json"
            if p.stem == "XGBClassifier" and json_path.exists():
                try:
                    with open(DIR / "preprocessor.pkl", "rb") as f:
                        pre = pickle.load(f)
                    xgb = XGBClassifier()
                    xgb.load_model(str(json_path))
                    out[p.stem] = Pipeline([("preprocessor", pre),
                                            ("model", xgb)])
                    continue
                except Exception as e2:
                    e = e2
            gagal[p.stem] = str(e)[:200]
    return out, gagal


if not DIR_MODEL.exists():
    st.error(f"Folder tidak ditemukan: {DIR_MODEL}\n\n"
             "Unzip hasil_prediksi_botak_1.zip ke folder "
             "'hasil_prediksi_botak' di samping app.py.")
    st.stop()

info = load_json("model_info.json")
feat = load_json("feature_info.json")
models, model_gagal = load_models()
if model_gagal:
    st.warning("Model gagal dimuat: " + ", ".join(model_gagal.keys()))
test_set = load_csv("test_set.csv")
perm_all = load_csv("permutation_semua_model.csv")
cv_all = load_csv("cv_hasil.csv")
val_all = load_csv("perbandingan_model.csv")

FEATURE_COLS = feat["feature_columns"]
NUM_COLS = feat["numeric_cols"]
CAT_COLS = feat["categorical_cols"]
STATS = feat["numeric_stats"]
OPTS = feat["categorical_options"]

X_test = test_set[FEATURE_COLS]
y_test = test_set["label_asli"].astype(int)


# ------------------------------------------------------------
# 2. FUNGSI BANTU
# ------------------------------------------------------------
def predict_score(pipe, X):
    """Skor 0-1 kelas Botak. Model tanpa predict_proba memakai
    sigmoid(decision_function): ini BUKAN probabilitas asli."""
    model = pipe.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return pipe.predict_proba(X)[:, 1], True
    d = pipe.decision_function(X)
    return 1 / (1 + np.exp(-d)), False


@st.cache_data
def test_metrics(name):
    """Hanya MEMPREDIKSI test set (model tidak dilatih ulang)."""
    pipe = models[name]
    pred = pipe.predict(X_test)
    score, _ = predict_score(pipe, X_test)
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_test, pred),
        "Precision_Botak": precision_score(y_test, pred, zero_division=0),
        "Recall_Botak": recall_score(y_test, pred, zero_division=0),
        "F1_Botak": f1_score(y_test, pred, zero_division=0),
        "F1_Macro": f1_score(y_test, pred, average="macro",
                             zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, score),
    }


# ------------------------------------------------------------
# 3. SIDEBAR: PILIH ALGORITMA
# ------------------------------------------------------------
st.sidebar.title("⚙️ Pengaturan")
names = list(models.keys())
default_idx = names.index(info["best_model_name"]) \
    if info["best_model_name"] in names else 0
algo = st.sidebar.selectbox("Algoritma", names, index=default_idx)
pipe = models[algo]
st.sidebar.caption(PENJELASAN.get(algo, ""))
if algo == info["best_model_name"]:
    st.sidebar.success("⭐ Model terbaik (Validation F1 Botak)")

thr = st.sidebar.slider(
    "Ambang keputusan", 0.05, 0.95, 0.50, 0.05,
    help="Skor Botak >= ambang ini dianggap Botak.")

# ------------------------------------------------------------
# 4. SIDEBAR: SLIDER INPUT
#    Urut dari variabel paling berpengaruh pada algoritma terpilih
#    (diambil dari permutation_semua_model.csv)
# ------------------------------------------------------------
perm = (perm_all[perm_all["Model"] == algo]
        .drop(columns="Model")
        .sort_values("penurunan_F1", ascending=False)
        .reset_index(drop=True))
urutan = perm["variabel"].tolist()
top3 = set(urutan[:3])

st.sidebar.markdown("---")
st.sidebar.subheader("🧍 Data Orang")
st.sidebar.caption("Diurutkan dari variabel paling berpengaruh pada "
                   "algoritma ini. ⭐ = 3 teratas.")

user_input = {}
for col in urutan:
    nama = ("⭐ " if col in top3 else "") + col.replace("_", " ").title()

    if col in CAT_COLS:                                  # kategori
        user_input[col] = st.sidebar.selectbox(nama, OPTS[col],
                                               key=f"in-{col}")
        continue

    s = STATS[col]
    lo, hi, med = s["min"], s["max"], s["median"]

    if col.startswith("is_"):                            # biner
        user_input[col] = st.sidebar.radio(
            nama, [0, 1], index=int(round(med)),
            format_func=lambda v: "Ya" if v == 1 else "Tidak",
            horizontal=True, key=f"in-{col}")
    elif col in ("umur", "stress"):                      # bilangan bulat
        user_input[col] = st.sidebar.slider(
            nama, int(lo), int(hi), int(med), 1, key=f"in-{col}")
    elif col == "gaji":
        user_input[col] = float(st.sidebar.slider(
            nama + " (Rp)", int(lo), int(hi), int(med), 100_000,
            key=f"in-{col}"))
    else:                                                # berat, tinggi
        user_input[col] = st.sidebar.slider(
            nama, float(round(lo)), float(round(hi)), float(round(med)),
            0.5, key=f"in-{col}")

df_input = pd.DataFrame([user_input])[FEATURE_COLS]

# ------------------------------------------------------------
# 5. TAMPILAN UTAMA
# ------------------------------------------------------------
st.title("🧑‍🦲 Prediksi Kebotakan")
st.caption("Semua model sudah dilatih di Google Colab. Aplikasi ini hanya "
           "memuat dan memakainya.")

tab_pred, tab_rep, tab_cmp, tab_var, tab_info = st.tabs([
    "🎯 Prediksi", "📊 Report Algoritma", "🏁 Bandingkan Semua",
    "🔍 Variabel Berpengaruh", "ℹ️ Info"])

# ===================== TAB 1: PREDIKSI =====================
with tab_pred:
    score, is_proba = predict_score(pipe, df_input)
    score = float(score[0])
    pred = int(score >= thr) if is_proba else int(pipe.predict(df_input)[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Algoritma", algo)
    c2.metric("Skor Botak" if is_proba else "Skor keyakinan",
              f"{score*100:.1f}%")
    c3.metric("Hasil", LABELS[pred])

    if pred == 1:
        st.error(f"### BOTAK ({score*100:.1f}%)")
    else:
        st.success(f"### TIDAK BOTAK ({score*100:.1f}%)")
    st.progress(min(max(score, 0.0), 1.0))

    if not is_proba:
        st.info(f"{algo} tidak punya probabilitas asli. Skor ini hanya "
                "gambaran keyakinan (sigmoid dari decision function), dan "
                "slider ambang tidak berlaku untuk model ini.")

    st.markdown("#### Hasil semua algoritma pada input yang sama")
    rows = []
    for n, p in models.items():
        s, pr = predict_score(p, df_input)
        lbl = int(s[0] >= thr) if pr else int(p.predict(df_input)[0])
        rows.append({"Algoritma": n, "Skor Botak": round(float(s[0]), 4),
                     "Hasil": LABELS[lbl]})
    cmp_df = pd.DataFrame(rows)
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)
    st.caption(f"{(cmp_df['Hasil'] == 'Botak').sum()} dari "
               f"{len(cmp_df)} algoritma memprediksi Botak.")

    st.markdown("#### Simulasi: ubah satu variabel, lainnya tetap")
    var = st.selectbox("Variabel yang divariasikan", FEATURE_COLS)
    if var in CAT_COLS:
        xs = np.array(OPTS[var])
    elif var.startswith("is_"):
        xs = np.array([0, 1])
    else:
        xs = np.linspace(STATS[var]["min"], STATS[var]["max"], 40)

    sim = pd.concat([df_input] * len(xs), ignore_index=True)
    sim[var] = xs
    sim_score, _ = predict_score(pipe, sim)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    if var in CAT_COLS or len(xs) == 2:
        ax.bar([str(x) for x in xs], sim_score, color="#76B7B2")
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    else:
        ax.plot(xs, sim_score, marker="o", color="#76B7B2")
    ax.axhline(thr, color="red", linestyle="--", label=f"ambang {thr}")
    ax.set_ylim(0, 1)
    ax.set_xlabel(var)
    ax.set_ylabel("Skor Botak")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# ===================== TAB 2: REPORT ALGORITMA =====================
with tab_rep:
    st.subheader(f"Report: {algo}")
    st.write(PENJELASAN.get(algo, ""))

    with st.expander("Hyperparameter model ini (saat dilatih di Colab)"):
        st.json({k: str(v) for k, v in
                 pipe.named_steps["model"].get_params().items()})

    # --- Validation (dari Colab) ---
    st.markdown("**Validation set** (dihitung di Colab)")
    v = val_all[val_all["Model"] == algo].drop(columns="Model")
    st.dataframe(v.round(4), hide_index=True, use_container_width=True)

    # --- Test (prediksi ulang dari model tersimpan) ---
    st.markdown("**Test set** (model tersimpan memprediksi test_set.csv)")
    m = test_metrics(algo)
    cols = st.columns(len(m))
    for c, (k, val) in zip(cols, m.items()):
        c.metric(k, f"{val:.4f}")

    y_pred = pipe.predict(X_test)
    y_score, _ = predict_score(pipe, X_test)

    l, r = st.columns(2)
    with l:
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4.3))
        ConfusionMatrixDisplay(cm, display_labels=list(LABELS.values())
                               ).plot(ax=ax, cmap="Blues",
                                      values_format="d", colorbar=False)
        ax.set_title("Confusion Matrix (Test)")
        st.pyplot(fig)
        plt.close(fig)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"Benar Botak: {tp} | Benar Tidak Botak: {tn} | "
                   f"Botak tapi ditebak Tidak: {fn} | "
                   f"Tidak Botak tapi ditebak Botak: {fp}")
    with r:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        fig, ax = plt.subplots(figsize=(5, 4.3))
        ax.plot(fpr, tpr, label=f"AUC = {m['ROC_AUC']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Kurva ROC (Test)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    st.text("Classification Report (Test)")
    st.code(classification_report(
        y_test, y_pred, labels=[0, 1], target_names=list(LABELS.values()),
        digits=4, zero_division=0))

    # --- Cross-validation (dari Colab) ---
    st.markdown("**Cross-Validation 5-fold** (dihitung di Colab)")
    cv_row = cv_all[cv_all["Model"] == algo]
    if not cv_row.empty:
        r_ = cv_row.iloc[0]
        a, b, c = st.columns(3)
        a.metric("Accuracy", f"{r_['Accuracy_mean']:.4f}",
                 f"± {r_['Accuracy_std']:.4f}", delta_color="off")
        b.metric("F1 Botak", f"{r_['F1_mean']:.4f}",
                 f"± {r_['F1_std']:.4f}", delta_color="off")
        c.metric("ROC AUC", f"{r_['ROC_AUC_mean']:.4f}",
                 f"± {r_['ROC_AUC_std']:.4f}", delta_color="off")
        st.caption("Angka kecil di bawah (±) adalah standar deviasi antar "
                   "fold. Makin kecil, makin stabil modelnya.")

# ===================== TAB 3: BANDINGKAN SEMUA =====================
with tab_cmp:
    st.subheader("Perbandingan 13 algoritma")

    st.markdown("**Validation set** (dasar pemilihan model terbaik)")
    st.dataframe(val_all.round(4), use_container_width=True,
                 hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    g = val_all.sort_values("F1_Botak")
    ax.barh(g["Model"], g["F1_Botak"], color="#4C78A8")
    ax.set_xlim(g["F1_Botak"].min() - 0.01, 1)
    ax.set_xlabel("F1 Botak (validation)")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Cross-Validation 5-fold**")
    st.dataframe(cv_all.round(4), use_container_width=True,
                 hide_index=True)
    best_cv = cv_all.sort_values("F1_mean", ascending=False).iloc[0]
    st.success(f"Terbaik menurut CV (F1): **{best_cv['Model']}** "
               f"({best_cv['F1_mean']:.4f} ± {best_cv['F1_std']:.4f}).  "
               f"Terbaik menurut validation: **{info['best_model_name']}**.")

    st.markdown("**Test set** (semua model memprediksi test_set.csv)")
    test_df = pd.DataFrame({n: test_metrics(n) for n in models}).T
    st.dataframe(test_df.sort_values("F1_Botak", ascending=False).round(4),
                 use_container_width=True)

# ===================== TAB 4: VARIABEL BERPENGARUH =====================
with tab_var:
    st.subheader(f"Variabel yang paling mempengaruhi hasil: {algo}")
    st.write("Cara kerja *permutation importance*: satu kolom diacak di "
             "test set, lalu dilihat seberapa turun F1-nya. Makin besar "
             "penurunannya, makin bergantung model pada variabel itu. "
             "Hasil ini sudah dihitung di Colab.")
    l, r = st.columns(2)
    with l:
        st.dataframe(perm.round(4), hide_index=True,
                     use_container_width=True)
    with r:
        fig, ax = plt.subplots(figsize=(6, 5))
        g = perm.sort_values("penurunan_F1")
        ax.barh(g["variabel"], g["penurunan_F1"], xerr=g["std"],
                color="#E15759")
        ax.set_xlabel("Penurunan F1 saat kolom diacak")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Rata-rata semua algoritma**")
    avg = (perm_all.groupby("variabel")["penurunan_F1"].mean()
           .sort_values(ascending=False).reset_index())
    st.dataframe(avg.round(4), hide_index=True, use_container_width=True)

    st.warning("Ini menunjukkan variabel yang **dipakai model**, bukan "
               "sebab-akibat kebotakan di dunia nyata. Pola bisa saja "
               "kebetulan di data.")

# ===================== TAB 5: INFO =====================
with tab_info:
    st.subheader("Info model tersimpan")
    st.json(info)
    st.subheader("Contoh prediksi test set (model terbaik)")
    tp = load_csv("test_predictions.csv")
    tp["benar"] = tp["label_asli"] == tp["prediksi"]
    st.dataframe(tp.head(100), use_container_width=True)
    st.caption(f"Akurasi pada file ini: {tp['benar'].mean():.4f}")