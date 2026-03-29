import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# ── Configuração da Página ──────────────────────────────────────────────────
st.set_page_config(page_title="🛡️ Detecção de Fraudes", layout="wide")


# ── Carregamento de Dados ───────────────────────────────────────────────────
@st.cache_data(show_spinner="📥 Baixando dataset (144 MB) — aguarde...")
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    df["Amount_log"] = np.log1p(df["Amount"])
    X = df.drop(["Class", "Amount"], axis=1)
    y = df["Class"]
    splits = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    return splits, df


# ── Treinamento do Modelo ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Treinando modelo XGBoost — aguarde...")
def train_model(X_train, y_train):
    model = XGBClassifier(
        scale_pos_weight=10,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ── Interface Principal ─────────────────────────────────────────────────────
st.title("🛡️ Portal de Detecção de Fraudes Financeiras")

try:
    (X_train, X_test, y_train, y_test), raw_df = load_data()
    model = train_model(X_train, y_train)
except Exception as e:
    st.error(f"❌ Erro ao carregar dados ou treinar modelo: {e}")
    st.stop()

# ── Sidebar ─────────────────────────────────────────────────────────────────
menu = st.sidebar.radio(
    "Navegação",
    ["Visão Geral", "Ajuste de Limiar", "Explicabilidade (SHAP)"],
)

# ── Aba 1: Visão Geral ───────────────────────────────────────────────────────
if menu == "Visão Geral":
    st.header("📊 Análise do Desbalanceamento")
    st.write("Fraudes representam apenas **0.17%** das transações.")

    counts = raw_df["Class"].value_counts().rename({0: "Legítimas", 1: "Fraudes"})
    st.bar_chart(counts)

    col1, col2 = st.columns(2)
    col1.metric("Total de Transações", f"{len(raw_df):,}")
    col2.metric("Total de Fraudes", f"{raw_df['Class'].sum():,}")

# ── Aba 2: Ajuste de Limiar ──────────────────────────────────────────────────
elif menu == "Ajuste de Limiar":
    st.header("⚖️ Ajuste Dinâmico de Threshold")

    threshold = st.slider("Limiar de decisão", 0.01, 0.99, 0.30, step=0.01)

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > threshold).astype(int)

    st.subheader("Relatório de Classificação")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    st.subheader("Matriz de Confusão")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legítima", "Fraude"])
    ax.set_yticklabels(["Legítima", "Fraude"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

# ── Aba 3: Explicabilidade SHAP ──────────────────────────────────────────────
elif menu == "Explicabilidade (SHAP)":
    st.header("🔍 Por que o modelo tomou a decisão?")
    st.info("Calculando valores SHAP para 30 amostras de teste...")

    try:
        # TreeExplainer é muito mais eficiente que shap.Explainer para XGBoost
        explainer = shap.TreeExplainer(model)
        sample = X_test.iloc[:30]
        shap_values = explainer.shap_values(sample)

        st.subheader("Importância Global das Features (SHAP)")
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Distribuição dos Impactos (Beeswarm)")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Erro ao calcular SHAP: {e}")
