import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, confusion_matrix
from xgboost import XGBClassifier

# Configuração da Página
st.set_page_config(page_title="🛡️ Detecção de Fraudes", layout="wide")

@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    df["Amount_log"] = np.log1p(df["Amount"])
    X = df.drop(["Class", "Amount"], axis=1)
    y = df["Class"]
    return train_test_split(X, y, stratify=y, test_size=0.3, random_state=42), df

@st.cache_resource
def train_model(X_train, y_train):
    model = XGBClassifier(scale_pos_weight=10, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    return model

# Interface
st.title("🛡️ Portal de Detecção de Fraudes Financeiras")
(X_train, X_test, y_train, y_test), raw_df = load_data()
model = train_model(X_train, y_train)

# Sidebar
menu = st.sidebar.radio("Navegação", ["Visão Geral", "Ajuste de Limiar", "Explicabilidade (SHAP)"])

if menu == "Visão Geral":
    st.header("Análise do Desbalanceamento")
    st.write("Fraudes representam apenas 0.17% das transações.")
    st.bar_chart(raw_df["Class"].value_counts())

elif menu == "Ajuste de Limiar":
    st.header("Ajuste Dinâmico de Threshold")
    threshold = st.slider("Limiar", 0.01, 0.99, 0.30)
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > threshold).astype(int)
    
    st.text("Relatório de Classificação:")
    st.text(classification_report(y_test, y_pred))

elif menu == "Explicabilidade (SHAP)":
    st.header("Por que o modelo tomou a decisão?")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test[:100])
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig)
