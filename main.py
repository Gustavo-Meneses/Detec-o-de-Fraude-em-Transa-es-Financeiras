"""
main.py — Pipeline técnico de treinamento
Detecção de Fraudes em Transações Financeiras

Executa: python main.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")


# ── 1. Carregamento dos Dados ───────────────────────────────────────────────
def load_data():
    print("📥 Carregando dataset...")
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    print(f"   Shape: {df.shape} | Fraudes: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    return df


# ── 2. Pré-processamento ────────────────────────────────────────────────────
def preprocess(df):
    print("🔧 Pré-processando...")
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount"])
    X = df.drop(["Class", "Amount"], axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train["Amount_log"] = scaler.fit_transform(X_train[["Amount_log"]])
    X_test["Amount_log"] = scaler.transform(X_test[["Amount_log"]])

    return X_train, X_test, y_train, y_test


# ── 3. Balanceamento com SMOTE ──────────────────────────────────────────────
def apply_smote(X_train, y_train):
    print("⚖️  Aplicando SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"   Shape após SMOTE: {X_res.shape} | Fraudes: {y_res.sum()}")
    return X_res, y_res


# ── 4. Treinamento ──────────────────────────────────────────────────────────
def train(X_train, y_train):
    print("🤖 Treinando XGBoost...")
    model = XGBClassifier(
        scale_pos_weight=10,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ── 5. Avaliação ────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, threshold=0.30):
    print(f"\n📊 Avaliação (threshold={threshold}):")
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > threshold).astype(int)

    print(classification_report(y_test, y_pred, target_names=["Legítima", "Fraude"]))
    print(f"   ROC-AUC:          {roc_auc_score(y_test, y_probs):.4f}")
    print(f"   Avg Precision:    {average_precision_score(y_test, y_probs):.4f}")


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    model = train(X_train_res, y_train_res)
    evaluate(model, X_test, y_test)
    print("\n✅ Pipeline concluído com sucesso!")
