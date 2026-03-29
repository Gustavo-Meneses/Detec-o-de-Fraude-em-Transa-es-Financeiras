import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

def load_and_preprocess_data():
    print("Carregando os dados...")
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    
    print("Aplicando engenharia de variáveis...")
    # Transformação Logarítmica e Escalonamento
    df["Amount_log"] = np.log1p(df["Amount"])
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    
    # Separando Features e Target
    X = df.drop(["Class", "Amount"], axis=1) # Removendo Amount original, mantendo as tratadas
    y = df["Class"]
    
    # Divisão de Treino e Teste com Estratificação (muito importante para dados desbalanceados)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def balance_data(X_train, y_train):
    print("Aplicando SMOTE para balanceamento dos dados de treino...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

def train_and_evaluate_xgboost(X_train, X_test, y_train, y_test):
    print("Treinando modelo XGBoost...")
    # scale_pos_weight ajuda o modelo a focar na classe minoritária
    xgb = XGBClassifier(
        scale_pos_weight=10, 
        eval_metric="logloss",
        random_state=42
    )
    xgb.fit(X_train, y_train)
    
    print("\nAvaliando o modelo (Limiar Padrão de 0.5):")
    y_pred = xgb.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print("\nAvaliando com Limiar Ajustado (Threshold = 0.3):")
    threshold = 0.3
    y_probs = xgb.predict_proba(X_test)[:, 1]
    y_pred_custom = (y_probs > threshold).astype(int)
    print(classification_report(y_test, y_pred_custom))
    
    # Exibindo a curva ROC e AUC
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)
    print(f"AUC Score: {auc_score:.4f}")
    
    return xgb

def explain_model_shap(model, X_test):
    print("\nGerando explicabilidade com SHAP...")
    # Usando uma amostra menor para o SHAP processar mais rápido no exemplo
    X_test_sample = X_test.sample(100, random_state=42)
    
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test_sample)
    
    # Plota a importância das variáveis
    shap.plots.bar(shap_values)

if __name__ == "__main__":
    # 1. Preparação dos Dados
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # 2. Balanceamento (Opcional se usar scale_pos_weight muito alto, mas bom para garantir)
    X_res, y_res = balance_data(X_train, y_train)
    
    # 3. Treinamento e Avaliação (Usando dados originais pois o XGB lida via scale_pos_weight)
    # Se preferir testar com SMOTE, troque X_train por X_res e y_train por y_res
    modelo_treinado = train_and_evaluate_xgboost(X_train, X_test, y_train, y_test)
    
    # 4. Explicabilidade
    explain_model_shap(modelo_treinado, X_test)
