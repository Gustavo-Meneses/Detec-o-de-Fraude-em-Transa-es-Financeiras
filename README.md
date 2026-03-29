# 🛡️ Detecção de Fraudes em Cartão de Crédito

Este projeto aplica Machine Learning para identificar transações fraudulentas, lidando com o desafio de dados altamente desbalanceados (apenas 0,17% de fraudes).

## 🚀 Como visualizar o projeto
Acesse o Dashboard interativo aqui: `[COLE_SEU_LINK_DO_STREAMLIT_AQUI]`

## 🛠️ Tecnologias
- **XGBoost**: Algoritmo de alta performance.
- **SMOTE**: Técnica de oversampling para balancear as classes.
- **SHAP**: Para explicabilidade do modelo (XAI).
- **Streamlit**: Para a interface visual do usuário.

## 📁 Estrutura do Repositório
- `app.py`: Interface interativa para avaliadores.
- `main.py`: Pipeline técnico de treinamento.
- `requirements.txt`: Dependências do ambiente.

## 💻 Instalação Local
1. Instale as dependências: `pip install -r requirements.txt`
2. Rode o app: `streamlit run app.py`
