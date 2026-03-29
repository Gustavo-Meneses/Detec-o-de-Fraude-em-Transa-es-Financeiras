# 🛡️ Detecção de Fraude em Transações Financeiras

**Visualize o Desafio em Ação!**
Fizemos uma aplicação Streamlit interativa para que você possa explorar o modelo, ajustar os limiares e ver as explicações visuais SHAP diretamente no navegador.

🚀 **Dica para Avaliadores:** Navegue direto para o Portal Interativo.

---

## 🎨 Visualização com Streamlit (A Demo Visual)

Criamos uma demo funcional que unifica toda a análise e modelagem avançada. Nela, você pode:
1.  **Explorar o Desbalanceamento:** Ver a raridade das fraudes (0,17%).
2.  **Ajustar Dinamicamente o Limiar (Threshold):** Brincar com um controle deslizante para ver como aumentar o **Recall** (capturar mais fraudes) impacta a precisão (falsos positivos).
3.  **Ver Explicabilidade SHAP:** Visualizar gráficos que eliminam a natureza "caixa preta" do modelo, mostrando *por que* o algoritmo classificou uma transação específica como fraude.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas
* **Linguagem:** Python
* **A aplicação Interativa:** Streamlit
* **Modelagem & Explicação:** Scikit-Learn, XGBoost, SHAP

... (Mantenha o resto do README original aqui) ...

## 💻 Como Executar

### Opção A: Executar a Demo Interativa (Recomendado)

Esta aplicação Streamlit carrega os dados e o modelo avançado automaticamente.

1. Instale o Streamlit e as dependências:
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib streamlit
```

2. Execute a aplicação:
```bash
streamlit run app.py
```
Isso abrirá uma nova janela no seu navegador com o portal interativo.

### Opção B: Executar o Código Completo (Script Padrão)

Execute o script unificado (`main.py`) para reproduzir todo o pipeline de treino e avaliação avançada (com SMOTE e SHAP) em terminal:

```bash
python main.py
```
*(Certifique-se de ter as dependências instaladas)*
