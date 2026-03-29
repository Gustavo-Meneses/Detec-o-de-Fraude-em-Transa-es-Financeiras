# 🛡️ Detecção de Fraude em Transações Financeiras

> Projeto de Machine Learning para identificação de transações fraudulentas em cartões de crédito, com foco em dados altamente desbalanceados, explicabilidade do modelo e interface interativa para avaliação de resultados.

🔗 **[Acesse o Dashboard ao vivo →](https://detec-o-de-fraude-em-transa-es-financeiras.streamlit.app/)**

---

## 📌 Descrição do Projeto

O setor financeiro enfrenta perdas bilionárias anuais com fraudes em cartões de crédito. O grande desafio técnico não está apenas em construir um modelo preditivo, mas em lidar com um dataset onde **menos de 0,2% das transações são fraudulentas** — um cenário clássico de desbalanceamento extremo de classes.

Este projeto aborda esse desafio de ponta a ponta: desde o pré-processamento e balanceamento dos dados, passando pelo treinamento de um modelo de alta performance, até a entrega de uma interface interativa que permite ao avaliador explorar o comportamento do modelo em tempo real.

---

## 🎯 Objetivos

- Detectar transações fraudulentas com alta precisão em um dataset severamente desbalanceado
- Aplicar técnicas de balanceamento (SMOTE) para melhorar a sensibilidade do modelo às fraudes
- Utilizar XGBoost como algoritmo principal, ajustado para o contexto de desbalanceamento
- Garantir **explicabilidade** das decisões do modelo via SHAP (XAI)
- Entregar uma interface interativa que permita ajuste dinâmico do limiar de decisão

---

## 📊 Dataset

| Atributo | Valor |
|---|---|
| Fonte | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total de transações | 284.807 |
| Transações fraudulentas | 492 (0,17%) |
| Features | 30 (V1–V28 via PCA + Time + Amount) |
| Período | 2 dias de transações europeias (setembro 2013) |

As features V1 a V28 são componentes principais obtidas via PCA para preservar a confidencialidade dos dados originais. As únicas features não transformadas são `Time` e `Amount`.

---

## 🏗️ Arquitetura da Solução

```
📦 Projeto
├── app.py              # Interface interativa (Streamlit)
├── main.py             # Pipeline técnico de treinamento
├── requirements.txt    # Dependências do ambiente
├── runtime.txt         # Versão Python para o Streamlit Cloud
└── .python-version     # Versão Python (uv/pyenv)
```

### Pipeline de ML (`main.py`)

```
Dados Brutos → Pré-processamento → SMOTE → XGBoost → Avaliação
```

1. **Carregamento** — download direto do dataset via URL pública
2. **Feature Engineering** — transformação logarítmica do `Amount` (`Amount_log`)
3. **Split estratificado** — 70% treino / 30% teste, mantendo proporção de fraudes
4. **Balanceamento** — SMOTE aplicado apenas no treino (sem vazamento de dados)
5. **Treinamento** — XGBClassifier com `scale_pos_weight=10`
6. **Avaliação** — ROC-AUC, Average Precision, Classification Report

---

## 🤖 Modelo

### XGBoost com ajuste de desbalanceamento

O XGBoost foi escolhido por sua robustez, velocidade e suporte nativo ao parâmetro `scale_pos_weight`, que penaliza mais os erros na classe minoritária (fraude).

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `scale_pos_weight` | 10 | Penaliza erros em fraudes |
| `eval_metric` | logloss | Adequado para classificação binária |
| `random_state` | 42 | Reprodutibilidade |

### Resultados (threshold = 0.30)

| Métrica | Classe 0 (Legítima) | Classe 1 (Fraude) |
|---|---|---|
| Precision | 1.00 | 0.91 |
| Recall | 1.00 | 0.78 |
| F1-Score | 1.00 | 0.84 |

- **Macro avg F1:** 0.92
- O modelo detecta **78% das fraudes reais** com precisão de 91% — ou seja, de cada 100 alertas de fraude, 91 são genuínos.

---

## ⚖️ Balanceamento com SMOTE

O **SMOTE (Synthetic Minority Over-sampling Technique)** gera amostras sintéticas da classe minoritária no espaço de features, em vez de simplesmente duplicar exemplos existentes. Isso força o modelo a aprender uma fronteira de decisão mais robusta para identificar fraudes.

> Importante: o SMOTE é aplicado **apenas no conjunto de treino**, evitando vazamento de informação para o conjunto de teste.

---

## 🔍 Explicabilidade com SHAP

O projeto utiliza **SHAP (SHapley Additive exPlanations)** para interpretar as decisões individuais do modelo, respondendo à pergunta: *"por que o modelo classificou essa transação como fraude?"*

- **TreeExplainer** — implementação otimizada para modelos baseados em árvores (XGBoost)
- **Summary Plot (bar)** — importância global de cada feature
- **Beeswarm Plot** — distribuição dos impactos por feature e por amostra

As features com maior influência nas predições são **V14, V4, V12 e V19** — componentes PCA que capturam padrões comportamentais distintos entre transações legítimas e fraudulentas.

---

## 🖥️ Dashboard Interativo

O dashboard foi construído com **Streamlit** e oferece três seções:

### 1. Visão Geral
- Visualização do desbalanceamento das classes
- Métricas gerais do dataset (total de transações e fraudes)

### 2. Ajuste de Limiar
- Slider interativo para ajuste do threshold de decisão (0.01 a 0.99)
- Relatório de classificação atualizado em tempo real
- Matriz de Confusão dinâmica

> Esta seção é especialmente útil para demonstrar o **trade-off entre Precision e Recall**: reduzir o limiar aumenta a detecção de fraudes (recall), mas também gera mais falsos positivos.

### 3. Explicabilidade (SHAP)
- Gráfico de importância global das features
- Beeswarm plot com distribuição de impactos por amostra
- Interpretação de quais features mais influenciam cada predição

---

## 🚀 Como Executar Localmente

```bash
# 1. Clone o repositório
git clone https://github.com/Gustavo-Meneses/Detec-o-de-Fraude-em-Transa-es-Financeiras.git
cd Detec-o-de-Fraude-em-Transa-es-Financeiras

# 2. Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute o pipeline de treino (opcional)
python main.py

# 5. Suba o dashboard
streamlit run app.py
```

---

## 🛠️ Tecnologias

| Tecnologia | Versão | Uso |
|---|---|---|
| Python | 3.11 | Linguagem base |
| Streamlit | 1.40.1 | Interface interativa |
| XGBoost | 2.1.2 | Modelo de classificação |
| SHAP | 0.46.0 | Explicabilidade (XAI) |
| scikit-learn | 1.5.2 | Pipeline de ML e métricas |
| imbalanced-learn | 0.12.4 | SMOTE |
| pandas | 2.2.3 | Manipulação de dados |
| numpy | 1.26.4 | Computação numérica |
| matplotlib | 3.9.2 | Visualizações |

---

## 📁 Estrutura dos Arquivos

### `app.py` — Interface Streamlit
Carrega os dados, treina o modelo com cache (`@st.cache_resource`) e expõe três abas de navegação. O SHAP é carregado de forma lazy (apenas quando a aba é acessada) para evitar sobrecarga na inicialização.

### `main.py` — Pipeline Técnico
Pipeline completo e modular: carregamento → pré-processamento → SMOTE → treinamento → avaliação com ROC-AUC e Average Precision. Projetado para execução standalone via terminal.

### `requirements.txt`
Dependências com versões fixadas (pinned) para garantir reprodutibilidade total do ambiente.

### `.python-version` / `runtime.txt`
Fixam Python 3.11 no Streamlit Cloud, evitando incompatibilidades com Python 3.14+ onde alguns pacotes ainda não possuem wheels pré-compilados.

---

## 👤 Autor

**Gustavo Meneses**
[GitHub](https://github.com/Gustavo-Meneses)
