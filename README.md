# 🛡️ Detecção de Fraude em Transações Financeiras

Este projeto demonstra a construção de um modelo de Machine Learning robusto para identificar anomalias (fraudes) em transações de cartão de crédito. O principal desafio técnico abordado aqui é o **extremo desbalanceamento dos dados**, onde as transações fraudulentas representam uma fração minúscula (aprox. 0,17%) do volume total.

## 🚀 Tecnologias e Bibliotecas Utilizadas
* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
* **Explicabilidade:** SHAP (SHapley Additive exPlanations)
* **Visualização:** Matplotlib

## 🧠 Arquitetura e Soluções Técnicas

Para evitar que o modelo atinja uma "falsa acurácia" de 99% apenas classificando tudo como normal, implementamos as seguintes estratégias:

1. **Engenharia de Variáveis (Feature Engineering):** - Aplicação de transformação logarítmica (`np.log1p`) na variável de valor da transação (`Amount`) para reduzir o impacto de outliers.
   - Padronização via `StandardScaler`.
2. **Reamostragem de Dados (Data Resampling):**
   - Utilização do algoritmo **SMOTE** (Synthetic Minority Over-sampling Technique) para gerar instâncias sintéticas da classe minoritária, ajudando o modelo a reconhecer os padrões da fraude.
3. **Modelagem Avançada (XGBoost):**
   - Implementação do algoritmo `XGBClassifier`, ajustando o parâmetro `scale_pos_weight` para penalizar severamente os erros na classe minoritária.
4. **Ajuste de Limiar (Threshold Tuning):**
   - Redução do limiar de decisão de classificação (ex: de 0.5 para 0.3) para priorizar o **Recall**, garantindo que o mínimo possível de fraudes passe despercebido, mesmo que isso custe um leve aumento nos falsos positivos.
5. **Explicabilidade do Modelo:**
   - Integração com a biblioteca **SHAP** para eliminar a natureza "caixa preta" do modelo. O gráfico de barras gerado mostra exatamente quais variáveis mais impactaram a decisão do algoritmo para classificar uma transação como fraude.

## 📊 Métricas de Avaliação
Neste cenário, a acurácia global não é a melhor métrica. Focamos em:
- **Recall (Revocação):** Capacidade de encontrar todas as fraudes reais.
- **Precision (Precisão):** Garantir que, quando o modelo alerta sobre uma fraude, a chance de ser correta seja alta.
- **AUC-ROC:** Avaliação geral da capacidade do modelo de distinguir entre as classes.

## 💻 Como Executar

1. Clone este repositório:
   ```bash
   git clone [https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git](https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git)

2. Instale as dependências:
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib

3. Execute o script principal (ou abra o notebook):
```bash
python main.py


Desenvolvido com foco em automação, inteligência e escalabilidade.
