import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# 1. Carga e Preprocessamento
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
X = df.drop(['Class', 'Amount'], axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# 2. Balanceamento com SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# 3. Treino com XGBoost
xgb = XGBClassifier(scale_pos_weight=1, eval_metric="logloss")
xgb.fit(X_res, y_res)

# 4. Avaliação
y_pred = xgb.predict(X_test)
print("Relatório Técnico do Modelo Avançado:")
print(classification_report(y_test, y_pred))
