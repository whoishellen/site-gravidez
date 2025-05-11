import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import joblib

# 1) Lê CSV com ; de separador
df = pd.read_csv("oficial.csv", sep=";")

# 2) Define X e y
X = df[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]
y = df['RiskLevel']

# 3) Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) Pipeline com StandardScaler + DecisionTree
pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('dt', DecisionTreeClassifier(random_state=42))
])

# 5) Treina e salva
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'modelo.pkl')
print("✅ Modelo treinado e salvo em modelo.pkl")
