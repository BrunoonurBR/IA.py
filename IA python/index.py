import pandas as pd

# substitua your_dataset.csve target_column pelo nome do arquivo do conjunto de dados e pelo nome da coluna de destino.
data = pd.read_csv("your_dataset.csv")
target_column = "target_column"
X = data.drop(target_column, axis=1)
y = data[target_column]

#Divida o conjunto de dados em conjuntos de treinamento e teste:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Crie o modelo de IA: Neste caso, usamos o algoritmo Random Forest da biblioteca scikit-learn.
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

#Treine o modelo:
model.fit(X_train, y_train)

#Faça previsões com os dados de teste:
y_pred = model.predict(X_test)

# Avalie o desempenho do modelo: calcule a pontuação de precisão para avaliar o desempenho do modelo.
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)