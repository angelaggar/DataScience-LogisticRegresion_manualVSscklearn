import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time as t

df = pd.read_csv("MCDO/breast-cancer.csv")
y = df["diagnosis"].map({'M': 1, 'B': 0})
x = df.iloc[:, 2:]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.3, random_state=42, shuffle=True
)

model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=50)

# Tiempo de entrenamiento
timer1 = t.time()
model.fit(x_train, y_train)
timer2 = t.time()
timetraining = timer2 - timer1
print(f"Tardo en entrenar {timetraining:.5f} segundos")

# Tiempo de predicción
timer3 = t.time()
prob_train = model.predict_proba(x_train)[:, 1]
prob_test = model.predict_proba(x_test)[:, 1]
pred_train = model.predict(x_train)
pred_test = model.predict(x_test)
timer4 = t.time()
timeprediction = timer4 - timer3
print(f"Tardo en predecir {timeprediction:.5f} segundos")

exact_train = accuracy_score(y_train, pred_train)
exact_test = accuracy_score(y_test, pred_test)

print(f"Exactitud en train: {exact_train*100:.2f}%")
print(f"Exactitud en test: {exact_test*100:.2f}%")

primeras5 = [f"{p:.4f}" for p in prob_test[:5]]
ultimas5 = [f"{p:.4f}" for p in prob_test[-5:]]
print(f"Primeras 5 probabilidades test: {primeras5}")
print(f"Últimas 5 probabilidades test: {ultimas5}")

fila1 = x_scaled[20].reshape(1, -1)
y_ej1 = y.iloc[20]
prob_caso1 = model.predict_proba(fila1)[0, 1]
pred_caso1 = "Maligno" if prob_caso1 >= 0.5 else "Benigno"
print(f"Probabilidad de malignidad para la fila 21 del CSV: {prob_caso1:.4f}")
print(f"Predicción: {pred_caso1}")
print("Real:", "Benigno" if y_ej1 == 0 else "Maligno")

fila2 = x_scaled[32].reshape(1, -1)
y_ej2 = y.iloc[32]
prob_caso2 = model.predict_proba(fila2)[0, 1]
pred_caso2 = "Maligno" if prob_caso2 >= 0.5 else "Benigno"
print(f"Probabilidad de malignidad para la fila 33 del CSV: {prob_caso2:.4f}")
print(f"Predicción: {pred_caso2}")
print("Real:", "Benigno" if y_ej2 == 0 else "Maligno")
