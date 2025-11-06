from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import time as t

df = pd.read_csv("MCDO/breast-cancer.csv")
y = df["diagnosis"].map({'M': 1, 'B': 0})
x = df.iloc[:, 2:]

# Normalizar
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Separar en train/test
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42, shuffle=True)

# Entrenar regresión logística con L2 y balanceo de clases
timer1 = t.time()
model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=500, class_weight='balanced')
model.fit(x_train, y_train)
timer2 = t.time()
print(f"Tardo en entrenar {timer2 - timer1:.5f} segundos")

# Predicciones
pred_train = model.predict(x_train)
pred_test = model.predict(x_test)
prob_test = model.predict_proba(x_test)[:, 1]  # probabilidad de clase 1 (maligno)

# Exactitud
exact_train = accuracy_score(y_train, pred_train)
exact_test = accuracy_score(y_test, pred_test)
print(f"Exactitud en train: {exact_train*100:.2f}%")
print(f"Exactitud en test: {exact_test*100:.2f}%")

# Primeras y últimas 5 probabilidades
primeras5 = [f"{p:.4f}" for p in prob_test[:5]]
ultimas5 = [f"{p:.4f}" for p in prob_test[-5:]]
print(f"Primeras 5 probabilidades test: {primeras5}")
print(f"Últimas 5 probabilidades test: {ultimas5}")

# Predicción de casos individuales
def predecir_caso_skl(fila, modelo):
    prob = modelo.predict_proba(fila.reshape(1, -1))[0, 1]
    pred = "Maligno" if prob >= 0.5 else "Benigno"
    return prob, pred

fila21 = x.iloc[20].to_numpy()
y_21 = y.iloc[20]
prob_21, pred_21 = predecir_caso_skl(fila21, model)
print(f"Probabilidad de malignidad fila 21: {prob_21:.4f}")
print(f"Predicción: {pred_21}")
print("Real:", "Maligno" if y_21 == 1 else "Benigno")

fila33 = x.iloc[32].to_numpy()
y_33 = y.iloc[32]
prob_33, pred_33 = predecir_caso_skl(fila33, model)
print(f"Probabilidad de malignidad fila 33: {prob_33:.4f}")
print(f"Predicción: {pred_33}")
print("Real:", "Maligno" if y_33 == 1 else "Benigno")
