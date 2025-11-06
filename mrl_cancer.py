import pandas as pd
import math
import random
import time as t

df = pd.read_csv("MCDO/breast-cancer.csv")
y = df["diagnosis"]
x = df.iloc[:, 2:]

inter = 0.0
vars_x = x.shape[1]
loops = 500
a = 0.01
m = len(y)

y = y.map({'M': 1, 'B': 0})

x = (x - x.mean()) / x.std()

wi = [random.uniform(-1, 1) for _ in range(vars_x)]

def sigmoide(z):
    return 1 / (1 + math.exp(-z))


for l in range(loops):
    timer1= t.time()
    gpeso = [0.0] * vars_x
    ginter = 0.0

    for i, row in x.iterrows():
        z_val = inter + sum(row[col] * wi[j] for j, col in enumerate(x.columns))
        y_pre = sigmoide(z_val)
        e = y_pre - y[i]

        ginter += e
        for j, col in enumerate(x.columns):
            gpeso[j] += e * row[col]

    ginter /= m
    gpeso = [g / m for g in gpeso]

    inter -= a * ginter
    wi = [w - a * grad for w, grad in zip(wi, gpeso)]

    if (l + 1) % 50 == 0:
        z_list = [inter + sum(row[col] * w for w, col in zip(wi, x.columns)) for _, row in x.iterrows()]
        loss = -sum(y[i]*math.log(sigmoide(z_list[i])) + (1-y[i])*math.log(1-sigmoide(z_list[i])) for i in range(m)) / m
        #print(f"Loop {l+1}, Loss: {loss:.4f}")
    timer2=t.time()

z_final = [inter + sum(row[col] * w for w, col in zip(wi, x.columns)) for _, row in x.iterrows()]
probabilidades = [sigmoide(val) for val in z_final]

print("Primeras 10 probabilidades:", probabilidades[:10])
predicciones = [1 if p >= 0.5 else 0 for p in probabilidades]
exactitud = sum(pred == true for pred, true in zip(predicciones, y)) / len(y)
print(f"Exactitud del modelo: {exactitud*100:.2f}%")

fila = x.iloc[30]
y_ej= y[30]
z_caso = inter + sum(fila[col] * w for w, col in zip(wi, x.columns))
prob_caso = sigmoide(z_caso)

print(f"Probabilidad de malignidad para la primera muestra: {prob_caso:.4f}")

if prob_caso >= 0.5:
    print("Predicción: Maligno")
else:
    print("Predicción: Benigno")

print(y_ej)