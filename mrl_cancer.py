import pandas as pd
import math
import random
import time as t

df = pd.read_csv("MCDO/breast-cancer.csv")
y = df["diagnosis"]
x = df.iloc[:, 2:]

y = y.map({'M': 1, 'B': 0})
x = (x - x.mean()) / x.std()

inter = 0.0
vars_x = x.shape[1]
loops = 500
a = 0.01
lambda_reg = 0.01

m = len(y)
indices = list(range(m))
random.shuffle(indices)
split = int(0.7 * m)
train_idx = indices[:split]
test_idx = indices[split:]

x_train = x.iloc[train_idx]
y_train = y.iloc[train_idx]
x_test = x.iloc[test_idx]
y_test = y.iloc[test_idx]

wi = [random.uniform(-1, 1) for _ in range(vars_x)]

def sigmoide(z):
    return 1 / (1 + math.exp(-z))

#* Calcular pesos de clases para balanceo
n_malignos = (y_train == 1).sum()  #*
n_benignos = (y_train == 0).sum()  #*
peso_maligno = len(y_train) / (2 * n_malignos)  #*
peso_benigno = len(y_train) / (2 * n_benignos)  #*

def entrenar(x, y, wi, inter, loops, a, m, lambda_reg):
    for l in range(loops):
        gpeso = [0.0] * vars_x
        ginter = 0.0
        for i, row in x.iterrows():
            z_val = inter + sum(row[col] * wi[j] for j, col in enumerate(x.columns))
            y_pre = sigmoide(z_val)
            e = y_pre - y[i]

            #* Aplicar peso según clase
            if y[i] == 1:  # maligno
                e *= peso_maligno  #*
            else:           # benigno
                e *= peso_benigno  #*

            ginter += e
            for j, col in enumerate(x.columns):
                gpeso[j] += e * row[col] + lambda_reg * wi[j]
        ginter /= m
        gpeso = [g / m for g in gpeso]
        inter -= a * ginter
        wi = [w - a * grad for w, grad in zip(wi, gpeso)]
    return wi, inter

def predecir(x, wi, inter):
    z_list = [inter + sum(row[col] * w for w, col in zip(wi, x.columns)) for _, row in x.iterrows()]
    probabilidades = [sigmoide(val) for val in z_list]
    predicciones = [1 if p >= 0.5 else 0 for p in probabilidades]
    return probabilidades, predicciones

def predecir_caso(fila, wi, inter):
    z = inter + sum(fila[col] * w for w, col in zip(wi, x.columns))
    prob = sigmoide(z)
    pred = "Maligno" if prob >= 0.5 else "Benigno"
    return prob, pred

timer1 = t.time()
wi, inter = entrenar(x_train, y_train, wi, inter, loops, a, len(y_train), lambda_reg)
timer2 = t.time()
timertot = timer2 - timer1
print(f"Tardo en entrenar {timertot:.5f} segundos")

prob_train, pred_train = predecir(x_train, wi, inter)
prob_test, pred_test = predecir(x_test, wi, inter)

exact_train = sum(pred == true for pred, true in zip(pred_train, y_train)) / len(y_train)
exact_test = sum(pred == true for pred, true in zip(pred_test, y_test)) / len(y_test)

print(f"Exactitud en train: {exact_train*100:.2f}%")
print(f"Exactitud en test: {exact_test*100:.2f}%")

primeras5 = [f"{p:.4f}" for p in prob_test[:5]]
ultimas5 = [f"{p:.4f}" for p in prob_test[-5:]]
print(f"Primeras 5 probabilidades test: {primeras5}")
print(f"Últimas 5 probabilidades test: {ultimas5}")

fila1 = x.iloc[20]
y_ej1 = y.iloc[20]
prob_caso1, pred_caso1 = predecir_caso(fila1, wi, inter)
print(f"Probabilidad de malignidad para la primera fila del CSV: {prob_caso1:.4f}")
print(f"Predicción: {pred_caso1}")
print("Real:", "Benigno" if y_ej1 == 0 else "Maligno")

fila2 = x.iloc[32]
y_ej2 = y.iloc[32]
prob_caso2, pred_caso2 = predecir_caso(fila2, wi, inter)
print(f"Probabilidad de malignidad para la fila 33 del CSV: {prob_caso2:.4f}")
print(f"Predicción: {pred_caso2}")
print("Real:", "Benigno" if y_ej2 == 0 else "Maligno")

# Calcular probabilidades para todo el dataset
probabilidades_todas, _ = predecir(x, wi, inter)

# Crear DataFrame con las probabilidades
df_prob = x.copy()
df_prob["probabilidad"] = probabilidades_todas

# Filtrar filas apenas por encima de 0.5
limite_inferior = 0.5001
limite_superior = 0.52
df_cercano_05 = df_prob[(df_prob["probabilidad"] >= limite_inferior) & 
                        (df_prob["probabilidad"] <= limite_superior)]

print("Filas con probabilidad apenas superior a 0.5:")
print(df_cercano_05)