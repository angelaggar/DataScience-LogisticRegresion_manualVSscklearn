import pandas as pd
import numpy as np
import time as t
import random
from printer import printCasos

df = pd.read_csv("MCDO/breast-cancer.csv")
y = df["diagnosis"]
x = df.iloc[:, 2:]

y = y.map({'M': 1, 'B': 0})
x = (x - x.mean()) / x.std()
yneg = sum(1 for val in y if val != 1)
#print(yneg)

inter = 0.0
vars_x = x.shape[1]
loops = 1000
a = 0.01
l_reg = 0.01
lossi= 1e-4

m = len(y)
idx = list(range(m))
random.seed(56)
random.shuffle(idx)

split = int(0.7 * m)
x_train, y_train = x.iloc[idx[:split]], y.iloc[idx[:split]]
x_test,  y_test  = x.iloc[idx[split:]], y.iloc[idx[split:]]
wi = [random.uniform(-1, 1) for _ in range(vars_x)]

def sigmoide(z): return 1 / (1 + np.exp(-z))

numM = (y_train == 1).sum()
numB = (y_train == 0).sum()
wm = len(y_train) / (2 * numM)
wb = len(y_train) / (2 * numB)

def entrenar(x, y, wi, inter, loops, a, m, l_reg, lossi):
    prevloss = float('inf')
    for loop in range(loops + 1):
        gpeso = [0.0] * vars_x
        ginter = 0.0
        for i, row in x.iterrows():
            z_val = inter + sum(row[col] * wi[j] for j, col in enumerate(x.columns))
            ypre = sigmoide(z_val)
            err = ypre - y[i]
            if y[i] == 1: err *= wm
            else: err *= wb
            ginter += err
            for j, col in enumerate(x.columns):
                gpeso[j] += err * row[col] + l_reg * wi[j]

        ginter /= m
        gpeso = [g / m for g in gpeso]
        inter -= a * ginter
        wi = [w - a * grad for w, grad in zip(wi, gpeso)]
        

        if loop % 50 == 0 or loop == loops:
            z_list = [inter + sum(row[col] * w for w, col in zip(wi, x.columns)) for _, row in x.iterrows()]
            prob_list = [sigmoide(z) for z in z_list]
            loss = -np.mean([y_i*np.log(p+1e-8) + (1-y_i)*np.log(1-p+1e-8) for y_i,p in zip(y, prob_list)])
            accuracy = sum((1 if p >= 0.5 else 0) == y_i for p, y_i in zip(prob_list, y)) / len(y) * 100
            print(f"Iteración {loop}: pérdida = {loss:.5f}, exactitud = {accuracy:.4f}%")

            if lossi is not None and abs(prevloss - loss) < lossi:
                print(f"Convergencia alcanzada en iteración {loop}, con pérdida de {loss:.5f}")
                break
            prevloss=loss
        
    return wi, inter

def predecir(x, wi, inter):
    z_list = [inter + sum(row[col] * w for w, col in zip(wi, x.columns)) for _, row in x.iterrows()]
    probabilidades = [sigmoide(val) for val in z_list]
    predicciones = [1 if p >= 0.5 else 0 for p in probabilidades]
    return probabilidades, predicciones

def predecir_caso(fila, wi, inter):
    caso = inter + sum(fila[col] * w for w, col in zip(wi, x.columns))
    prob = sigmoide(caso)
    pred = "Maligno" if prob >= 0.5 else "Benigno"
    return prob, pred

timer1 = t.time()
wi, inter = entrenar(x_train, y_train, wi, inter, loops, a, len(y_train), l_reg, lossi)
timer2 = t.time()
timertot = timer2 - timer1
print(f"Tardo en entrenar {timertot:.5f} segundos")

prob_train, pred_train = predecir(x_train, wi, inter)
prob_test, pred_test = predecir(x_test, wi, inter)

exact_train = sum(pred == true for pred, true in zip(pred_train, y_train)) / len(y_train)
exact_test = sum(pred == true for pred, true in zip(pred_test, y_test)) / len(y_test)

print(f"Exactitud en train: {exact_train*100:.2f}%")
print(f"Exactitud en test: {exact_test*100:.2f}%")


indices = [41, 106, 484, 538]
ruta= f"/home/angelagar/MCDO/curva{loops}loops.png"
printCasos(wi, inter, x, y, indices, ruta, loops)
