import pandas as pd
import numpy as np
import random
import time as t
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv("MCDO/breast-cancer.csv")
y = df["diagnosis"].map({'M': 1, 'B': 0})
x = df.iloc[:, 2:]

scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

m = len(y)
idx = list(range(m))
random.seed(56)
random.shuffle(idx)
split = int(0.7 * m)

x_train, y_train = x_scaled.iloc[idx[:split]], y.iloc[idx[:split]]
x_test,  y_test  = x_scaled.iloc[idx[split:]], y.iloc[idx[split:]]

loops = 10000
start = t.time()

model = LogisticRegression(
    solver='saga',
    penalty='l2',
    C=1/0.01,
    max_iter=loops,
    tol=1e-6,
    random_state=56
)

model.fit(x_train, y_train)
end = t.time()

prob_train = model.predict_proba(x_train)[:, 1]
prob_test = model.predict_proba(x_test)[:, 1]
pred_train = (prob_train >= 0.5).astype(int)
pred_test = (prob_test >= 0.5).astype(int)

exact_train = accuracy_score(y_train, pred_train)
exact_test = accuracy_score(y_test, pred_test)
loss_train = log_loss(y_train, prob_train)
loss_test = log_loss(y_test, prob_test)

print(f"Tardó en entrenar {end-start:.5f} segundos")
print(f"Pérdida train: {loss_train:.5f}, Pérdida test: {loss_test:.5f}")
print(f"Exactitud train: {exact_train*100:.2f}%")
print(f"Exactitud test: {exact_test*100:.2f}%")

indices = [41, 106, 484, 538]
for i in indices:
    fila = x_scaled.iloc[i]
    z = model.intercept_[0] + np.dot(model.coef_[0], fila)
    prob = 1 / (1 + np.exp(-z))
    pred = "Maligno" if prob >= 0.5 else "Benigno"
    real = "Maligno" if y[i] == 1 else "Benigno"
    print(f"Fila {i+1}: Probabilidad={prob*100:.4f}%, Predicción={pred}, Real={real}")

