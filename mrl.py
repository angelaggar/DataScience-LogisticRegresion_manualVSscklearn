import pandas as pd
import numpy as np
from data import burnout_mapping, columnas_relevantes, clasificar_edad

# ===============================================================
# 🧩 1️⃣ Cargar y preparar los datos base
# ===============================================================
print("\n📥 Cargando datos de la encuesta...")
df = pd.read_csv("MCDO/survey.csv")

# Filtramos edades fuera de rango (valores atípicos)
df["Age"] = df["Age"].apply(lambda x: np.nan if x < 15 or x > 80 else x)

# Convertimos edad a rango y luego a número
def edad_a_num(edad):
    grupo = clasificar_edad(edad)
    grupos = {"18-24": 1, "25-34": 2, "35-44": 3, "45-54": 4, "55+": 5}
    return grupos.get(grupo, np.nan)

df["Age_Num"] = df["Age"].apply(edad_a_num)

# Variable objetivo
df["burnout"] = df["work_interfere"].map({
    "Never": 0, "Rarely": 0, "Sometimes": 1, "Often": 1
})

print("✅ Datos base preparados correctamente.\n")

# ===============================================================
# 🔧 2️⃣ Mapear respuestas categóricas a valores numéricos
# ===============================================================
df_mapped = df.copy()

for col in columnas_relevantes:
    if col in df_mapped.columns and col in burnout_mapping:
        df_mapped[col] = df_mapped[col].map(burnout_mapping[col])

df_mapped.fillna(0, inplace=True)

# ===============================================================
# 📊 3️⃣ Preparar datos para el modelo (sin Gender)
# ===============================================================
# Normalizamos edad
df_mapped["Age_Num"] = df["Age_Num"].fillna(df["Age_Num"].mean())

# Construimos las columnas finales de predictores (omitimos Gender)
columnas_finales = columnas_relevantes + ["Age_Num"]

# Aseguramos que todas estén presentes
for col in columnas_finales:
    if col not in df_mapped.columns:
        df_mapped[col] = 0

# Creamos matrices de entrenamiento
X = df_mapped[columnas_finales].astype(float)
y = df_mapped["burnout"].astype(float).values.reshape(-1, 1)

# Verificación rápida de datos
if np.isnan(X.values).any():
    print("⚠️ Advertencia: se encontraron valores NaN en X. Se reemplazarán por 0.")
    X = X.fillna(0)
if np.isnan(y).any():
    print("⚠️ Advertencia: se encontraron valores NaN en y. Se reemplazarán por 0.")
    y = np.nan_to_num(y)

# Normalizar edad
columnas_continuas = ["Age_Num"]
medias = X[columnas_continuas].mean()
stds = X[columnas_continuas].std().replace(0, 1)  # Evita división por 0

for col in columnas_continuas:
    X[col] = (X[col] - medias[col]) / stds[col]

# Agregar columna de 1s (intercepto)
X = np.hstack([np.ones((X.shape[0], 1)), X.values])

# ===============================================================
# 🧮 4️⃣ Funciones de regresión logística
# ===============================================================
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Evita overflow numérico
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# ===============================================================
# 🧠 5️⃣ Entrenamiento del modelo
# ===============================================================
print("⚙️ Entrenando modelo de regresión logística...\n")

weights = np.zeros((X.shape[1], 1))
lr = 0.01
n_iterations = 5000

for i in range(n_iterations):
    z = np.dot(X, weights)
    y_pred = sigmoid(z)
    gradient = np.dot(X.T, (y_pred - y)) / y.size
    weights -= lr * gradient

    if i % 500 == 0:
        loss = compute_loss(y, y_pred)
        print(f"Iteración {i:>4} | Pérdida: {loss:.4f}")

# ===============================================================
# 📈 6️⃣ Evaluación del modelo
# ===============================================================
y_prob = sigmoid(np.dot(X, weights))
y_pred = (y_prob >= 0.5).astype(int)
accuracy = (y_pred == y).mean()

print("\n✅ Entrenamiento completado.")
print(f"🎯 Exactitud del modelo: {accuracy:.2%}\n")

coef_df = pd.DataFrame({
    "Variable": ["Intercept"] + columnas_finales,
    "Coeficiente": weights.flatten()
})
print("📊 Coeficientes (impacto sobre burnout):")
print(coef_df.head(10))

# ===============================================================
# 🔍 7️⃣ Predicción para un nuevo encuestado (sin Gender)
# ===============================================================
def predecir_burnout(nuevo_encuestado, weights, medias, stds):
    df_nueva = nuevo_encuestado.copy()

    # Mapear categorías a valores numéricos (según burnout_mapping)
    for col in columnas_relevantes:
        if col in df_nueva.columns and col in burnout_mapping:
            df_nueva[col] = df_nueva[col].map(burnout_mapping[col])

    # Calcular variables adicionales
    df_nueva["Age_Num"] = df_nueva["Age"].apply(edad_a_num)

    # Normalizar la edad
    df_nueva["Age_Num"] = (df_nueva["Age_Num"] - medias["Age_Num"]) / stds["Age_Num"]

    # Aseguramos que todas las columnas necesarias estén presentes
    for col in columnas_finales:
        if col not in df_nueva.columns:
            df_nueva[col] = 0

    # Seleccionamos solo las columnas finales
    X_nueva = df_nueva[columnas_finales].astype(float).values
    X_nueva = np.hstack([np.ones((X_nueva.shape[0], 1)), X_nueva])

    # Calcular probabilidad de burnout
    prob = sigmoid(np.dot(X_nueva, weights))
    return prob

# ===============================================================
# 🧾 8️⃣ Prueba con nueva encuesta
# ===============================================================
nuevo_encuestado = pd.DataFrame([{
    'Age': 29,
    'work_interfere': 'Often',
    'family_history': 'Yes',
    'treatment': 'No',
    'remote_work': 'Yes',
    'tech_company': 'Yes',
    'benefits': 'No',
    'care_options': 'Yes',
    'wellness_program': 'No',
    'seek_help': 'Yes',
    'leave': 'Somewhat easy',
    'mental_health_consequence': 'No',
    'phys_health_consequence': 'No',
    'obs_consequence': 'No',
    'coworkers': 'Yes',
    'supervisor': 'Yes',
    'mental_health_interview': 'No',
    'phys_health_interview': 'No',
    'mental_vs_physical': 'Yes'
}])

prob = predecir_burnout(nuevo_encuestado, weights, medias, stds)
valor = float(prob[0][0])

if valor < 0.33:
    riesgo = "Bajo riesgo"
elif valor < 0.66:
    riesgo = "Riesgo moderado"
else:
    riesgo = "Alto riesgo"

print("\n🧠 Resultado para el nuevo encuestado:")
print(f"➡️  Probabilidad estimada de burnout: {valor:.2%}")
print(f"📌 Interpretación: {riesgo.upper()}")
