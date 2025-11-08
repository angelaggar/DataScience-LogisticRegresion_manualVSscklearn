import matplotlib.pyplot as plt
import imageio
import numpy as np
import os

# Configuración
loops = 20
loss = np.exp(-0.2 * np.arange(loops)) + 0.05 * np.random.rand(loops)
convergence_threshold = 0.05
carpeta_salida = "diagramas"
os.makedirs(carpeta_salida, exist_ok=True)

filenames = []

for i in range(loops):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.axis('off')
    
    steps = [
        "Inicio",
        "Cargar y dividir dataset",
        "Normalizar variables",
        "Inicializar parámetros del modelo",
        "Calcular pesos de clase",
        "Iterar hasta convergencia",
        "Evaluar modelo con datos de prueba",
        "Calcular métricas de desempeño",
        "Generar curvas de probabilidad/predicciones",
        "Fin"
    ]
    
    y_pos = np.linspace(4, 0, len(steps))
    
    for j, (text, y) in enumerate(zip(steps, y_pos)):
        color = "lightgray"
        if text == "Iterar hasta convergencia":
            if i == loops-1 or abs(loss[i]-loss[i-1]) < convergence_threshold:
                color = "lightgreen"  # convergencia alcanzada
            else:
                color = "orange"      # aún en proceso
        ax.text(2.5, y, text, bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.5'),
                ha='center', va='center', fontsize=10)
        if j > 0:
            ax.arrow(2.5, y_pos[j-1]-0.3, 0, -0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')

    ax.text(4.5, 4, f"Iteración: {i+1}/{loops}", fontsize=10, ha='right')
    ax.text(4.5, 3.6, f"Pérdida: {loss[i]:.4f}", fontsize=10, ha='right')
    
    filename = os.path.join(carpeta_salida, f'_tmp_flow_{i}.png')
    plt.savefig(filename)
    filenames.append(filename)
    plt.close()

# Guardar GIF
gif_path = os.path.join(carpeta_salida, '/home/angelagar/MCDO/flujo_convergencia.gif')
with imageio.get_writer(gif_path, mode='I', duration=0.7) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Guardar PNG final “profesional”
png_final = os.path.join(carpeta_salida, '/home/angelagar/MCDO/flujo_convergencia.png')
# Para el PNG final, reutilizamos el último frame y resaltamos la convergencia
fig, ax = plt.subplots(figsize=(8,5))
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.axis('off')
y_pos = np.linspace(4, 0, len(steps))
for j, (text, y) in enumerate(zip(steps, y_pos)):
    color = "lightgreen" if text == "Iterar hasta convergencia" else "lightgray"
    ax.text(2.5, y, text, bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.5'),
            ha='center', va='center', fontsize=10)
    if j > 0:
        ax.arrow(2.5, y_pos[j-1]-0.3, 0, -0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
ax.text(4.5, 4, f"Iteración: {loops}/{loops}", fontsize=10, ha='right')
ax.text(4.5, 3.6, f"Pérdida: {loss[-1]:.4f}", fontsize=10, ha='right')
plt.savefig(png_final)
plt.close()

# Limpiar archivos temporales intermedios
for filename in filenames:
    os.remove(filename)

print(f"GIF guardado en: {gif_path}")
print(f"PNG final guardado en: {png_final}")
