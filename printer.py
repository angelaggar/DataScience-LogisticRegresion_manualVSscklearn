import numpy as np
import matplotlib.pyplot as plt

def printCasos(wi, inter, x_original, y_original, indices, ruta=None, loops=None):
    def sigmoide(z):
        return 1 / (1 + np.exp(-z))
    
    umbral = 0.5
    z_vals = np.linspace(-10, 10, 300)
    sigmoid_curve = [sigmoide(z) for z in z_vals]

    plt.figure(figsize=(8,6))
    plt.plot(z_vals, sigmoid_curve, label="Curva sigmoide", color='blue')

    colores = ['red', 'green', 'orange', 'purple', 'brown']

    for idx_num, idx in enumerate(indices):
        fila = x_original.iloc[idx]
        valor_real = y_original.iloc[idx]
        z_caso = inter + sum(w * fila[col] for w, col in zip(wi, x_original.columns))
        prob_caso = sigmoide(z_caso)
        pred_caso = "Maligno" if prob_caso >= umbral else "Benigno"
        real_caso = "Maligno" if valor_real == 1 else "Benigno"

        color = colores[idx_num % len(colores)]
        plt.scatter(z_caso, prob_caso, color=color, s=100, label=f"Caso fila {idx+1}")
        plt.text(z_caso, prob_caso + 0.05, f"P={prob_caso*100:.2f}%\nPred={pred_caso}", ha='center', color=color)

        print(f"Fila {idx+1}: Probabilidad={prob_caso:.4f}, Predicción={pred_caso}, Real={real_caso}")

    plt.axhline(umbral, color='gray', linestyle='--', label=f"Umbral {umbral}")
    plt.title("Curva Sigmoide de regresión logística y casos seleccionados \n")
    plt.xlabel("Valor z (intercepto + suma ponderada)")
    plt.ylabel("Probabilidad de malignidad")
    plt.legend()
    plt.grid(True)

    if ruta:
        plt.savefig(ruta, dpi=300)
    else:
        plt.savefig(f"curva{loops}casos.png", dpi=300)
    plt.close()
