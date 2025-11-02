import requests
import csv

# 🔹 Parámetros de búsqueda
categoria_id = "MLA1051"  # reemplaza con la categoría que necesites
q = None                   # término de búsqueda (opcional)
limit = 50                 # cantidad máxima de productos a traer

# 🔹 Construir la URL de la API
url = f"https://api.mercadolibre.com/sites/MLA/search?category={categoria_id}&limit={limit}"
if q:
    url += f"&q={q}"

# 🔹 Hacer la solicitud
response = requests.get(url)
data = response.json()

# 🔹 Guardar resultados en CSV
with open("productos.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # Encabezados
    writer.writerow(["id", "titulo", "precio", "moneda", "link", "thumbnail"])
    
    for item in data.get("results", []):
        writer.writerow([
            item["id"],
            item["title"],
            item["price"],
            item["currency_id"],
            item["permalink"],
            item["thumbnail"]
        ])

print(f"Se descargaron {len(data.get('results', []))} productos en productos.csv")