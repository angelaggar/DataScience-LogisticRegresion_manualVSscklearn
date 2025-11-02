import requests
import csv
import time

# 🔹 Configuración
categoria_id = "MLM1055"  # categoría Celulares en MercadoLibre México
productos_a_descargar = 1000
limit_por_pagina = 50  # máximo por solicitud
url_base = "https://api.mercadolibre.com/sites/MLM/search"

# 🔹 Función para obtener productos por página
def obtener_productos(offset=0):
    params = {
        "category": categoria_id,
        "limit": limit_por_pagina,
        "offset": offset
    }
    response = requests.get(url_base, params=params)
    return response.json().get("results", [])

# 🔹 Descargar todos los productos
todos_productos = []
for offset in range(0, productos_a_descargar, limit_por_pagina):
    productos = obtener_productos(offset)
    if not productos:
        break
    todos_productos.extend(productos)
    print(f"Descargados hasta el offset {offset + limit_por_pagina}")
    time.sleep(0.5)  # evitar sobrecargar la API

# 🔹 Guardar en CSV
with open("celulares_mx.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "titulo", "precio_original", "precio_venta", "descripcion", "moneda", "link"])
    
    for item in todos_productos[:productos_a_descargar]:
        precio_original = item.get("original_price") or item.get("price")
        precio_venta = item.get("price")
        descripcion = item.get("title")  # título como descripción brev
