import requests
import csv
import time

# 🔹 Configuración
categoria_id = "MLM1051"  # Celulares México
limit_por_pagina = 50
url_base_busqueda = "https://api.mercadolibre.com/sites/MLM/search"
url_base_item = "https://api.mercadolibre.com/items"
output_csv = "celulares_mx_completo_descuentos.csv"

# 🔹 Función para obtener productos por página
def obtener_productos(offset=0):
    params = {
        "category": categoria_id,
        "limit": limit_por_pagina,
        "offset": offset
    }
    response = requests.get(url_base_busqueda, params=params)
    if response.status_code != 200:
        print(f"Error al obtener productos en offset {offset}")
        return []
    return response.json().get("results", [])

# 🔹 Función para obtener detalles de un producto
def obtener_detalles_producto(item_id):
    response = requests.get(f"{url_base_item}/{item_id}")
    if response.status_code != 200:
        return {}
    return response.json()

# 🔹 Abrir CSV para escritura
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # Encabezados
    writer.writerow([
        "id", "titulo", "precio_original", "precio_venta", "descuento",
        "porcentaje_descuento", "descripcion", "moneda", "link",
        "marca", "modelo", "memoria", "ram", "almacenamiento",
        "stock_disponible"
    ])

    offset = 0
    total_descargados = 0

    while True:
        productos = obtener_productos(offset)
        if not productos:
            break

        for item in productos:
            precio_venta = item.get("price")
            precio_original = item.get("original_price") or precio_venta
            descuento = precio_original - precio_venta
            porcentaje_descuento = round((descuento / precio_original) * 100, 2) if precio_original else 0
            descripcion = item.get("title")
            stock_disponible = item.get("available_quantity")

            detalles = obtener_detalles_producto(item.get("id"))

            # Inicializar atributos técnicos como None si no existen
            marca = modelo = memoria = ram = almacenamiento = None
            for attr in detalles.get("attributes", []):
                if attr.get("id") == "BRAND":
                    marca = attr.get("value_name")
                elif attr.get("id") == "MODEL":
                    modelo = attr.get("value_name")
                elif attr.get("id") == "MEMORY":
                    memoria = attr.get("value_name")
                elif attr.get("id") == "RAM":
                    ram = attr.get("value_name")
                elif attr.get("id") == "INTERNAL_MEMORY":
                    almacenamiento = attr.get("value_name")

            writer.writerow([
                item.get("id"),
                descripcion,
                precio_original,
                precio_venta,
                descuento,
                porcentaje_descuento,
                descripcion,
                item.get("currency_id"),
                item.get("permalink"),
                marca,
                modelo,
                memoria,
                ram,
                almacenamiento,
                stock_disponible
            ])
            total_descargados += 1
            time.sleep(0.1)  # pausa pequeña para no saturar la API

        print(f"Productos descargados: {total_descargados}")
        offset += limit_por_pagina
        time.sleep(0.5)  # pausa entre páginas

print(f"Se descargaron un total de {total_descargados} productos en {output_csv}")
