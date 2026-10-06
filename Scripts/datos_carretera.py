"""Descarga la matriz 47x47 de distancias por carretera (km) desde OSRM.
Se ejecuta UNA vez; el resultado se guarda en distancias_carretera.csv."""
import csv
import json
import urllib.request

from datos_tsp import CAPITALES, CIUDADES

# OSRM espera "lon,lat" separados por ';'
coords = ";".join(f"{lon},{lat}" for _, lat, lon in CAPITALES)
url = f"https://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance"

with urllib.request.urlopen(url, timeout=60) as r:
    datos = json.load(r)
assert datos["code"] == "Ok", datos

M = [[d / 1000 for d in fila] for fila in datos["distances"]]  # metros → km

with open("distancias_carretera.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["ciudad"] + CIUDADES)
    for c, fila in zip(CIUDADES, M):
        w.writerow([c] + [f"{d:.2f}" for d in fila])
print("Guardado distancias_carretera.csv")