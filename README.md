# TSP España — paquete DEMO ejecutable

## Qué es esta versión
Esta carpeta contiene una versión, para que el equipo pruebe el software antes de sustituir los datos DEMO por los datos documentados del trabajo.

**Importante:** `rutas_2020.csv` y `rutas_2026.csv` son generadas por `src/generar_demo.py` y son **datos sintéticos de demostración**. No representan tarifas, tiempos ni distancias oficiales de España.

El archivo `data/capitales.xlsx` sí es la base real entregada por el equipo, con las 47 capitales y sus coordenadas.

## 1. Instalar

```bash
python -m venv .venv
```
Windows:
```bash
.venv\Scripts\activate
```
macOS/Linux:
```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

## 2. Ejecutar la demo completa
Windows:
```bash
run_demo.bat
```

macOS/Linux:
```bash
./run_demo.sh
```

O manualmente:
```bash
python src/generar_demo.py
python src/tsp_espana.py --scenario both --sensitivity
```

## 3. Qué debe aparecer
Se generan:

- `results/resumen_principal.csv`
- `results/contraste_2020_2026.csv`
- `results/sensibilidad_2020.csv`
- `results/sensibilidad_2026.csv`
- gráficos de convergencia ACO/GA
- mapas esquemáticos de las rutas
- GIF de evolución de ACO y GA
- matrices de costos y rutas seleccionadas para cada escenario

## 4. Qué significa la arquitectura
Para cada par de capitales puede haber varias rutas candidatas. El programa calcula:

`costo = valor_hora * tiempo + peaje + combustible`

con:

`combustible = distancia_km * consumo_L_100km/100 * precio_eur_L`

y selecciona la ruta candidata de menor costo generalizado antes de resolver el TSP.

## 5. Para el trabajo real
Después de aprobar esta demo, se reemplazarán los archivos DEMO por:

- `rutas_2020.csv`: datos históricos realmente verificables.
- `rutas_2026.csv`: datos actuales realmente verificables.
- `parametros.json`: vehículo, combustible y valor de la hora con sus fuentes.

