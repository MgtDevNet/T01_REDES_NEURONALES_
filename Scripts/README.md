# TSP · 47 capitales de la España peninsular · ACO vs GA

Problema del vendedor viajero (ruta cerrada) sobre las 47 capitales de provincia peninsulares,
resuelto con **Colonia de Hormigas (ACO)** y **Algoritmo Genético (GA)** en Python puro.

## Ejecución (un solo comando)

```bash
python tsp_bio.py
```

Requisitos: Python 3.8+, con `datos_tsp.py`, `tsp_bio.py` y `config.ini` en la misma carpeta. No hace falta instalar nada.
Si `matplotlib` está instalado (`pip install matplotlib`), también se genera `resultados.png`.

## Hiperparámetros: `config.ini`

Todos los hiperparámetros del ACO y del GA se cambian en **`config.ini`**, sin tocar el código.
Cada línea indica entre corchetes su valor por defecto y una breve explicación.

```ini
[aco]
beta = 5.0          ; antes 3.0
[ga]
generaciones = 800  ; antes 500
```

- Si borras una línea (o el archivo entero), se usa el valor por defecto.
- Si escribes mal un nombre (`betta`), un tipo (`rho = abc`) o un valor fuera de rango (`rho = 1.5`),
  el programa se detiene con un mensaje que indica la línea exacta.
- La cabecera de cada ejecución imprime la configuración efectiva, así cada resultado queda documentado.
- La sección `[puro]` define qué valores cambian en el modo clásico (sin 2-opt).
- Si cambias la matriz de distancias (p. ej., por carretera), pon `optimo_referencia_km = ninguno`
  o el nuevo valor exacto.

Las opciones de terminal tienen prioridad sobre `config.ini`:

| Opción | Efecto |
|---|---|
| `--seed 7` | Cambia la semilla aleatoria (misma semilla ⇒ mismo resultado) |
| `--inicio Sevilla` | Muestra el recorrido empezando en otra capital |
| `--modo puro` o `--puro` | ACO y GA clásicos, sin búsqueda local 2-opt (para comparar) |
| `--config otro.ini` | Usa otro archivo de configuración (útil para guardar varios experimentos) |

## Archivos

| Archivo | Contenido |
|---|---|
| `datos_tsp.py` | Coordenadas de las 47 capitales y matriz de distancias (Haversine, km) |
| `config.ini` | Hiperparámetros de ACO y GA, semilla, modo y archivos de salida |
| `tsp_bio.py` | ACO, GA, búsqueda local 2-opt y presentación de resultados |
| `resultados.csv` | Orden de visita de cada algoritmo (se genera al ejecutar) |
| `resultados.png` | Mapas de las rutas y curva de convergencia (se genera al ejecutar) |

## Qué hace el programa, paso a paso

1. **Carga los datos** de `datos_tsp.py`: 47 ciudades y la matriz `D[i][j]` en km.
2. **Calcula una referencia voraz** (vecino más cercano) como punto de comparación.
3. **Ejecuta el ACO.** En cada iteración, 20 hormigas construyen rutas eligiendo la siguiente ciudad
   con probabilidad ∝ τ^α · (1/d)^β. La mejor ruta de la iteración se pule con 2-opt. Después, la feromona
   se evapora (ρ = 0,1) y se refuerza en las aristas de las buenas rutas, con un refuerzo extra para la mejor global.
4. **Ejecuta el GA.** Parte de 150 rutas aleatorias. En cada generación aplica selección por torneo (k = 3),
   cruce OX, mutación por inversión y 2-opt a un 5 % de los hijos (GA *memético*). Conserva las 2 mejores rutas
   (elitismo) y descarta los clones para mantener la diversidad.
5. **Muestra los resultados**: la ruta de cada algoritmo tramo a tramo, una tabla resumen (longitud,
   diferencia con el óptimo, iteración en la que se halló y tiempo), el CSV y la gráfica.

## Resultado esperado (semilla 42)

| Algoritmo | Longitud | Diferencia con el óptimo | Tiempo aprox. |
|---|---|---|---|
| ACO + 2-opt | 4798,7 km | 0,00 % | ~1 s |
| GA memético | 4798,7 km | 0,00 % | ~4 s |
| Vecino más cercano | 5884,5 km | +22,6 % | — |

El óptimo exacto para estos datos (**4798,7 km**) se verificó por separado con programación lineal entera
(formulación DFJ con eliminación de subtours). Solo se usa para medir la diferencia; los algoritmos no lo conocen.

Con `--puro` los algoritmos clásicos quedan aproximadamente un 3–4 % por encima del óptimo.
Así se ve el aporte de la búsqueda local.

Ruta óptima desde Madrid:

Madrid → Toledo → Ciudad Real → Cáceres → Badajoz → Huelva → Cádiz → Sevilla → Córdoba → Jaén →
Málaga → Granada → Almería → Murcia → Alicante → Albacete → Cuenca → Teruel → Valencia →
Castellón de la Plana → Tarragona → Barcelona → Girona → Lleida → Huesca → Zaragoza → Soria → Logroño →
Pamplona → San Sebastián → Vitoria-Gasteiz → Bilbao → Santander → Burgos → Palencia → Valladolid → León →
Oviedo → Lugo → A Coruña → Pontevedra → Ourense → Zamora → Salamanca → Ávila → Segovia → Guadalajara → Madrid
