"""
Trabajo 1 - Parte 1: optimización numérica sobre Rosenbrock (2D y 3D).

Métodos: descenso por gradiente (GD), algoritmo genético (GA), PSO y evolución
diferencial (DE). Adem�s una l�ea base opcional: b�squeda aleatoria.

Comandos (todos con un solo comando, sin editar código):
    python punto_1/funcion1.py check
    python punto_1/funcion1.py run   --algo pso --dim 2 --seed 42
    python punto_1/funcion1.py stats --runs 30 --dims 2 3
    python punto_1/funcion1.py gif   --algo gradient_descent --out figs/gd.gif

Los hiperparámetros se cambian con --param (p. ej. --param tasa_aprendizaje_inicial=0.5
--param pso.inercia=0.6) o con un archivo --config config.json.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

__all__ = [
    "optimize",
    "graficar_optimizacion",
    "make_gif",
    "ALGORITHMS",
    "PROBLEMS",
    "MAIN_ALGOS",
    "run_batch",
    "OptimizeResult",
]


# ======================================================================
# 1. Función de prueba
# ======================================================================
def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def rosenbrock_grad(x):
    """Gradiente analítico. Solo se usa en `check` para validar el numérico."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    g[:-1] = -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return g

def rastrigin(x):
    """
    Calcula la función de Rastrigin para un vector x de N dimensiones.
    x: array_like (puede ser un vector 1D o una matriz de puntos)
    A: constante de amplitud (por defecto 10)
    """
    A = 10
    x = np.asarray(x, dtype=float)
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rastrigin_grad(x):
    """
    Calcula el gradiente de la función de Rastrigin para un vector x de N dimensiones.
    x: array_like (puede ser un vector 1D o una matriz de puntos)
    A: constante de amplitud (por defecto 10)
    """
    A = 10
    x = np.asarray(x, dtype=float)
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)


PROBLEMS: dict[str, dict] = {}


def register_problem(name, func, bounds, supported_dims=None,
                     known_optimum=None, grad=None):
    PROBLEMS[name] = {"func": func, "bounds": bounds,
                      "supported_dims": supported_dims,
                      "known_optimum": known_optimum, "grad": grad}


register_problem(
    "rosenbrock", rosenbrock, bounds=(-2.048, 2.048), supported_dims=(2, 3),
    known_optimum=lambda d: np.ones(d), grad=rosenbrock_grad,
)

register_problem(
    "rastrigin", rastrigin, bounds=(-5.12, 5.12), supported_dims=(2, 3),
    known_optimum=lambda d: np.zeros(d), grad=rastrigin_grad,
)

# ======================================================================
# 2. Conteo de evaluaciones (f y ∇f por separado)
# ======================================================================
class CountedObjective:
    """
    n_f     : llamadas directas a f (población, búsqueda lineal, punto inicial).
    n_grad  : número de gradientes calculados.
    Equivalencia: un gradiente por diferencias centrales cuesta 2*dim
    evaluaciones de f, así que  n_equiv = n_f + 2*dim*n_grad.
    Las 2*dim evaluaciones internas del gradiente NO se suman a n_f
    (se cuentan una sola vez, dentro de n_grad).
    """

    def __init__(self, func: Callable, dim: int):
        self._func = func
        self.dim = dim
        self.n_f = 0
        self.n_grad = 0
        self.raw_calls = 0      # llamadas reales a func; solo para verificar

    @property
    def cost_grad(self) -> int:
        return 2 * self.dim

    @property
    def n_equiv(self) -> int:
        return self.n_f + self.cost_grad * self.n_grad

    def _raw(self, x) -> float:
        self.raw_calls += 1
        return float(self._func(x))

    def __call__(self, x) -> float:
        self.n_f += 1
        return self._raw(x)

    def gradient(self, x, h: float = 1e-6) -> np.ndarray:
        self.n_grad += 1
        g = np.empty(self.dim)
        for i in range(self.dim):
            e = np.zeros(self.dim)
            e[i] = h
            g[i] = (self._raw(x + e) - self._raw(x - e)) / (2.0 * h)
        return g


# ======================================================================
# 3. Resultado
# ======================================================================
@dataclass
class OptimizeResult:
    problem: str
    algorithm: str
    dim: int
    random_seed: int | None
    x_best: np.ndarray
    f_best: float
    n_f: int
    n_grad: int
    n_equiv: int
    history: list = field(default_factory=list, repr=False)     # mejor f por iteración
    eval_trace: list = field(default_factory=list, repr=False)  # n_equiv en cada iteración
    frames: list | None = field(default=None, repr=False)       # para animación / GIF
    fig: object | None = field(default=None, repr=False)        # figura interactiva de Plotly

    def __str__(self) -> str:
        return (
            f"{self.algorithm} | {self.problem} {self.dim}D | seed={self.random_seed}\n"
            f"  x_best  = {np.array2string(self.x_best, precision=6)}\n"
            f"  f_best  = {self.f_best:.6e}\n"
            f"  evals f = {self.n_f} | evals grad = {self.n_grad} "
            f"| equivalentes (f) = {self.n_equiv}"
        )


# ======================================================================
# 4. Algoritmos
# ======================================================================
# Firma común:
#   algo(obj, dim, bounds, rng, max_evals, record, **params)
#       -> (x_best, f_best, history, eval_trace, frames)
# `max_evals` es un presupuesto en evaluaciones EQUIVALENTES de f (obj.n_equiv),
# idéntico para todos los métodos => comparación justa.
ALGORITHMS: dict[str, Callable] = {}
_RESERVED = {"obj", "dim", "bounds", "rng", "max_evals", "record"}


def register_algorithm(name: str):
    def decorator(fn):
        ALGORITHMS[name] = fn
        return fn
    return decorator


def accepted_params(algo: str) -> set[str]:
    return set(inspect.signature(ALGORITHMS[algo]).parameters) - _RESERVED


# ---------------- Descenso por gradiente ----------------
@register_algorithm("gradient_descent")
def gradient_descent(obj, dim, bounds, rng, max_evals, record=False,
                     tolerancia=1e-8, paso_gradiente=1e-6, tasa_aprendizaje_inicial=1.0,
                     factor_armijo=1e-4, factor_reduccion=0.5, max_busqueda_lineal=60, x0=None):
    """x_{k+1} = P(x_k - a_k g_k), g_k por diferencias centrales, a_k por
    Armijo con retroceso (a <- factor_reduccion*a) y P = proyección al dominio."""
    lim_inf, lim_sup = bounds
    if x0 is None:
        x = rng.uniform(lim_inf, lim_sup, dim)
    else:
        x = np.asarray(x0, dtype=float)
        if x.shape != (dim,) or not np.isfinite(x).all():
            raise ValueError(f'x0 debe contener {dim} coordenadas finitas')
        if ((x < lim_inf) | (x > lim_sup)).any():
            raise ValueError(f'x0 debe estar en [{lim_inf}, {lim_sup}]')
        x = x.copy()
    fx = obj(x)
    historial, traza_evals = [fx], [obj.n_equiv]
    frames = [x.copy()] if record else None

    while obj.n_equiv + obj.cost_grad + 1 <= max_evals:
        gradiente = obj.gradient(x, paso_gradiente)
        if np.linalg.norm(gradiente) < tolerancia:
            break
        tasa_aprendizaje, aceptado = tasa_aprendizaje_inicial, False
        for _ in range(max_busqueda_lineal):
            if obj.n_equiv + 1 > max_evals:
                break
            candidato = np.clip(x - tasa_aprendizaje * gradiente, lim_inf, lim_sup)
            f_candidato = obj(candidato)
            # Armijo con proyección: f(candidato) <= f(x) - factor_armijo * g.(x - candidato)
            if f_candidato <= fx - factor_armijo * float(gradiente @ (x - candidato)):
                aceptado = True
                break
            tasa_aprendizaje *= factor_reduccion
        if not aceptado or np.array_equal(candidato, x):
            break
        x, fx = candidato, f_candidato
        historial.append(fx)
        traza_evals.append(obj.n_equiv)
        if record:
            frames.append(x.copy())
    return x, fx, historial, traza_evals, frames


# ---------------- Algoritmo genético (codificación real) ----------------
@register_algorithm("genetic")
def genetic(obj, dim, bounds, rng, max_evals, record=False,
            tam_poblacion=50, individuos_elite=2, tam_torneo=3, prob_cruce=0.9,
            alpha_blx=0.5, sigma_inicial=0.1, sigma_final=1e-4):
    """Selección por torneo, cruce BLX-alpha (prob. prob_cruce), mutación gaussiana por
    gen (prob. 1/dim) con sigma decreciente (geométrico, fracción del rango) y
    elitismo."""
    lim_inf, lim_sup = bounds
    rango = lim_sup - lim_inf
    n_hijos = tam_poblacion - individuos_elite
    poblacion = rng.uniform(lim_inf, lim_sup, (tam_poblacion, dim))
    aptitud = np.array([obj(p) for p in poblacion])
    historial, traza_evals = [float(aptitud.min())], [obj.n_equiv]
    frames = [poblacion.copy()] if record else None

    while obj.n_equiv + n_hijos <= max_evals:
        sigma = sigma_inicial * rango * (sigma_final / sigma_inicial) ** (obj.n_equiv / max_evals)
        # torneo: para cada hijo, dos padres
        idx = rng.integers(0, tam_poblacion, (n_hijos, 2, tam_torneo))
        best_in_t = aptitud[idx].argmin(axis=2)
        win = np.take_along_axis(idx, best_in_t[..., None], axis=2)[..., 0]
        padre1, padre2 = poblacion[win[:, 0]], poblacion[win[:, 1]]
        # BLX-alpha
        d = np.abs(padre1 - padre2)
        hijos = rng.uniform(np.minimum(padre1, padre2) - alpha_blx * d,
                            np.maximum(padre1, padre2) + alpha_blx * d)
        hijos = np.where((rng.random(n_hijos) < prob_cruce)[:, None], hijos, padre1)
        # mutación gaussiana
        mut = rng.random((n_hijos, dim)) < 1.0 / dim
        hijos = np.clip(hijos + mut * rng.normal(0.0, sigma, (n_hijos, dim)), lim_inf, lim_sup)
        aptitud_hijos = np.array([obj(c) for c in hijos])
        # elitismo
        orden = np.argsort(aptitud)[:individuos_elite]
        poblacion = np.vstack([poblacion[orden], hijos])
        aptitud = np.concatenate([aptitud[orden], aptitud_hijos])
        historial.append(float(aptitud.min()))
        traza_evals.append(obj.n_equiv)
        if record:
            frames.append(poblacion.copy())
    i = int(aptitud.argmin())
    return poblacion[i].copy(), float(aptitud[i]), historial, traza_evals, frames


# ---------------- PSO ----------------
@register_algorithm("pso")
def pso(obj, dim, bounds, rng, max_evals, record=False,
        num_particulas=30, inercia=0.7298, coef_cognitivo=1.49618,
        coef_social=1.49618, frac_velocidad_max=0.2):
    """v <- inercia v + coef_cognitivo r1 (pbest - x) + coef_social r2 (gbest - x);  x <- x + v.
    Topología global (gbest). Al tocar la frontera: x se recorta y v=0."""
    lim_inf, lim_sup = bounds
    vel_max = frac_velocidad_max * (lim_sup - lim_inf)
    posiciones = rng.uniform(lim_inf, lim_sup, (num_particulas, dim))
    velocidades = rng.uniform(-vel_max, vel_max, (num_particulas, dim))
    fx = np.array([obj(p) for p in posiciones])
    mejor_personal, aptitud_personal = posiciones.copy(), fx.copy()
    g = int(aptitud_personal.argmin())
    mejor_global, aptitud_global = mejor_personal[g].copy(), float(aptitud_personal[g])
    historial, traza_evals = [aptitud_global], [obj.n_equiv]
    frames = [posiciones.copy()] if record else None

    while obj.n_equiv + num_particulas <= max_evals:
        r1 = rng.random((num_particulas, dim))
        r2 = rng.random((num_particulas, dim))
        velocidades = np.clip(inercia * velocidades + coef_cognitivo * r1 * (mejor_personal - posiciones) +
                              coef_social * r2 * (mejor_global - posiciones), -vel_max, vel_max)
        posiciones = posiciones + velocidades
        fuera = (posiciones < lim_inf) | (posiciones > lim_sup)
        posiciones = np.clip(posiciones, lim_inf, lim_sup)
        velocidades[fuera] = 0.0
        fx = np.array([obj(p) for p in posiciones])
        mejora = fx < aptitud_personal
        mejor_personal[mejora], aptitud_personal[mejora] = posiciones[mejora], fx[mejora]
        g = int(aptitud_personal.argmin())
        if aptitud_personal[g] < aptitud_global:
            mejor_global, aptitud_global = mejor_personal[g].copy(), float(aptitud_personal[g])
        historial.append(aptitud_global)
        traza_evals.append(obj.n_equiv)
        if record:
            frames.append(posiciones.copy())
    return mejor_global, aptitud_global, historial, traza_evals, frames


# ---------------- Evolución diferencial ----------------
@register_algorithm("differential_evolution")
def differential_evolution(obj, dim, bounds, rng, max_evals, record=False,
                           tam_poblacion=40, factor_mutacion=0.5, tasa_cruce=0.9):
    """DE/rand/1/bin:  v = x_a + factor_mutacion (x_b - x_c);  cruce binomial con tasa_cruce (al
    menos un gen del mutante);  selección voraz  u reemplaza a x si f(u)<=f(x)."""
    if tam_poblacion < 4:
        raise ValueError("DE requiere tam_poblacion >= 4")
    lim_inf, lim_sup = bounds
    poblacion = rng.uniform(lim_inf, lim_sup, (tam_poblacion, dim))
    aptitud = np.array([obj(p) for p in poblacion])
    historial, traza_evals = [float(aptitud.min())], [obj.n_equiv]
    frames = [poblacion.copy()] if record else None
    filas = np.arange(tam_poblacion)

    while obj.n_equiv + tam_poblacion <= max_evals:
        # tres índices distintos entre sí y distintos de i
        r = rng.random((tam_poblacion, tam_poblacion))
        r[filas, filas] = 2.0
        a, b, c = np.argsort(r, axis=1)[:, :3].T
        mutante = poblacion[a] + factor_mutacion * (poblacion[b] - poblacion[c])
        mascara = rng.random((tam_poblacion, dim)) < tasa_cruce
        mascara[filas, rng.integers(0, dim, tam_poblacion)] = True
        vector_prueba = np.where(mascara, mutante, poblacion)
        fuera = (vector_prueba < lim_inf) | (vector_prueba > lim_sup)
        vector_prueba = np.where(fuera, rng.uniform(lim_inf, lim_sup, vector_prueba.shape), vector_prueba)
        aptitud_prueba = np.array([obj(u) for u in vector_prueba])
        gana = aptitud_prueba <= aptitud
        poblacion[gana], aptitud[gana] = vector_prueba[gana], aptitud_prueba[gana]
        historial.append(float(aptitud.min()))
        traza_evals.append(obj.n_equiv)
        if record:
            frames.append(poblacion.copy())
    i = int(aptitud.argmin())
    return poblacion[i].copy(), float(aptitud[i]), historial, traza_evals, frames


# ---------------- Líneas base (opcionales) ----------------
@register_algorithm("random_search")
def random_search(obj, dim, bounds, rng, max_evals, record=False):
    lim_inf, lim_sup = bounds
    mejor_x = rng.uniform(lim_inf, lim_sup, dim)
    mejor_f = obj(mejor_x)
    historial, traza_evals = [mejor_f], [obj.n_equiv]
    frames = [mejor_x.copy()] if record else None
    while obj.n_equiv + 1 <= max_evals:
        candidato_x = rng.uniform(lim_inf, lim_sup, dim)
        f_candidato = obj(candidato_x)
        if f_candidato < mejor_f:
            mejor_x, mejor_f = candidato_x, f_candidato
        historial.append(mejor_f)
        traza_evals.append(obj.n_equiv)
        if record:
            frames.append(mejor_x.copy())
    return mejor_x, mejor_f, historial, traza_evals, frames


MAIN_ALGOS = ("gradient_descent", "genetic", "pso", "differential_evolution")


# ======================================================================
# 5. Punto de entrada
# ======================================================================
def optimize(algorithm="gradient_descent", problem="rosenbrock", dim=2,
             random_seed=42, max_evals=20000, record=False, plot=False,
             modo_2d: str = "superficie_3d",
             **algo_params) -> OptimizeResult:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algoritmo '{algorithm}' no registrado. Opciones: {sorted(ALGORITHMS)}")
    if problem not in PROBLEMS:
        raise ValueError(f"Problema '{problem}' no registrado. Opciones: {sorted(PROBLEMS)}")
    spec = PROBLEMS[problem]
    allowed = spec["supported_dims"]
    if allowed is not None and dim not in allowed:
        raise ValueError(f"'{problem}' admite dim en {allowed}, se recibió {dim}.")
    bad = set(algo_params) - accepted_params(algorithm)
    if bad:
        raise ValueError(f"Parámetros inválidos para '{algorithm}': {sorted(bad)}. "
                         f"Válidos: {sorted(accepted_params(algorithm))}")

    if plot:
        record = True

    rng = np.random.default_rng(random_seed)
    obj = CountedObjective(spec["func"], dim)
    x_best, f_best, hist, ev, frames = ALGORITHMS[algorithm](
        obj, dim, spec["bounds"], rng, max_evals, record, **algo_params)
    assert obj.raw_calls == obj.n_f + obj.cost_grad * obj.n_grad, "conteo inconsistente"
    res = OptimizeResult(problem, algorithm, dim, random_seed,
                         np.asarray(x_best, dtype=float), float(f_best),
                         obj.n_f, obj.n_grad, obj.n_equiv, hist, ev, frames)
    if plot:
        res.fig = graficar_optimizacion(res, mostrar=False, modo_2d=modo_2d)
        try:
            from IPython import get_ipython
            from IPython.display import display
            if get_ipython() is not None:
                display(res.fig)
        except Exception:
            pass
    return res


# ======================================================================
# 6. Estadísticas sobre varias corridas independientes
# ======================================================================
def run_batch(algorithm, problem, dim, n_runs, seed0, max_evals, threshold, params):
    res = [optimize(algorithm, problem, dim, seed0 + i, max_evals, **params)
           for i in range(n_runs)]
    f = np.array([r.f_best for r in res])
    ok = f < threshold
    to_success = []
    for r in res:
        h = np.asarray(r.history)
        if (h < threshold).any():
            to_success.append(r.eval_trace[int(np.argmax(h < threshold))])
    return {
        "algorithm": algorithm, "dim": dim, "runs": n_runs,
        "mean": f.mean(), "std": f.std(ddof=1) if n_runs > 1 else 0.0,
        "median": float(np.median(f)), "best": f.min(), "worst": f.max(),
        "success_rate": ok.mean(),
        "mean_n_f": np.mean([r.n_f for r in res]),
        "mean_n_grad": np.mean([r.n_grad for r in res]),
        "mean_n_equiv": np.mean([r.n_equiv for r in res]),
        "mean_evals_to_success": float(np.mean(to_success)) if to_success else float("nan"),
    }


# ======================================================================
# 7. Graficación Interactiva y Animada (Plotly 2D y 3D) y GIF
# ======================================================================
def graficar_optimizacion(res: OptimizeResult, max_frames: int = 60, mostrar: bool = False,
                          modo_2d: str = "superficie_3d"):
    """
    Genera una animación interactiva en Plotly del proceso de optimización
    mostrando la superficie real de la función objetivo tanto para dimensión 2 como para dimensión 3.

    Parámetros:
        res        : Objeto OptimizeResult resultante de optimize().
        max_frames : Número máximo de fotogramas en la animación.
        mostrar    : Si True, intenta desplegar la figura en el entorno.
        modo_2d    : 'superficie_3d' (default) muestra la superficie z = f(x₁, x₂) en 3D
                     con la trayectoria o partículas deslizándose sobre ella.
                     'contorno_2d' muestra el plano de contornos proyectado en 2D.
    """
    import plotly.graph_objects as go

    if res.frames is None or len(res.frames) == 0:
        raise ValueError("El resultado no contiene frames. Ejecute optimize(..., record=True o plot=True).")

    spec = PROBLEMS[res.problem]
    lim_inf, lim_sup = spec["bounds"]
    optimo = spec["known_optimum"](res.dim) if spec["known_optimum"] else None
    f_opt = spec["func"](optimo) if optimo is not None else 0.0
    n_raw = len(res.frames)
    sel = np.unique(np.linspace(0, n_raw - 1, min(n_raw, max_frames)).astype(int))

    es_trayectoria = np.ndim(res.frames[0]) == 1

    if res.dim == 2:
        if modo_2d == "superficie_3d":
            # Grilla para la superficie z = f(x1, x2)
            n_grid = 75
            g = np.linspace(lim_inf, lim_sup, n_grid)
            X, Y = np.meshgrid(g, g)
            if res.problem == "rosenbrock":
                Z = 100.0 * (Y - X ** 2) ** 2 + (1.0 - X) ** 2
            elif res.problem == "rastrigin":
                Z = 20.0 + (X ** 2 - 10.0 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10.0 * np.cos(2 * np.pi * Y))
            else:
                grid_pts = np.column_stack([X.ravel(), Y.ravel()])
                Z = np.array([spec["func"](p) for p in grid_pts]).reshape(X.shape)

            def f_eval_point(pt):
                if res.problem == "rosenbrock":
                    return 100.0 * (pt[1] - pt[0] ** 2) ** 2 + (1.0 - pt[0]) ** 2
                elif res.problem == "rastrigin":
                    return 20.0 + (pt[0] ** 2 - 10.0 * np.cos(2 * np.pi * pt[0])) + (pt[1] ** 2 - 10.0 * np.cos(2 * np.pi * pt[1]))
                return spec["func"](pt)

            def f_eval_population(pop):
                if res.problem == "rosenbrock":
                    return 100.0 * (pop[:, 1] - pop[:, 0] ** 2) ** 2 + (1.0 - pop[:, 0]) ** 2
                elif res.problem == "rastrigin":
                    return 20.0 + (pop[:, 0] ** 2 - 10.0 * np.cos(2 * np.pi * pop[:, 0])) + (pop[:, 1] ** 2 - 10.0 * np.cos(2 * np.pi * pop[:, 1]))
                return np.array([spec["func"](p) for p in pop])

            trace_surface = go.Surface(
                x=g, y=g, z=Z,
                colorscale="Viridis",
                opacity=0.82,
                showscale=True,
                colorbar=dict(title="f(x₁, x₂)", len=0.75),
                name="Superficie f(x)",
                hoverinfo="none"
            )

            z_opt = f_opt if f_opt is not None else 0.0
            trace_opt = go.Scatter3d(
                x=[optimo[0]] if optimo is not None else [],
                y=[optimo[1]] if optimo is not None else [],
                z=[z_opt] if optimo is not None else [],
                mode="markers",
                marker=dict(symbol="diamond", size=10, color="red", line=dict(color="black", width=2)),
                name=f"Óptimo ({optimo[0]:g}, {optimo[1]:g}, z={z_opt:g})" if optimo is not None else "Óptimo"
            )

            idx0 = sel[0]
            f0 = res.frames[idx0]
            if es_trayectoria:
                P_init = np.array([res.frames[i][:2] for i in sel[:1]])
                Z_init = np.array([f_eval_point(p) for p in P_init])
                trace_path = go.Scatter3d(
                    x=P_init[:, 0], y=P_init[:, 1], z=Z_init,
                    mode="lines+markers",
                    line=dict(color="cyan", width=5),
                    marker=dict(color="yellow", size=4),
                    name="Trayectoria"
                )
                z_act0 = f_eval_point(f0[:2])
                trace_actual = go.Scatter3d(
                    x=[f0[0]], y=[f0[1]], z=[z_act0],
                    mode="markers",
                    marker=dict(color="red", size=8, line=dict(color="white", width=2)),
                    name="Punto actual"
                )
                data_init = [trace_surface, trace_opt, trace_path, trace_actual]
            else:
                pop0 = np.asarray(f0)
                z_pop0 = f_eval_population(pop0)
                trace_pop = go.Scatter3d(
                    x=pop0[:, 0], y=pop0[:, 1], z=z_pop0,
                    mode="markers",
                    marker=dict(color="white", size=5, opacity=0.9, line=dict(color="black", width=1)),
                    name="Población"
                )
                data_init = [trace_surface, trace_opt, trace_pop]

            frames = []
            for step, k in enumerate(sel):
                evals_k = res.eval_trace[k] if k < len(res.eval_trace) else res.n_equiv
                fbest_k = res.history[k] if k < len(res.history) else res.f_best
                frame_title = f"{res.algorithm.upper()} | {res.problem.capitalize()} (Superficie 3D) | Evals: {evals_k} | f_best: {fbest_k:.2e}"

                if es_trayectoria:
                    P_k = np.array([res.frames[i][:2] for i in sel[:step + 1]])
                    Z_k = np.array([f_eval_point(p) for p in P_k])
                    pt_k = res.frames[k]
                    z_k = f_eval_point(pt_k[:2])
                    fr_data = [
                        trace_surface,
                        trace_opt,
                        go.Scatter3d(x=P_k[:, 0], y=P_k[:, 1], z=Z_k, mode="lines+markers",
                                     line=dict(color="cyan", width=5), marker=dict(color="yellow", size=4)),
                        go.Scatter3d(x=[pt_k[0]], y=[pt_k[1]], z=[z_k], mode="markers",
                                     marker=dict(color="red", size=8, line=dict(color="white", width=2)))
                    ]
                else:
                    pop_k = np.asarray(res.frames[k])
                    z_pop_k = f_eval_population(pop_k)
                    fr_data = [
                        trace_surface,
                        trace_opt,
                        go.Scatter3d(x=pop_k[:, 0], y=pop_k[:, 1], z=z_pop_k, mode="markers",
                                     marker=dict(color="white", size=5, opacity=0.9, line=dict(color="black", width=1)))
                    ]
                frames.append(go.Frame(data=fr_data, name=str(step), layout=go.Layout(title=frame_title)))

            slider_steps = [
                dict(
                    method="animate",
                    args=[[str(step)], dict(mode="immediate", frame=dict(duration=100, redraw=True), transition=dict(duration=0))],
                    label=f"{res.eval_trace[k] if k < len(res.eval_trace) else k}"
                )
                for step, k in enumerate(sel)
            ]

            updatemenus = [
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.05, y=-0.1,
                    buttons=[
                        dict(label="▶ Reproducir", method="animate",
                             args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True, mode="immediate")]),
                        dict(label="⏸ Pausar", method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                    ]
                )
            ]

            sliders = [dict(
                steps=slider_steps,
                active=0,
                x=0.25, y=-0.08, len=0.72,
                currentvalue=dict(font=dict(size=12), prefix="Evaluaciones: ", visible=True, xanchor="right")
            )]

            fig = go.Figure(
                data=data_init,
                layout=go.Layout(
                    title=f"{res.algorithm.upper()} | {res.problem.capitalize()} Superficie 3D | f_best = {res.f_best:.3e}",
                    scene=dict(
                        xaxis=dict(title="x₁", range=[lim_inf, lim_sup]),
                        yaxis=dict(title="x₂", range=[lim_inf, lim_sup]),
                        zaxis=dict(title="f(x₁, x₂)"),
                        camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2))
                    ),
                    updatemenus=updatemenus,
                    sliders=sliders,
                    template="plotly_dark",
                    width=850, height=650
                ),
                frames=frames
            )

        else:
            # Modo 2D Contorno clásico
            g = np.linspace(lim_inf, lim_sup, 120)
            X, Y = np.meshgrid(g, g)
            if res.problem == "rosenbrock":
                Z = 100.0 * (Y - X ** 2) ** 2 + (1.0 - X) ** 2
            elif res.problem == "rastrigin":
                Z = 20.0 + (X ** 2 - 10.0 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10.0 * np.cos(2 * np.pi * Y))
            else:
                grid_pts = np.column_stack([X.ravel(), Y.ravel()])
                Z = np.array([spec["func"](p) for p in grid_pts]).reshape(X.shape)

            trace_contour = go.Contour(
                x=g, y=g, z=Z,
                colorscale="Viridis",
                contours=dict(showlines=False),
                opacity=0.75,
                showscale=True,
                colorbar=dict(title="f(x)", len=0.8),
                hoverinfo="none",
                name="Superficie f(x)"
            )

            trace_opt = go.Scatter(
                x=[optimo[0]] if optimo is not None else [],
                y=[optimo[1]] if optimo is not None else [],
                mode="markers",
                marker=dict(symbol="star", size=15, color="red", line=dict(color="black", width=1.5)),
                name=f"Óptimo ({optimo[0]:g}, {optimo[1]:g})" if optimo is not None else "Óptimo",
                hoverinfo="name"
            )

            idx0 = sel[0]
            f0 = res.frames[idx0]
            if es_trayectoria:
                P_init = np.array([res.frames[i][:2] for i in sel[:1]])
                trace_path = go.Scatter(
                    x=P_init[:, 0], y=P_init[:, 1],
                    mode="lines+markers",
                    line=dict(color="white", width=2),
                    marker=dict(color="cyan", size=6),
                    name="Trayectoria"
                )
                trace_actual = go.Scatter(
                    x=[f0[0]], y=[f0[1]],
                    mode="markers",
                    marker=dict(color="yellow", size=12, line=dict(color="black", width=2)),
                    name="Punto actual"
                )
                data_init = [trace_contour, trace_opt, trace_path, trace_actual]
            else:
                pop0 = np.asarray(f0)
                trace_pop = go.Scatter(
                    x=pop0[:, 0], y=pop0[:, 1],
                    mode="markers",
                    marker=dict(color="white", size=8, line=dict(color="black", width=1)),
                    name="Población"
                )
                data_init = [trace_contour, trace_opt, trace_pop]

            frames = []
            for step, k in enumerate(sel):
                evals_k = res.eval_trace[k] if k < len(res.eval_trace) else res.n_equiv
                fbest_k = res.history[k] if k < len(res.history) else res.f_best
                frame_title = f"{res.algorithm.upper()} | {res.problem.capitalize()} 2D | Evals: {evals_k} | f_best: {fbest_k:.2e}"

                if es_trayectoria:
                    P_k = np.array([res.frames[i][:2] for i in sel[:step + 1]])
                    pt_k = res.frames[k]
                    fr_data = [
                        trace_contour,
                        trace_opt,
                        go.Scatter(x=P_k[:, 0], y=P_k[:, 1], mode="lines+markers",
                                   line=dict(color="white", width=2), marker=dict(color="cyan", size=6)),
                        go.Scatter(x=[pt_k[0]], y=[pt_k[1]], mode="markers",
                                   marker=dict(color="yellow", size=12, line=dict(color="black", width=2)))
                    ]
                else:
                    pop_k = np.asarray(res.frames[k])
                    fr_data = [
                        trace_contour,
                        trace_opt,
                        go.Scatter(x=pop_k[:, 0], y=pop_k[:, 1], mode="markers",
                                   marker=dict(color="white", size=8, line=dict(color="black", width=1)))
                    ]
                frames.append(go.Frame(data=fr_data, name=str(step), layout=go.Layout(title=frame_title)))

            slider_steps = [
                dict(
                    method="animate",
                    args=[[str(step)], dict(mode="immediate", frame=dict(duration=100, redraw=True), transition=dict(duration=0))],
                    label=f"{res.eval_trace[k] if k < len(res.eval_trace) else k}"
                )
                for step, k in enumerate(sel)
            ]

            updatemenus = [
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.05, y=-0.18,
                    buttons=[
                        dict(label="▶ Reproducir", method="animate",
                             args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True, mode="immediate")]),
                        dict(label="⏸ Pausar", method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                    ]
                )
            ]

            sliders = [dict(
                steps=slider_steps,
                active=0,
                x=0.25, y=-0.15, len=0.72,
                currentvalue=dict(font=dict(size=12), prefix="Evaluaciones: ", visible=True, xanchor="right")
            )]

            fig = go.Figure(
                data=data_init,
                layout=go.Layout(
                    title=f"{res.algorithm.upper()} | {res.problem.capitalize()} 2D | f_best = {res.f_best:.3e}",
                    xaxis=dict(title="x₁", range=[lim_inf, lim_sup], zeroline=False),
                    yaxis=dict(title="x₂", range=[lim_inf, lim_sup], zeroline=False),
                    updatemenus=updatemenus,
                    sliders=sliders,
                    template="plotly_dark",
                    width=800, height=640
                ),
                frames=frames
            )

    elif res.dim == 3:
        # Grilla 3D para isosuperficies de nivel de f(x₁, x₂, x₃)
        n_grid3 = 20
        g3 = np.linspace(lim_inf, lim_sup, n_grid3)
        X3, Y3, Z3 = np.meshgrid(g3, g3, g3, indexing="ij")
        grid_pts3 = np.column_stack([X3.ravel(), Y3.ravel(), Z3.ravel()])
        func3 = spec["func"]
        vals3 = np.array([func3(p) for p in grid_pts3])

        v_min_iso = float(np.percentile(vals3, 5))
        v_max_iso = float(np.percentile(vals3, 75))

        trace_iso = go.Isosurface(
            x=X3.flatten(),
            y=Y3.flatten(),
            z=Z3.flatten(),
            value=vals3.flatten(),
            isomin=v_min_iso,
            isomax=v_max_iso,
            surface_count=3,
            opacity=0.22,
            colorscale="Viridis",
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=True,
            colorbar=dict(title="f(x) nivel", len=0.75),
            name="Superficie f(x)"
        )

        trace_opt = go.Scatter3d(
            x=[optimo[0]] if optimo is not None else [],
            y=[optimo[1]] if optimo is not None else [],
            z=[optimo[2]] if optimo is not None else [],
            mode="markers",
            marker=dict(symbol="diamond", size=10, color="red", line=dict(color="black", width=2)),
            name=f"Óptimo ({optimo[0]:g}, {optimo[1]:g}, {optimo[2]:g})" if optimo is not None else "Óptimo"
        )

        idx0 = sel[0]
        f0 = res.frames[idx0]
        if es_trayectoria:
            P_init = np.array([res.frames[i] for i in sel[:1]])
            trace_path = go.Scatter3d(
                x=P_init[:, 0], y=P_init[:, 1], z=P_init[:, 2],
                mode="lines+markers",
                line=dict(color="cyan", width=5),
                marker=dict(color="yellow", size=4),
                name="Trayectoria 3D"
            )
            trace_actual = go.Scatter3d(
                x=[f0[0]], y=[f0[1]], z=[f0[2]],
                mode="markers",
                marker=dict(color="red", size=8, line=dict(color="white", width=2)),
                name="Punto actual"
            )
            data_init = [trace_iso, trace_opt, trace_path, trace_actual]
        else:
            pop0 = np.asarray(f0)
            trace_pop = go.Scatter3d(
                x=pop0[:, 0], y=pop0[:, 1], z=pop0[:, 2],
                mode="markers",
                marker=dict(color="white", size=5, opacity=0.9, line=dict(color="black", width=1)),
                name="Población 3D"
            )
            data_init = [trace_iso, trace_opt, trace_pop]

        frames = []
        for step, k in enumerate(sel):
            evals_k = res.eval_trace[k] if k < len(res.eval_trace) else res.n_equiv
            fbest_k = res.history[k] if k < len(res.history) else res.f_best
            frame_title = f"{res.algorithm.upper()} | {res.problem.capitalize()} 3D | Evals: {evals_k} | f_best: {fbest_k:.2e}"

            if es_trayectoria:
                P_k = np.array([res.frames[i] for i in sel[:step + 1]])
                pt_k = res.frames[k]
                fr_data = [
                    trace_iso,
                    trace_opt,
                    go.Scatter3d(x=P_k[:, 0], y=P_k[:, 1], z=P_k[:, 2], mode="lines+markers",
                                 line=dict(color="cyan", width=5), marker=dict(color="yellow", size=4)),
                    go.Scatter3d(x=[pt_k[0]], y=[pt_k[1]], z=[pt_k[2]], mode="markers",
                                 marker=dict(color="red", size=8, line=dict(color="white", width=2)))
                ]
            else:
                pop_k = np.asarray(res.frames[k])
                fr_data = [
                    trace_iso,
                    trace_opt,
                    go.Scatter3d(x=pop_k[:, 0], y=pop_k[:, 1], z=pop_k[:, 2], mode="markers",
                                 marker=dict(color="white", size=5, opacity=0.9, line=dict(color="black", width=1)))
                ]
            frames.append(go.Frame(data=fr_data, name=str(step), layout=go.Layout(title=frame_title)))

        slider_steps = [
            dict(
                method="animate",
                args=[[str(step)], dict(mode="immediate", frame=dict(duration=100, redraw=True), transition=dict(duration=0))],
                label=f"{res.eval_trace[k] if k < len(res.eval_trace) else k}"
            )
            for step, k in enumerate(sel)
        ]

        updatemenus = [
            dict(
                type="buttons",
                showactive=False,
                x=0.05, y=-0.1,
                buttons=[
                    dict(label="▶ Reproducir", method="animate",
                         args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True, mode="immediate")]),
                    dict(label="⏸ Pausar", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ]
            )
        ]

        sliders = [dict(
            steps=slider_steps,
            active=0,
            x=0.25, y=-0.08, len=0.72,
            currentvalue=dict(font=dict(size=12), prefix="Evaluaciones: ", visible=True, xanchor="right")
        )]

        fig = go.Figure(
            data=data_init,
            layout=go.Layout(
                title=f"{res.algorithm.upper()} | {res.problem.capitalize()} 3D (Superficie de nivel) | f_best = {res.f_best:.3e}",
                scene=dict(
                    xaxis=dict(title="x₁", range=[lim_inf, lim_sup]),
                    yaxis=dict(title="x₂", range=[lim_inf, lim_sup]),
                    zaxis=dict(title="x₃", range=[lim_inf, lim_sup]),
                    camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2))
                ),
                updatemenus=updatemenus,
                sliders=sliders,
                template="plotly_dark",
                width=850, height=650
            ),
            frames=frames
        )
    else:
        raise ValueError(f"La visualización animada está implementada para 2D y 3D (se recibió dim={res.dim}).")

    fig.add_annotation(text='Fuente: elaboración propia; punto_1/funcion1.py.',
                       xref='paper', yref='paper', x=0, y=-0.2, showarrow=False)
    fig.update_layout(margin=dict(b=140))
    if mostrar:
        try:
            from IPython import get_ipython
            from IPython.display import display
            if get_ipython() is not None:
                display(fig)
            else:
                fig.show()
        except Exception:
            pass
    return fig


def make_gif(algorithm, out, dim=2, seed=42, max_evals=20000, params=None,
             max_frames=80, fps=10, problem="rosenbrock"):
    if dim != 2:
        raise ValueError("El GIF solo está implementado para dim=2.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    res = optimize(algorithm, problem, dim, seed, max_evals, record=True, **(params or {}))
    if res.frames is None:
        raise ValueError(f"'{algorithm}' no guarda trayectoria; usa otro algoritmo.")
    lo, hi = PROBLEMS[problem]["bounds"]
    func = PROBLEMS[problem]["func"]
    known_opt = PROBLEMS[problem]["known_optimum"](2) if PROBLEMS[problem]["known_optimum"] else None
    g = np.linspace(lo, hi, 300)
    X, Y = np.meshgrid(g, g)
    if problem == "rosenbrock":
        Z = 100.0 * (Y - X ** 2) ** 2 + (1.0 - X) ** 2
        levels = np.logspace(-2, 3.5, 25)
    elif problem == "rastrigin":
        Z = 20.0 + (X ** 2 - 10.0 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10.0 * np.cos(2 * np.pi * Y))
        levels = np.linspace(0, 80, 25)
    else:
        grid_pts = np.column_stack([X.ravel(), Y.ravel()])
        Z = np.array([func(p) for p in grid_pts]).reshape(X.shape)
        levels = 25

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.contourf(X, Y, Z, levels=levels, extend="both", cmap="viridis")
    if known_opt is not None:
        ax.plot(known_opt[0], known_opt[1], "r*", ms=14, label=f"óptimo ({known_opt[0]:g},{known_opt[1]:g})")
    is_path = np.ndim(res.frames[0]) == 1
    if is_path:
        line, = ax.plot([], [], "w.-", lw=1, ms=3)
        pt, = ax.plot([], [], "wo", mec="k")
    else:
        sc = ax.scatter([], [], c="white", edgecolors="k", s=25)
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
    ax.set_xlabel("$x_1$"), ax.set_ylabel("$x_2$")
    ax.legend(loc="upper left", fontsize=8)
    title = ax.set_title("")

    sel = np.unique(np.linspace(0, len(res.frames) - 1, max_frames).astype(int))

    def update(k):
        i = sel[k]
        if is_path:
            P = np.array(res.frames[: i + 1])
            line.set_data(P[:, 0], P[:, 1])
            pt.set_data([P[-1, 0]], [P[-1, 1]])
        else:
            sc.set_offsets(res.frames[i][:, :2])
        title.set_text(f"{algorithm} | evals={res.eval_trace[i]} | f_best={res.history[i]:.2e}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, update, frames=len(sel)).save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return res


# ======================================================================
# 8. Verificaciones
# ======================================================================
def run_checks() -> bool:
    ok = True
    spec = PROBLEMS["rosenbrock"]
    rng = np.random.default_rng(0)
    for d in (2, 3):
        f_opt = spec["func"](spec["known_optimum"](d))
        print(f"[{'OK' if f_opt == 0 else 'FALLA'}] f(1,...,1) = {f_opt} en {d}D")
        ok &= f_opt == 0
        obj = CountedObjective(spec["func"], d)
        errs = []
        for _ in range(100):
            x = rng.uniform(*spec["bounds"], d)
            errs.append(np.max(np.abs(obj.gradient(x) - spec["grad"](x))))
        e = max(errs)
        print(f"[{'OK' if e < 1e-3 else 'FALLA'}] gradiente numérico vs analítico, {d}D: "
              f"error máx = {e:.2e}")
        ok &= e < 1e-3
    for a in MAIN_ALGOS:
        r = optimize(a, "rosenbrock", 2, 1, 5000)
        c = r.n_equiv <= 5000
        print(f"[{'OK' if c else 'FALLA'}] {a}: equivalentes={r.n_equiv} <= 5000 "
              f"(n_f={r.n_f}, n_grad={r.n_grad})")
        ok &= c
    r1 = optimize("pso", random_seed=7)
    r2 = optimize("pso", random_seed=7)
    same = r1.f_best == r2.f_best
    print(f"[{'OK' if same else 'FALLA'}] misma semilla => mismo resultado")
    ok &= same
    return bool(ok)


# ======================================================================
# 9. CLI
# ======================================================================
def _val(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def parse_overrides(raw, selected):
    """--param k=v aplica a los algoritmos seleccionados que acepten k;
    --param algo.k=v aplica solo a ese algoritmo."""
    out = {a: {} for a in ALGORITHMS}
    for item in raw or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise SystemExit(f"--param debe tener la forma clave=valor, se recibió '{item}'")
        if "." in key:
            a, k = key.split(".", 1)
            if a not in ALGORITHMS or k not in accepted_params(a):
                raise SystemExit(f"Parámetro desconocido: {key}")
            out[a][k] = _val(val)
        else:
            targets = [a for a in selected if key in accepted_params(a)]
            if not targets:
                raise SystemExit(f"Ningún algoritmo seleccionado acepta '{key}'")
            for a in targets:
                out[a][key] = _val(val)
    return out


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--problem", choices=sorted(PROBLEMS), default="rosenbrock",
                        help="función de prueba (por defecto: rosenbrock)")
    common.add_argument("--config", help="archivo JSON con valores por defecto")
    common.add_argument("--max-evals", dest="max_evals", type=int)
    common.add_argument('--x0', nargs='+', type=float,
                        help='condición inicial de GD, ejemplo --x0 -1.2 1')
    common.add_argument("--param", action="append", metavar="[algo.]clave=valor")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="verificaciones rápidas de matemática y conteo")

    r = sub.add_parser("run", parents=[common], help="una corrida")
    r.add_argument("--algo", choices=sorted(ALGORITHMS))
    r.add_argument("--dim", type=int)
    r.add_argument("--seed", type=int)

    s = sub.add_parser("stats", parents=[common], help="estadísticas de N corridas")
    s.add_argument("--algos", nargs="+", help="nombres o 'all' (incluye líneas base)")
    s.add_argument("--dims", nargs="+", type=int)
    s.add_argument("--runs", type=int)
    s.add_argument("--seed0", type=int)
    s.add_argument("--threshold", type=float)
    s.add_argument("--out")

    g = sub.add_parser("gif", parents=[common], help="GIF (dim=2)")
    g.add_argument("--algo", choices=sorted(ALGORITHMS))
    g.add_argument("--seed", type=int)
    g.add_argument("--out")
    g.add_argument("--fps", type=int)
    g.add_argument("--max-frames", dest="max_frames", type=int)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "check":
        sys.exit(0 if run_checks() else 1)

    cfg = json.loads(Path(args.config).read_text()) if args.config else {}

    def pick(name, default):
        v = getattr(args, name, None)
        return v if v is not None else cfg.get(name, default)

    max_evals = pick("max_evals", 20000)
    problem = pick("problem", "rosenbrock")

    def params_for(selected):
        base = {a: dict(cfg.get("params", {}).get(a, {})) for a in ALGORITHMS}
        for a, d in parse_overrides(args.param, selected).items():
            base[a].update(d)
        punto = getattr(args, 'x0', None)
        if punto is None:
            punto = base['gradient_descent'].get('x0', cfg.get('x0'))
        if punto is not None:
            if 'gradient_descent' not in selected:
                raise SystemExit('--x0 solo se aplica a gradient_descent')
            base['gradient_descent']['x0'] = punto
        return base

    if args.cmd == "run":
        algo, dim, seed = pick("algo", "gradient_descent"), pick("dim", 2), pick("seed", 42)
        print(optimize(algo, problem, dim, seed, max_evals, **params_for([algo])[algo]))

    elif args.cmd == "gif":
        algo, seed = pick("algo", "gradient_descent"), pick("seed", 42)
        out = pick("out", f"figs/{algo}_{problem}_2d.gif")
        res = make_gif(algo, out, 2, seed, max_evals, params_for([algo])[algo],
                       pick("max_frames", 80), pick("fps", 10), problem=problem)
        print(f"GIF guardado en {out}\n{res}")

    elif args.cmd == "stats":
        algos = pick("algos", list(MAIN_ALGOS))
        if algos == ["all"]:
            algos = sorted(ALGORITHMS)
        dims = pick("dims", [2, 3])
        n_runs, seed0 = pick("runs", 30), pick("seed0", 0)
        thr, out = pick("threshold", 1e-4), pick("out", "results")
        P = params_for(algos)
        rows, t0 = [], time.time()
        print(f"{problem.capitalize()} | corridas={n_runs} | presupuesto={max_evals} evals equivalentes "
              f"| éxito: f < {thr:g} | semillas {seed0}..{seed0 + n_runs - 1}")
        for d in dims:
            for a in algos:
                rows.append(run_batch(a, problem, d, n_runs, seed0, max_evals, thr, P[a]))
        hdr = (f"{'algoritmo':24s}{'dim':>4s}{'media':>11s}{'std':>11s}{'mejor':>11s}"
               f"{'peor':>11s}{'éxito':>7s}{'n_f':>9s}{'n_grad':>8s}{'equiv':>9s}")
        print(hdr)
        for r in rows:
            print(f"{r['algorithm']:24s}{r['dim']:>4d}{r['mean']:>11.2e}{r['std']:>11.2e}"
                  f"{r['best']:>11.2e}{r['worst']:>11.2e}{r['success_rate']:>7.0%}"
                  f"{r['mean_n_f']:>9.0f}{r['mean_n_grad']:>8.0f}{r['mean_n_equiv']:>9.0f}")
        Path(out).mkdir(parents=True, exist_ok=True)
        path = Path(out) / "stats_rosenbrock.csv"
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {path}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()