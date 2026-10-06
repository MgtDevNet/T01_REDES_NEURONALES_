
#Marco modular de optimización.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)


# 1. Resultado

@dataclass
class OptimizeResult:
    problem: str
    algorithm: str
    dim: int
    random_seed: int | None
    x_best: np.ndarray
    f_best: float
    n_evals: int
    history: list[float] = field(default_factory=list)  # mejor f por iteración

    def __str__(self) -> str:
        return (
            f"{self.algorithm} | {self.problem} {self.dim}D | seed={self.random_seed}\n"
            f"  x_best  = {np.array2string(self.x_best, precision=6)}\n"
            f"  f_best  = {self.f_best:.6e}\n"
            f"  n_evals = {self.n_evals}"
        )


# ======================================================================
# 2. Registro de problemas (funciones de prueba)
# ======================================================================
PROBLEMS: dict[str, dict] = {}


def register_problem(
    name: str,
    func: Callable[[np.ndarray], float],
    bounds: tuple[float, float],
    supported_dims: tuple[int, ...] | None = None,
    known_optimum: Callable[[int], np.ndarray] | None = None,
) -> None:
    """Registra una función objetivo. supported_dims=None => cualquier dimensión."""
    PROBLEMS[name] = {
        "func": func,
        "bounds": bounds,
        "supported_dims": supported_dims,
        "known_optimum": known_optimum,
    }


register_problem(
    "rosenbrock",
    rosenbrock,
    bounds=(-2.048, 2.048),
    supported_dims=(2, 3),
    known_optimum=lambda d: np.ones(d),
)
# Para agregar otra función de prueba:
# register_problem("sphere", sphere, bounds=(-5.12, 5.12))


# ======================================================================
# 3. Registro de algoritmos
# ======================================================================
# Firma común de todo algoritmo:
#   algo(func, dim, bounds, rng, max_iter, **params) -> (x_best, f_best, history)
ALGORITHMS: dict[str, Callable] = {}


def register_algorithm(name: str):
    def decorator(fn: Callable) -> Callable:
        ALGORITHMS[name] = fn
        return fn
    return decorator


class _CountedObjective:
    """Envuelve la función objetivo para contar evaluaciones."""

    def __init__(self, func: Callable):
        self.func = func
        self.n_evals = 0

    def __call__(self, x: np.ndarray) -> float:
        self.n_evals += 1
        return float(self.func(x))


# ----------------------------------------------------------------------
# Algoritmo 1: búsqueda aleatoria
# ----------------------------------------------------------------------
@register_algorithm("random_search")
def random_search(func, dim, bounds, rng, max_iter=5000):
    lo, hi = bounds
    x_best = rng.uniform(lo, hi, dim)
    f_best = func(x_best)
    history = [f_best]

    for _ in range(max_iter - 1):
        x = rng.uniform(lo, hi, dim)
        fx = func(x)
        if fx < f_best:
            x_best, f_best = x, fx
        history.append(f_best)
    return x_best, f_best, history


# ----------------------------------------------------------------------
# Algoritmo 2: recocido simulado
# ----------------------------------------------------------------------
@register_algorithm("simulated_annealing")
def simulated_annealing(func, dim, bounds, rng, max_iter=20000,
                        T0=10.0, alpha=0.9995, step=0.1):
    lo, hi = bounds
    x = rng.uniform(lo, hi, dim)
    fx = func(x)
    x_best, f_best = x.copy(), fx
    history = [f_best]
    T = T0

    for _ in range(max_iter - 1):
        cand = np.clip(x + rng.normal(0.0, step, dim), lo, hi)
        fc = func(cand)
        df = fc - fx
        if df < 0 or rng.random() < np.exp(-df / max(T, 1e-12)):
            x, fx = cand, fc
            if fx < f_best:
                x_best, f_best = x.copy(), fx
        T *= alpha
        history.append(f_best)
    return x_best, f_best, history


# ----------------------------------------------------------------------
# Algoritmo 3: descenso por gradiente (gradiente numérico + backtracking)
# ----------------------------------------------------------------------
@register_algorithm("gradient_descent")
def gradient_descent(func, dim, bounds, rng, max_iter=5000,
                     tol=1e-8, h=1e-6, lr0=1.0, c=1e-4, shrink=0.5):
    lo, hi = bounds
    x = rng.uniform(lo, hi, dim)       # el único elemento aleatorio: el punto inicial
    fx = func(x)
    history = [fx]

    def num_grad(p):
        g = np.zeros(dim)
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = h
            g[i] = (func(p + e) - func(p - e)) / (2 * h)
        return g

    for _ in range(max_iter - 1):
        g = num_grad(x)
        gnorm2 = float(g @ g)
        if np.sqrt(gnorm2) < tol:
            break
        lr = lr0
        for _ in range(60):            # búsqueda lineal de Armijo
            cand = x - lr * g
            fc = func(cand)
            if fc <= fx - c * lr * gnorm2:
                break
            lr *= shrink
        else:
            break                      # no se logró mejora
        x, fx = cand, fc
        history.append(fx)
    return x, fx, history


# ======================================================================
# 4. Punto de entrada
# ======================================================================
def optimize(
    algorithm: str = "random_search",
    problem: str = "rosenbrock",
    dim: int = 2,
    random_seed: int | None = 42,
    max_iter: int | None = None,
    **algo_params,
) -> OptimizeResult:
    """
    Ejecuta `algorithm` sobre `problem` en `dim` dimensiones.

    random_seed : semilla del generador. Misma semilla => mismos resultados.
    max_iter    : iteraciones (si None, se usa el valor por defecto del algoritmo).
    algo_params : hiperparámetros propios del algoritmo (p. ej. T0=5, step=0.05).
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algoritmo '{algorithm}' no registrado. Opciones: {sorted(ALGORITHMS)}")
    if problem not in PROBLEMS:
        raise ValueError(f"Problema '{problem}' no registrado. Opciones: {sorted(PROBLEMS)}")

    spec = PROBLEMS[problem]
    allowed = spec["supported_dims"]
    if allowed is not None and dim not in allowed:
        raise ValueError(f"'{problem}' admite dim en {allowed}, se recibió {dim}.")

    rng = np.random.default_rng(random_seed)
    counted = _CountedObjective(spec["func"])

    if max_iter is not None:
        algo_params["max_iter"] = max_iter

    x_best, f_best, history = ALGORITHMS[algorithm](
        counted, dim, spec["bounds"], rng, **algo_params
    )

    return OptimizeResult(
        problem=problem,
        algorithm=algorithm,
        dim=dim,
        random_seed=random_seed,
        x_best=np.asarray(x_best, dtype=float),
        f_best=float(f_best),
        n_evals=counted.n_evals,
        history=history,
    )


# ======================================================================
# 5. Ejemplo de uso
# ======================================================================
if __name__ == "__main__":
    RANDOM_SEED = 42   # <- cambia aquí la semilla

    for algo in ALGORITHMS:
        for d in (2, 3):
            print(optimize(algo, "rosenbrock", dim=d, random_seed=RANDOM_SEED))
            print()