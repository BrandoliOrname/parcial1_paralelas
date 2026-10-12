"""Motor secuencial del Juego de la Vida de Conway.

Dos implementaciones con las mismas reglas:

1. next_generation_loops  -> bucles anidados (base del speedup)
2. next_generation_numpy  -> vectorizada con NumPy (comparación extra)

Reglas (con bordes muertos: fuera de la matriz no hay nada):
- Viva con < 2 vecinos vivos: muere (soledad).
- Viva con 2 o 3 vecinos vivos: sobrevive.
- Viva con > 3 vecinos vivos: muere (sobrepoblación).
- Muerta con exactamente 3 vecinos vivos: nace.
"""

from typing import Callable

import numpy as np

from .grid import DTYPE


def next_generation_loops(grid: np.ndarray) -> np.ndarray:
    """Calcula la siguiente generación con bucles anidados (Python puro).

    Se convierte a listas de Python para el cálculo porque indexar un arreglo
    NumPy elemento a elemento es mucho más lento que indexar una lista.
    Lee siempre de `old` y escribe en `new`: así todas las células se calculan
    a partir de la misma generación anterior.
    """
    rows, cols = grid.shape
    old = grid.tolist()
    new = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            alive_neighbors = 0
            for di in (-1, 0, 1):
                ni = i + di
                if ni < 0 or ni >= rows:
                    continue  # fuera de la matriz: no hay nada
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue  # no contarse a sí misma
                    nj = j + dj
                    if 0 <= nj < cols:
                        alive_neighbors += old[ni][nj]

            if old[i][j] == 1:
                new[i][j] = 1 if alive_neighbors in (2, 3) else 0
            else:
                new[i][j] = 1 if alive_neighbors == 3 else 0

    return np.array(new, dtype=DTYPE)


def next_generation_numpy(grid: np.ndarray) -> np.ndarray:
    """Calcula la siguiente generación de forma vectorizada con NumPy.

    Idea: rodear la matriz con un anillo de ceros (bordes muertos) y sumar las
    8 versiones desplazadas. Así se obtiene, para todas las células a la vez,
    el número de vecinos vivos, sin ningún bucle en Python.
    """
    rows, cols = grid.shape
    padded = np.pad(grid, 1, mode="constant", constant_values=0)

    neighbors = (
        padded[0:rows, 0:cols] + padded[0:rows, 1:cols + 1] + padded[0:rows, 2:cols + 2]
        + padded[1:rows + 1, 0:cols] + padded[1:rows + 1, 2:cols + 2]
        + padded[2:rows + 2, 0:cols] + padded[2:rows + 2, 1:cols + 1] + padded[2:rows + 2, 2:cols + 2]
    )

    survives = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    born = (grid == 0) & (neighbors == 3)
    return (survives | born).astype(DTYPE)


# Versión que se usa como base para medir el speedup.
next_generation = next_generation_loops


def simulate(
    initial_grid: np.ndarray,
    generations: int,
    step: Callable[[np.ndarray], np.ndarray] = next_generation_loops,
) -> np.ndarray:
    """Contrato común de los motores: recibe la matriz inicial y el número de
    generaciones, y devuelve la matriz final. No modifica la matriz original."""
    grid = initial_grid.copy()
    for _ in range(generations):
        grid = step(grid)
    return grid


def simulate_loops(initial_grid: np.ndarray, generations: int) -> np.ndarray:
    return simulate(initial_grid, generations, next_generation_loops)


def simulate_numpy(initial_grid: np.ndarray, generations: int) -> np.ndarray:
    return simulate(initial_grid, generations, next_generation_numpy)