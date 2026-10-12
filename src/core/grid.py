"""
- La matriz es un arreglo NumPy 2D.
- 1 = célula viva, 0 = célula muerta.
- Fuera de los límites de la matriz no hay nada (se considera muerto).
"""

import numpy as np

DTYPE = np.uint8


def empty_grid(rows: int, cols: int) -> np.ndarray:
    """Matriz con todas las células muertas."""
    return np.zeros((rows, cols), dtype=DTYPE)


def random_grid(rows: int, cols: int, density: float = 0.3, seed: int | None = None) -> np.ndarray:
    """Matriz aleatoria. Con la misma semilla siempre sale la misma matriz,
    así secuencial y paralelo parten exactamente del mismo estado inicial."""
    rng = np.random.default_rng(seed)
    return (rng.random((rows, cols)) < density).astype(DTYPE)


def from_pattern(pattern: list[str], rows: int, cols: int, top: int = 0, left: int = 0) -> np.ndarray:
    """Crea una matriz vacía y dibuja un patrón. Usa 'X' para viva y '.' para muerta.

    Ejemplo:
        from_pattern([".X.", "..X", "XXX"], 10, 10, top=1, left=1)  # glider
    """
    grid = empty_grid(rows, cols)
    for i, line in enumerate(pattern):
        for j, ch in enumerate(line):
            if ch == "X":
                grid[top + i, left + j] = 1
    return grid