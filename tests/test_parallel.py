import numpy as np
import pytest

from src.core.grid import random_grid, empty_grid
try:
    from src.core.paralelo import split_matrix
except RuntimeError:
    pytest.skip("MPI library no instalada en el sistema", allow_module_level=True)


def test_split_matrix_exact_division():
    """Comprueba que la división de filas sea exacta cuando no hay residuo."""
    grid = random_grid(100, 50, seed=42)
    parts = split_matrix(grid, 4)

    assert len(parts) == 4
    for part in parts:
        assert part.shape == (25, 50)

    # Reconstrucción idéntica
    reconstructed = np.vstack(parts)
    assert np.array_equal(grid, reconstructed)


def test_split_matrix_with_remainder():
    """Comprueba el reparto balanceado cuando el número de filas no es divisible."""
    grid = random_grid(10, 10, seed=123)
    parts = split_matrix(grid, 3)

    assert len(parts) == 3
    # 10 // 3 = 3 con residuo 1 -> tamaños deben ser 4, 3, 3
    assert parts[0].shape == (4, 10)
    assert parts[1].shape == (3, 10)
    assert parts[2].shape == (3, 10)

    reconstructed = np.vstack(parts)
    assert np.array_equal(grid, reconstructed)


def test_split_matrix_single_process():
    """Con 1 proceso, debe devolver la matriz original intacta."""
    grid = random_grid(30, 30, seed=99)
    parts = split_matrix(grid, 1)

    assert len(parts) == 1
    assert np.array_equal(parts[0], grid)


def test_split_matrix_empty_grid():
    """Verifica el comportamiento con matrices vacías."""
    grid = empty_grid(40, 20)
    parts = split_matrix(grid, 4)

    assert len(parts) == 4
    for part in parts:
        assert part.shape == (10, 20)
        assert part.sum() == 0

