import numpy as np
import pytest

from src.core.grid import random_grid
from src.core.mpi_bridge import MPIWorkerManager


def test_mpi_manager_availability():
    """Comprueba la detección de MPI en el sistema."""
    available = MPIWorkerManager.is_mpi_available()
    assert isinstance(available, bool)


def test_mpi_manager_lifecycle():
    """Valida el ciclo de vida del clúster MPI: inicio, paso, sincronización y detención."""
    if not MPIWorkerManager.is_mpi_available():
        pytest.skip("mpirun no disponible en el entorno")

    mgr = MPIWorkerManager()
    grid = random_grid(20, 20, seed=42)

    # 1. Iniciar con 2 procesos
    started = mgr.start(2, grid)
    assert started is True
    assert mgr.is_running() is True
    assert mgr.num_processes == 2

    # 2. Ejecutar un paso paralelo
    new_grid, t_calc = mgr.step()
    assert new_grid is not None
    assert new_grid.shape == (20, 20)
    assert t_calc >= 0.0

    # 3. Sincronizar nueva matriz
    updated_grid = np.zeros((20, 20), dtype=np.uint8)
    mgr.set_grid(updated_grid)
    next_grid, _ = mgr.step()
    assert next_grid is not None
    assert np.all(next_grid == 0)

    # 4. Detener clúster
    mgr.stop()
    assert mgr.is_running() is False
    assert mgr.proc is None
