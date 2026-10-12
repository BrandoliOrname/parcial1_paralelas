import numpy as np
import pytest

from src.core.grid import empty_grid, from_pattern, random_grid
from src.core.secuencial import (
    next_generation_loops,
    next_generation_numpy,
    simulate_loops,
    simulate_numpy,
)

ENGINES = [next_generation_loops, next_generation_numpy]
SIMULATORS = [simulate_loops, simulate_numpy]


# ---------- Reglas individuales ----------

@pytest.mark.parametrize("step", ENGINES)
def test_matriz_vacia_sigue_vacia(step):
    g = empty_grid(6, 6)
    assert np.array_equal(step(g), g)


@pytest.mark.parametrize("step", ENGINES)
def test_celula_sola_muere_por_soledad(step):
    g = from_pattern(["X"], 5, 5, top=2, left=2)
    assert step(g).sum() == 0


@pytest.mark.parametrize("step", ENGINES)
def test_celula_con_dos_vecinos_sobrevive(step):
    # Tres en fila: la del centro tiene 2 vecinos y sobrevive.
    g = from_pattern(["XXX"], 5, 5, top=2, left=1)
    assert step(g)[2, 2] == 1


@pytest.mark.parametrize("step", ENGINES)
def test_celula_con_mas_de_tres_vecinos_muere(step):
    # El centro rodeado de 4 vecinos vivos muere por sobrepoblación.
    g = from_pattern([".X.", "XXX", ".X."], 5, 5, top=1, left=1)
    assert step(g)[2, 2] == 0


@pytest.mark.parametrize("step", ENGINES)
def test_celula_muerta_con_tres_vecinos_nace(step):
    g = from_pattern(["XX", "X."], 5, 5, top=1, left=1)
    assert step(g)[2, 2] == 1


# ---------- Patrones conocidos ----------

@pytest.mark.parametrize("step", ENGINES)
def test_block_es_estable(step):
    g = from_pattern(["XX", "XX"], 6, 6, top=2, left=2)
    assert np.array_equal(step(g), g)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_blinker_oscila_con_periodo_2(sim):
    horizontal = from_pattern(["XXX"], 5, 5, top=2, left=1)
    vertical = from_pattern(["X", "X", "X"], 5, 5, top=1, left=2)
    assert np.array_equal(sim(horizontal, 1), vertical)
    assert np.array_equal(sim(horizontal, 2), horizontal)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_glider_se_desplaza_una_diagonal_cada_4_generaciones(sim):
    glider = [".X.", "..X", "XXX"]
    inicial = from_pattern(glider, 12, 12, top=1, left=1)
    esperado = from_pattern(glider, 12, 12, top=2, left=2)
    assert np.array_equal(sim(inicial, 4), esperado)


# ---------- Bordes muertos ----------

@pytest.mark.parametrize("step", ENGINES)
def test_bordes_muertos_no_dan_la_vuelta(step):
    g = from_pattern(["XXX"], 5, 5, top=0, left=1)
    esperado = from_pattern(["X", "X"], 5, 5, top=0, left=2)
    assert np.array_equal(step(g), esperado)


# ---------- Contrato de simulate ----------

@pytest.mark.parametrize("sim", SIMULATORS)
def test_simulate_no_modifica_la_matriz_original(sim):
    g = random_grid(10, 10, seed=1)
    copia = g.copy()
    sim(g, 5)
    assert np.array_equal(g, copia)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cero_generaciones_devuelve_la_misma_matriz(sim):
    g = random_grid(8, 8, seed=2)
    assert np.array_equal(sim(g, 0), g)


def test_random_grid_es_reproducible_con_semilla():
    assert np.array_equal(random_grid(20, 20, seed=42), random_grid(20, 20, seed=42))


# ---------- Equivalencia entre las dos versiones ----------

@pytest.mark.parametrize("rows,cols", [(1, 1), (1, 7), (7, 1), (10, 10), (13, 29), (50, 50)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bucles_y_numpy_dan_el_mismo_resultado(rows, cols, seed):
    g = random_grid(rows, cols, density=0.35, seed=seed)
    assert np.array_equal(simulate_loops(g, 15), simulate_numpy(g, 15))