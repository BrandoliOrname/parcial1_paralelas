from mpi4py import MPI
import numpy as np

from .grid import random_grid, DTYPE
from .secuencial import next_generation_loops


def split_matrix(
    grid: np.ndarray,
    num_processes: int
) -> list[np.ndarray]:
    """
    Divide la matriz por filas entre los procesos.

    Si el número de filas no es divisible exactamente entre el número
    de procesos, las filas restantes se distribuyen entre los primeros
    procesos para mantener la carga lo más balanceada posible.
    """

    rows, cols = grid.shape

    rows_per_process = rows // num_processes
    remainder = rows % num_processes

    submatrices = []
    start_row = 0

    for i in range(num_processes):

        rows_for_process = (
            rows_per_process
            + (1 if i < remainder else 0)
        )

        end_row = start_row + rows_for_process

        submatrices.append(
            grid[start_row:end_row, :]
        )

        start_row = end_row

    return submatrices


def distribute_grid(
    grid: np.ndarray | None,
    comm
) -> np.ndarray:
    """
    El rank 0 divide la matriz global.

    Posteriormente MPI distribuye físicamente una submatriz
    diferente a cada proceso.
    """

    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        if grid is None:
            raise ValueError(
                "El rank 0 debe recibir la matriz global"
            )

        if size > grid.shape[0]:
            raise ValueError(
                "No puede haber más procesos que filas en la matriz"
            )

        submatrices = split_matrix(
            grid,
            size
        )

    else:
        submatrices = None

    # -----------------------------------------
    # MESSAGE PASSING
    # -----------------------------------------
    # Cada proceso recibe físicamente su propia
    # región de la matriz.
    local_grid = comm.scatter(
        submatrices,
        root=0
    )

    return local_grid


def exchange_halos(
    local_grid: np.ndarray,
    comm
) -> tuple[np.ndarray, np.ndarray]:
    """
    Intercambia las filas frontera entre procesos vecinos.

    Cada proceso necesita conocer:

        - la última fila del proceso superior
        - la primera fila del proceso inferior

    para calcular correctamente sus células ubicadas
    en los límites de la partición.
    """

    rank = comm.Get_rank()
    size = comm.Get_size()

    cols = local_grid.shape[1]

    # Por defecto se inicializan con ceros.
    # Esto representa correctamente los bordes externos
    # del tablero, donde se considera que no hay células.
    top_halo = np.zeros(
        cols,
        dtype=DTYPE
    )

    bottom_halo = np.zeros(
        cols,
        dtype=DTYPE
    )

    # Proceso inmediatamente superior.
    # Para rank 0 no existe.
    upper_process = (
        rank - 1
        if rank > 0
        else MPI.PROC_NULL
    )

    # Proceso inmediatamente inferior.
    # Para el último rank no existe.
    lower_process = (
        rank + 1
        if rank < size - 1
        else MPI.PROC_NULL
    )

    # =====================================================
    # INTERCAMBIO 1
    #
    # Envío mi última fila al proceso inferior.
    #
    # Al mismo tiempo recibo del proceso superior
    # su última fila.
    #
    # Esa fila recibida será mi halo superior.
    # =====================================================

    comm.Sendrecv(
        sendbuf=local_grid[-1],
        dest=lower_process,
        sendtag=0,

        recvbuf=top_halo,
        source=upper_process,
        recvtag=0
    )

    # =====================================================
    # INTERCAMBIO 2
    #
    # Envío mi primera fila al proceso superior.
    #
    # Al mismo tiempo recibo del proceso inferior
    # su primera fila.
    #
    # Esa fila recibida será mi halo inferior.
    # =====================================================

    comm.Sendrecv(
        sendbuf=local_grid[0],
        dest=upper_process,
        sendtag=1,

        recvbuf=bottom_halo,
        source=lower_process,
        recvtag=1
    )

    return top_halo, bottom_halo


def next_local_generation(
    local_grid: np.ndarray,
    comm
) -> np.ndarray:
    """
    Calcula la siguiente generación correspondiente únicamente
    a la región propiedad de este proceso.
    """

    # ------------------------------------------------------
    # 1. Obtener información de los procesos vecinos.
    # ------------------------------------------------------

    top_halo, bottom_halo = exchange_halos(
        local_grid,
        comm
    )

    # ------------------------------------------------------
    # 2. Construir temporalmente una matriz extendida.
    #
    #           ghost row superior
    #           ------------------
    #           filas locales
    #           filas locales
    #           ...
    #           ------------------
    #           ghost row inferior
    #
    # Las ghost rows NO pertenecen realmente a este proceso.
    # Únicamente contienen información necesaria para
    # calcular las células de las fronteras.
    # ------------------------------------------------------

    extended_grid = np.vstack(
        (
            top_halo,
            local_grid,
            bottom_halo
        )
    )

    # ------------------------------------------------------
    # 3. Reutilizar las reglas de la versión secuencial.
    #
    # next_generation_loops calcula también las ghost rows,
    # pero esos resultados posteriormente se descartan.
    # ------------------------------------------------------

    extended_next_generation = next_generation_loops(
        extended_grid
    )

    # ------------------------------------------------------
    # 4. El proceso conserva solamente sus propias filas.
    #
    # [1:-1] elimina:
    #
    #   - ghost row superior
    #   - ghost row inferior
    # ------------------------------------------------------

    next_local_grid = (
        extended_next_generation[1:-1].copy()
    )

    return next_local_grid


def simulate_mpi(
    initial_grid: np.ndarray | None,
    generations: int,
    comm
) -> np.ndarray | None:
    """
    Ejecuta varias generaciones del Juego de la Vida utilizando MPI.

    El rank 0 recibe la matriz global inicial.

    Los demás procesos reciben None.

    Al finalizar:
        - rank 0 devuelve la matriz global reconstruida.
        - los demás ranks devuelven None.
    """

    rank = comm.Get_rank()

    # ======================================================
    # ETAPA 1: PARTITION + SCATTER
    # ======================================================

    local_grid = distribute_grid(
        initial_grid,
        comm
    )

    # Esperar hasta que todos tengan su submatriz antes
    # de comenzar la medición.
    comm.Barrier()

    start_time = MPI.Wtime()

    # ======================================================
    # ETAPA 2: SIMULACIÓN
    # ======================================================

    for _ in range(generations):

        # En cada generación:
        #
        # 1. intercambiar halos
        # 2. calcular región propia
        # 3. reemplazar generación local
        local_grid = next_local_generation(
            local_grid,
            comm
        )

    # Esperar hasta que todos hayan terminado.
    comm.Barrier()

    end_time = MPI.Wtime()

    local_elapsed = end_time - start_time

    # ======================================================
    # El tiempo paralelo real está determinado por el
    # proceso que más tardó.
    # ======================================================

    parallel_time = comm.reduce(
        local_elapsed,
        op=MPI.MAX,
        root=0
    )

    # ======================================================
    # ETAPA 3: GATHER
    #
    # Cada proceso devuelve físicamente su región final
    # al proceso raíz.
    # ======================================================

    final_parts = comm.gather(
        local_grid,
        root=0
    )

    if rank == 0:

        final_grid = np.vstack(
            final_parts
        )

        print(
            f"Tiempo paralelo: "
            f"{parallel_time:.6f} s"
        )

        return final_grid

    return None


if __name__ == "__main__":

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    ROWS = 10
    COLS = 10
    GENERATIONS = 5

    # ======================================================
    # SOLO RANK 0 CREA EL ESTADO GLOBAL INICIAL
    # ======================================================

    if rank == 0:

        grid = random_grid(
            rows=ROWS,
            cols=COLS,
            density=0.3,
            seed=42
        )

        print("Generación inicial:")
        print(grid)

    else:
        grid = None

    # ======================================================
    # SIMULACIÓN MPI
    # ======================================================

    final_grid = simulate_mpi(
        initial_grid=grid,
        generations=GENERATIONS,
        comm=comm
    )

    # ======================================================
    # SOLO RANK 0 VUELVE A TENER LA MATRIZ COMPLETA
    # ======================================================

    if rank == 0:

        print(
            f"\nGeneración final "
            f"después de {GENERATIONS} generaciones:"
        )

        print(final_grid)