"""Puente de comunicación IPC entre la UI de Pygame y el motor paralelo MPI.

Permite que la interfaz gráfica interactiva (que se ejecuta en un único proceso)
se comunique de forma persistente y ultrarrápida con un clúster de procesos MPI
(lanzados mediante `mpirun -n P`) usando sockets locales (multiprocessing.connection).
"""
import atexit
import shutil
import subprocess
import sys
import time
from multiprocessing.connection import Client, Listener
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AUTH_KEY = b"mpi_game_of_life_key"


def run_worker_process(port: int):
    """Bucle principal ejecutado por cada proceso del clúster MPI."""
    from mpi4py import MPI
    from src.core.paralelo import split_matrix, next_local_generation

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    conn = None
    if rank == 0:
        try:
            conn = Client(("127.0.0.1", port), authkey=AUTH_KEY)
        except Exception as e:
            sys.stderr.write(f"[MPI Worker R0] Error al conectar con UI: {e}\n")
            sys.stderr.flush()
            comm.Abort(1)
            return

    local_grid = None

    while True:
        if rank == 0:
            try:
                msg = conn.recv()
            except EOFError:
                msg = ("exit",)
            except Exception as e:
                sys.stderr.write(f"[MPI Worker R0] Error en recv: {e}\n")
                msg = ("exit",)
        else:
            msg = None

        msg = comm.bcast(msg, root=0)
        cmd = msg[0]

        if cmd == "exit":
            break

        elif cmd == "set_grid":
            grid = msg[1]
            if rank == 0:
                sub = split_matrix(grid, size)
            else:
                sub = None
            local_grid = comm.scatter(sub, root=0)
            if rank == 0:
                conn.send("OK")

        elif cmd == "step":
            if local_grid is None:
                if rank == 0:
                    conn.send(("error", "Matriz no inicializada en workers"))
                continue

            t0 = MPI.Wtime()
            local_grid = next_local_generation(local_grid, comm)
            parts = comm.gather(local_grid, root=0)
            t_calc = MPI.Wtime() - t0

            if rank == 0:
                full = np.vstack(parts)
                conn.send((full, t_calc))

    if rank == 0 and conn is not None:
        try:
            conn.close()
        except Exception:
            pass


class MPIWorkerManager:
    """Administrador del clúster de procesos MPI para la UI."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.conn = None
        self.listener: Listener | None = None
        self.num_processes: int = 0
        self.current_shape: tuple[int, int] | None = None
        atexit.register(self.stop)

    @staticmethod
    def is_mpi_available() -> bool:
        """Verifica si mpirun está disponible."""
        return shutil.which("mpirun") is not None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and self.conn is not None

    def start(self, num_processes: int, initial_grid: np.ndarray) -> bool:
        """Inicia el clúster de procesos MPI y envía la matriz inicial."""
        self.stop()

        if not self.is_mpi_available():
            print("[MPI] mpirun no está disponible en el sistema.")
            return False

        try:
            self.listener = Listener(("127.0.0.1", 0), authkey=AUTH_KEY)
            if hasattr(self.listener, "_listener") and hasattr(self.listener._listener, "_socket"):
                self.listener._listener._socket.settimeout(6.0)
            port = self.listener.address[1]

            cmd = [
                "mpirun",
                "-n", str(num_processes),
                sys.executable,
                "-m", "src.core.mpi_bridge",
                str(port),
            ]

            self.proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Esperar conexión del rank 0
            self.conn = self.listener.accept()
            self.num_processes = num_processes

            # Enviar cuadrícula inicial
            self.set_grid(initial_grid)
            return True

        except Exception as e:
            err_msg = ""
            if self.proc is not None:
                try:
                    raw_err = self.proc.stderr.read()
                    if raw_err:
                        err_msg = raw_err.decode("utf-8", errors="replace").strip()
                except Exception:
                    pass
            detail = f" -> {err_msg}" if err_msg else f": {e}"
            print(f"[MPI] Error al iniciar worker MPI ({num_processes}P){detail}")
            self.stop()
            return False

    def set_grid(self, grid: np.ndarray):
        """Envía una nueva cuadrícula al clúster de procesos."""
        if not self.is_running():
            return
        try:
            self.conn.send(("set_grid", grid.astype(np.uint8)))
            resp = self.conn.recv()
            if resp == "OK":
                self.current_shape = grid.shape
        except Exception as e:
            print(f"[MPI] Error en set_grid: {e}")
            self.stop()

    def step(self) -> tuple[np.ndarray | None, float]:
        """Avanza una generación en paralelo en el clúster MPI.

        Devuelve (nueva_matriz, tiempo_de_calculo_segundos).
        """
        if not self.is_running():
            return None, 0.0

        try:
            self.conn.send(("step",))
            res = self.conn.recv()
            if isinstance(res, tuple) and len(res) == 2:
                full_grid, t_calc = res
                return full_grid, t_calc
        except Exception as e:
            print(f"[MPI] Error durante step: {e}")
            self.stop()

        return None, 0.0

    def stop(self):
        """Detiene de forma limpia el clúster MPI."""
        if self.conn is not None:
            try:
                self.conn.send(("exit",))
                self.conn.close()
            except Exception:
                pass
            self.conn = None

        if self.listener is not None:
            try:
                self.listener.close()
            except Exception:
                pass
            self.listener = None

        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

        self.num_processes = 0
        self.current_shape = None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        port_arg = int(sys.argv[1])
        run_worker_process(port_arg)
