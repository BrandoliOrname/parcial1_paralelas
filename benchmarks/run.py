"""Script para ejecutar experimentos controlados de benchmarking.

Mide tiempos de ejecución para distintos tamaños de matriz y número de procesos,
calcula la aceleración (Speedup) y la eficiencia paralela usando el motor
secuencial y el motor paralelo con MPI (src/core/paralelo.py).
Resultados guardados en results/ en formato JSON y CSV.
"""
import argparse
import csv
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.grid import random_grid
from src.core.secuencial import simulate_loops, simulate_numpy


def check_mpi_available() -> bool:
    """Verifica si mpirun está disponible en el sistema."""
    return shutil.which("mpirun") is not None


def run_mpi_trial(rows: int, cols: int, generations: int, processes: int, seed: int) -> float:
    """Ejecuta una medición llamando directamente a src.core.paralelo con mpirun."""
    cmd = [
        "mpirun",
        "-n", str(processes),
        sys.executable,
        "-m", "src.core.paralelo",
        "--rows", str(rows),
        "--cols", str(cols),
        "--generations", str(generations),
        "--seed", str(seed),
        "--quiet",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Error al ejecutar mpirun (código {proc.returncode}):\n{proc.stderr}")

    for line in proc.stdout.splitlines():
        if "Tiempo paralelo:" in line:
            match = re.search(r"Tiempo paralelo:\s*([0-9.]+)", line)
            if match:
                return float(match.group(1))

    raise RuntimeError(f"No se encontró el tiempo en la salida de MPI:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def measure_engine(fn, trials: int) -> dict:
    """Ejecuta una función de simulación varias veces y calcula estadísticas."""
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "times": times,
        "median": statistics.median(times),
        "mean": statistics.mean(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmarking del Juego de la Vida (Secuencial vs MPI paralelo)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 1000],
                        help="Tamaños de matriz cuadrada a evaluar (ej: 100 200 400 o 100 500 1000)")
    parser.add_argument("--generations", type=int, default=30,
                        help="Número fijo de generaciones por simulación")
    parser.add_argument("--trials", type=int, default=3,
                        help="Número de repeticiones para calcular mediana y filtrar ruido")
    parser.add_argument("--processes", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Número de procesos a evaluar con MPI (src/core/paralelo.py)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Carpeta donde se guardarán los resultados")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mpi_available = check_mpi_available()

    print("=" * 70)
    print("EXPERIMENTO DE RENDIMIENTO: JUEGO DE LA VIDA")
    print("=" * 70)
    print(f"- Tamaños de matriz:     {args.sizes}")
    print(f"- Generaciones fijas:    {args.generations}")
    print(f"- Repeticiones (trials): {args.trials}")
    print(f"- Procesos evaluados:    {args.processes}")
    print(f"- Motor paralelo:        MPI (src/core/paralelo.py)")
    print(f"- mpirun detectado:      {'SI' if mpi_available else 'NO (Requiere instalar OpenMPI)'}")
    print("=" * 70)

    if not mpi_available:
        print("\nADVERTENCIA: 'mpirun' no fue encontrado en el PATH del sistema.")
        print("   Para ejecutar los benchmarks paralelos con MPI, instala OpenMPI:")
        print("   sudo apt install openmpi-bin libopenmpi-dev\n")

    results_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "python_version": platform.python_version(),
            "system": platform.system(),
            "cpu_count": os.cpu_count(),
            "generations": args.generations,
            "trials": args.trials,
            "parallel_engine": "MPI (src/core/paralelo.py)",
        },
        "experiments": [],
    }

    csv_rows = []

    for size in args.sizes:
        print(f"\nEvaluando matriz {size}x{size} ({args.generations} generaciones, {args.trials} repeticiones)...")
        grid_base = random_grid(size, size, density=0.3, seed=42)

        # 1. Secuencial Loops (Base para el speedup)
        print("   [1/3] Midió Secuencial Loops...", end="", flush=True)
        seq_loops_stats = measure_engine(lambda: simulate_loops(grid_base, args.generations), args.trials)
        t_seq = seq_loops_stats["median"]
        print(f" Mediana: {t_seq:.4f} s (±{seq_loops_stats['std']:.4f})")

        csv_rows.append({
            "matrix_size": size,
            "engine": "secuencial_loops",
            "processes": 1,
            "generations": args.generations,
            "median_time": round(t_seq, 6),
            "mean_time": round(seq_loops_stats["mean"], 6),
            "std_time": round(seq_loops_stats["std"], 6),
            "speedup": 1.0,
            "efficiency": 1.0,
        })

        # 2. Secuencial NumPy (Referencia vectorizada)
        print("   [2/3] Midió Secuencial NumPy...", end="", flush=True)
        seq_numpy_stats = measure_engine(lambda: simulate_numpy(grid_base, args.generations), args.trials)
        t_numpy = seq_numpy_stats["median"]
        print(f" Mediana: {t_numpy:.4f} s (±{seq_numpy_stats['std']:.4f})")

        csv_rows.append({
            "matrix_size": size,
            "engine": "secuencial_numpy",
            "processes": 1,
            "generations": args.generations,
            "median_time": round(t_numpy, 6),
            "mean_time": round(seq_numpy_stats["mean"], 6),
            "std_time": round(seq_numpy_stats["std"], 6),
            "speedup": round(t_seq / t_numpy, 4) if t_numpy > 0 else 0,
            "efficiency": "-",
        })

        # 3. Paralelo MPI
        parallel_results = {}
        if mpi_available:
            for p in args.processes:
                print(f"   [3/3] Midió Paralelo MPI ({p} procesos)...", end="", flush=True)
                times = []
                for trial_idx in range(args.trials):
                    trial_seed = 42 + trial_idx
                    t = run_mpi_trial(size, size, args.generations, p, trial_seed)
                    times.append(t)

                par_median = statistics.median(times)
                par_mean = statistics.mean(times)
                par_std = statistics.stdev(times) if len(times) > 1 else 0.0
                speedup = t_seq / par_median if par_median > 0 else 0.0
                efficiency = speedup / p

                parallel_results[str(p)] = {
                    "times": times,
                    "median": par_median,
                    "mean": par_mean,
                    "std": par_std,
                    "speedup": speedup,
                    "efficiency": efficiency,
                }

                print(f" Mediana: {par_median:.4f} s | Speedup: {speedup:.2f}x | Eficiencia: {efficiency:.2%}")

                csv_rows.append({
                    "matrix_size": size,
                    "engine": "paralelo_mpi",
                    "processes": p,
                    "generations": args.generations,
                    "median_time": round(par_median, 6),
                    "mean_time": round(par_mean, 6),
                    "std_time": round(par_std, 6),
                    "speedup": round(speedup, 4),
                    "efficiency": round(efficiency, 4),
                })
        else:
            print("   [3/3] Omitiendo pruebas MPI (mpirun no disponible).")

        results_data["experiments"].append({
            "size": size,
            "generations": args.generations,
            "sequential_loops": seq_loops_stats,
            "sequential_numpy": seq_numpy_stats,
            "parallel": parallel_results,
        })

    # Guardar resultados en JSON
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    # Guardar resultados en CSV
    csv_path = output_dir / "benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "matrix_size", "engine", "processes", "generations",
            "median_time", "mean_time", "std_time", "speedup", "efficiency"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETADO EXITOSAMENTE")
    print(f"Resultados JSON guardados en: {json_path}")
    print(f"Resultados CSV guardados en:  {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
