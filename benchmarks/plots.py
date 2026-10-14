"""Generador de gráficas de rendimiento y speedup para el Juego de la Vida.

Lee los datos generados por benchmarks/run.py (results/benchmark_results.json)
y genera gráficas en alta resolución (PNG 300 DPI) guardadas en results/plots/:
1. execution_time.png  -> Tiempo de ejecución vs número de procesos y tamaños.
2. speedup.png         -> Curva de Speedup real vs Speedup Ideal.
3. efficiency.png      -> Eficiencia paralela E(p) = S(p)/p.
4. summary_dashboard.png -> Panel consolidado de 4 cuadrantes para el informe.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def set_plot_style():
    """Aplica un estilo visual moderno y limpio para informes académicos."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "figure.dpi": 300,
        "lines.linewidth": 2.2,
        "lines.markersize": 7,
    })


def plot_execution_time(experiments, output_path: Path):
    """Gráfico de barras agrupadas: Tiempo de ejecución vs Tamaño de matriz."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    sizes = [exp["size"] for exp in experiments]
    x = np.arange(len(sizes))

    # Recoger los procesos disponibles en los experimentos
    all_processes = sorted({int(p) for exp in experiments for p in exp["parallel"].keys()})

    total_bars = 2 + len(all_processes)  # loops + numpy + cada proceso
    width = 0.8 / total_bars

    # 1. Secuencial Loops
    loops_times = [exp["sequential_loops"]["median"] for exp in experiments]
    ax.bar(x - (total_bars / 2 - 0.5) * width, loops_times, width, label="Secuencial (Loops)", color="#e74c3c", alpha=0.9)

    # 2. Secuencial NumPy
    numpy_times = [exp["sequential_numpy"]["median"] for exp in experiments]
    ax.bar(x - (total_bars / 2 - 1.5) * width, numpy_times, width, label="Secuencial (NumPy)", color="#3498db", alpha=0.9)

    # 3. Paralelo para cada p
    colors = ["#2ecc71", "#1abc9c", "#9b59b6", "#f39c12"]
    for idx, p in enumerate(all_processes):
        par_times = [exp["parallel"].get(str(p), {}).get("median", 0) for exp in experiments]
        color = colors[idx % len(colors)]
        offset = x - (total_bars / 2 - (2.5 + idx)) * width
        ax.bar(offset, par_times, width, label=f"Paralelo ({p} proc)", color=color, alpha=0.9)

    ax.set_xlabel("Tamaño de matriz (N × N)")
    ax.set_ylabel("Tiempo de ejecución (segundos)")
    ax.set_title("Tiempo de Ejecución por Tamaño de Matriz y Motor")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}×{s}" for s in sizes])
    ax.set_yscale("log")  # Escala logarítmica para ver diferencias de varios órdenes de magnitud
    ax.set_ylabel("Tiempo de ejecución (s) [Escala Log]")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Grafica guardada: {output_path}")


def plot_speedup(experiments, output_path: Path):
    """Gráfico de Speedup S(p) = T_seq / T_par vs Número de Procesos."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Obtener el rango de procesos
    all_processes = sorted({int(p) for exp in experiments for p in exp["parallel"].keys()})
    if not all_processes:
        return

    # Línea de Speedup Ideal: S(p) = p
    max_p = max(all_processes)
    ideal_x = [1] + [p for p in all_processes if p > 1]
    ax.plot(ideal_x, ideal_x, "k--", label="Speedup Ideal Lineal (S = p)", linewidth=1.8, alpha=0.7)

    palette = ["#e67e22", "#2980b9", "#27ae60", "#8e44ad"]
    for i, exp in enumerate(experiments):
        size = exp["size"]
        x_vals = []
        y_vals = []
        for p in all_processes:
            p_str = str(p)
            if p_str in exp["parallel"]:
                x_vals.append(p)
                y_vals.append(exp["parallel"][p_str]["speedup"])

        color = palette[i % len(palette)]
        ax.plot(x_vals, y_vals, marker="o", label=f"Matriz {size}×{size}", color=color)

    ax.set_xlabel("Número de Procesos (p)")
    ax.set_ylabel("Speedup $S(p) = T_{seq} / T_{par}(p)$")
    ax.set_title("Aceleración (Speedup) vs Número de Procesos")
    ax.set_xticks(all_processes)
    ax.set_xlim(left=0.8, right=max_p + 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(frameon=True, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Grafica guardada: {output_path}")


def plot_efficiency(experiments, output_path: Path):
    """Gráfico de Eficiencia Paralela E(p) = S(p) / p."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    all_processes = sorted({int(p) for exp in experiments for p in exp["parallel"].keys()})
    if not all_processes:
        return

    # Línea ideal: 100% (E = 1.0)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Eficiencia Ideal (100%)", alpha=0.7)

    palette = ["#e67e22", "#2980b9", "#27ae60", "#8e44ad"]
    for i, exp in enumerate(experiments):
        size = exp["size"]
        x_vals = []
        y_vals = []
        for p in all_processes:
            p_str = str(p)
            if p_str in exp["parallel"]:
                x_vals.append(p)
                y_vals.append(exp["parallel"][p_str]["efficiency"])

        color = palette[i % len(palette)]
        ax.plot(x_vals, y_vals, marker="s", label=f"Matriz {size}×{size}", color=color)

    ax.set_xlabel("Número de Procesos (p)")
    ax.set_ylabel("Eficiencia $E(p) = S(p) / p$")
    ax.set_title("Eficiencia Paralela vs Número de Procesos")
    ax.set_xticks(all_processes)
    ax.set_ylim(bottom=0, top=1.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(frameon=True, loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Grafica guardada: {output_path}")


def plot_summary_dashboard(experiments, metadata, output_path: Path):
    """Genera un dashboard de 4 paneles para el informe final."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Dashboard de Rendimiento - Juego de la Vida ({metadata.get('generations', 50)} Generaciones)", fontsize=16, y=0.98)

    all_processes = sorted({int(p) for exp in experiments for p in exp["parallel"].keys()})
    sizes = [exp["size"] for exp in experiments]

    # Panel (0,0): Tiempos por tamaño
    ax = axs[0, 0]
    for exp in experiments:
        p_list = sorted([int(p) for p in exp["parallel"].keys()])
        t_list = [exp["parallel"][str(p)]["median"] for p in p_list]
        ax.plot(p_list, t_list, marker="o", label=f"Paralelo {exp['size']}×{exp['size']}")
    ax.set_title("A. Tiempos Paralelos por Proceso")
    ax.set_xlabel("Procesos (p)")
    ax.set_ylabel("Tiempo (s)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=9)

    # Panel (0,1): Speedup
    ax = axs[0, 1]
    if all_processes:
        ax.plot(all_processes, all_processes, "k--", label="Ideal S=p", alpha=0.7)
    for exp in experiments:
        p_list = sorted([int(p) for p in exp["parallel"].keys()])
        s_list = [exp["parallel"][str(p)]["speedup"] for p in p_list]
        ax.plot(p_list, s_list, marker="^", label=f"{exp['size']}×{exp['size']}")
    ax.set_title("B. Speedup $S(p) = T_{seq}/T_{par}$")
    ax.set_xlabel("Procesos (p)")
    ax.set_ylabel("Speedup")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=9)

    # Panel (1,0): Eficiencia
    ax = axs[1, 0]
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.7, label="Ideal (100%)")
    for exp in experiments:
        p_list = sorted([int(p) for p in exp["parallel"].keys()])
        e_list = [exp["parallel"][str(p)]["efficiency"] for p in p_list]
        ax.plot(p_list, e_list, marker="s", label=f"{exp['size']}×{exp['size']}")
    ax.set_title("C. Eficiencia Paralela $E(p) = S(p)/p$")
    ax.set_xlabel("Procesos (p)")
    ax.set_ylabel("Eficiencia")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=9)

    # Panel (1,1): Comparación de Motores (Loops vs NumPy vs Paralelo max_p)
    ax = axs[1, 1]
    x = np.arange(len(sizes))
    width = 0.25
    t_loops = [exp["sequential_loops"]["median"] for exp in experiments]
    t_numpy = [exp["sequential_numpy"]["median"] for exp in experiments]
    max_p_str = str(max(all_processes)) if all_processes else "1"
    t_par = [exp["parallel"].get(max_p_str, {}).get("median", 0) for exp in experiments]

    ax.bar(x - width, t_loops, width, label="Seq Loops", color="#e74c3c")
    ax.bar(x, t_par, width, label=f"Paralelo ({max_p_str}p)", color="#2ecc71")
    ax.bar(x + width, t_numpy, width, label="Seq NumPy", color="#3498db")
    ax.set_title(f"D. Comparativa de Enfoques (vs {max_p_str} procesos)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}×{s}" for s in sizes])
    ax.set_yscale("log")
    ax.set_ylabel("Tiempo (s) [Escala Log]")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Dashboard consolidado guardado: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generador de gráficas a partir de resultados de benchmarks")
    parser.add_argument("--input", type=str, default="results/benchmark_results.json",
                        help="Ruta al archivo JSON con los datos de benchmark")
    parser.add_argument("--output-dir", type=str, default="results/plots",
                        help="Carpeta donde se guardarán las imágenes generadas")
    args = parser.parse_args()

    input_file = PROJECT_ROOT / args.input
    if not input_file.exists():
        print(f"Error: Archivo de datos no encontrado en {input_file}")
        print("Ejecuta primero: python benchmarks/run.py")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    experiments = data.get("experiments", [])
    metadata = data.get("metadata", {})
    if not experiments:
        print("Error: No se encontraron experimentos en el archivo JSON.")
        sys.exit(1)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    set_plot_style()
    print("=" * 60)
    print("GENERANDO GRAFICAS DE RENDIMIENTO...")
    print("=" * 60)

    plot_execution_time(experiments, output_dir / "execution_time.png")
    plot_speedup(experiments, output_dir / "speedup.png")
    plot_efficiency(experiments, output_dir / "efficiency.png")
    plot_summary_dashboard(experiments, metadata, output_dir / "summary_dashboard.png")

    print("=" * 60)
    print(f"Todas las graficas han sido exportadas a: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
