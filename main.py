"""Punto de entrada principal del proyecto Juego de la Vida (Computación Paralela).

Permite:
- Ejecutar la aplicación gráfica interactiva (Pygame):
    python main.py --ui
- Ejecutar los experimentos de benchmarking:
    python main.py --benchmark --sizes 100 200 400
- Generar las gráficas de Speedup y Eficiencia a partir de resultados previos:
    python main.py --plots
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Juego de la Vida - Computación Paralela (Parcial 1)")
    parser.add_argument("--ui", action="store_true", default=False,
                        help="Lanzar la interfaz gráfica interactiva con Pygame")
    parser.add_argument("--benchmark", action="store_true", default=False,
                        help="Ejecutar suite de benchmarking de rendimiento y speedup")
    parser.add_argument("--plots", action="store_true", default=False,
                        help="Generar gráficas de rendimiento a partir de resultados")
    parser.add_argument("--all", action="store_true", default=False,
                        help="Ejecutar benchmark completo y generar todas las gráficas")

    # Argumentos adicionales para benchmark si se invoca desde main
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400],
                        help="Tamaños de matriz para el benchmark (ej: 100 200 400)")
    parser.add_argument("--generations", type=int, default=30,
                        help="Número de generaciones para el benchmark")
    parser.add_argument("--trials", type=int, default=3,
                        help="Repeticiones por prueba en el benchmark")

    args = parser.parse_args()

    # Si no se pasó ningún flag, por defecto lanzamos la UI
    if not (args.ui or args.benchmark or args.plots or args.all):
        args.ui = True

    if args.all or args.benchmark:
        from benchmarks.run import main as run_benchmark
        # Pasar argumentos relevantes a run
        sys.argv = [
            "benchmarks/run.py",
            "--sizes", *[str(s) for s in args.sizes],
            "--generations", str(args.generations),
            "--trials", str(args.trials),
        ]
        print("Ejecutando experimentos de rendimiento...")
        run_benchmark()

    if args.all or args.plots:
        from benchmarks.plots import main as run_plots
        sys.argv = ["benchmarks/plots.py"]
        print("\nGenerando gráficas de Speedup y Eficiencia...")
        run_plots()

    if args.ui and not (args.benchmark or args.plots or args.all):
        from src.ui.pygame_app import launch_app
        print("Iniciando interfaz gráfica Pygame...")
        launch_app()


if __name__ == "__main__":
    main()
