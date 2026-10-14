# Juego de la Vida de Conway - Computación Paralela (Parcial 1)

## Integrantes
- Brandon Alexis Franco
- Mateo Noguera Pinto
- Jonathan Aristizabal Vargas

---

## Descripción del Proyecto

Este proyecto consiste en la implementación y análisis experimental del **Juego de la Vida de Conway**, desarrollado para la asignatura de Infraestructuras Paralelas y Distribuidas.

El objetivo principal es comparar el rendimiento de una implementación secuencial frente a una paralelizada con paso de mensajes utilizando **MPI** (`mpi4py`). La versión paralela aplica una estrategia de **descomposición de dominio 1D por filas**, donde el espacio de trabajo se divide entre los procesos disponibles y las filas fronterizas se intercambian entre vecinos en cada generación mediante comunicaciones punto a punto (`Sendrecv`).

El sistema incluye:
1. **Motor secuencial** con dos variantes: bucles anidados en Python puro (línea base para el cálculo de speedup) y versión vectorizada con NumPy.
2. **Motor paralelo** distribuido con MPI.
3. **Suite de pruebas unitarias** para validar la corrección de las reglas de Conway, osciladores, naves espaciales y particionado de dominios.
4. **Módulo de experimentación y benchmarking** para registrar tiempos de ejecución, aceleración (*speedup*) y eficiencia en matrices de diversos tamaños y cantidades de procesos.
5. **Generador automático de gráficas** para el informe técnico.
6. **Interfaz gráfica interactiva** construida con Pygame para la visualización en tiempo real.

---

## Requisitos e Instalación

### 1. Requisitos del Sistema
- Python 3.14.
- Implementación de MPI a nivel de sistema operativo (OpenMPI o MPICH):
  ```bash
  sudo apt install openmpi-bin libopenmpi-dev
  ```

### 2. Configuración del Entorno Virtual
Crear y activar el entorno virtual:
```bash
uv venv .venv
source .venv/bin/activate
```
*(O con python3 si no se tiene uv: `python3 -m venv .venv && source .venv/bin/activate`)*

Instalar dependencias del proyecto:
```bash
pip install -r requirements.txt
```

---

## Instrucciones de Ejecución

El proyecto dispone de un punto de entrada unificado a través de `main.py`.

### 1. Interfaz Gráfica (Pygame)
Para iniciar la simulación visual interactiva:
```bash
python main.py --ui
```
O de forma directa:
```bash
python main.py
```

**Funcionalidades de la interfaz:**
- Selección de dimensiones de matriz: 100x100, 250x250, 500x500 y 1000x1000.
- Selección de motor de cómputo en caliente: Loops, NumPy y Paralelo.
- Configuración de velocidad (ticks/s).
- Controles de simulación: Iniciar, Pausar, Paso a paso, Reiniciar aleatorio y Vaciar cuadrícula.
- Edición interactiva: Con la simulación pausada, hacer clic o arrastrar el cursor sobre el tablero para conmutar el estado de las células.
- Atajos de teclado:
  - `Espacio`: Alternar Iniciar / Pausar.
  - `S`: Avanzar una sola generación (Paso).
  - `R`: Reiniciar con distribución aleatoria.
  - `C`: Vaciar el tablero (lienzo en blanco para dibujar).

---

### 2. Pruebas Unitarias
Para verificar la validez de los algoritmos y la consistencia de las reglas de Conway:
```bash
python -m pytest
```

---

### 3. Benchmarking de Rendimiento
Para ejecutar mediciones controladas de tiempo de ejecución entre la versión secuencial y la versión paralela MPI:

```bash
python main.py --benchmark --sizes 100 500 1000 --generations 30 --trials 3
```

**Parámetros configurables:**
- `--sizes`: Lista de dimensiones cuadradas de matriz (ejemplo: `--sizes 100 500 1000`).
- `--generations`: Número fijo de generaciones a calcular (ejemplo: `--generations 50`).
- `--trials`: Número de repeticiones para promediar y calcular la mediana estadística.

Los resultados numéricos detallados se exportan automáticamente en los siguientes formatos:
- `results/benchmark_results.json`
- `results/benchmark_results.csv`

---

### 4. Generación de Gráficas de Rendimiento
Una vez ejecutado el benchmark, se pueden generar las figuras estadísticas para el informe ejecutando:

```bash
python main.py --plots
```

Las figuras se guardan en el directorio `results/plots/` con resolución de 300 DPI:
- `execution_time.png`: Comparativa de tiempos de ejecución por tamaño de matriz y número de procesos.
- `speedup.png`: Curva de aceleración real frente al speedup ideal lineal ($S = p$).
- `efficiency.png`: Eficiencia del uso de recursos paralelos ($E(p) = S(p)/p$).
- `summary_dashboard.png`: Panel integrado con las cuatro métricas principales para el documento final.

---

### 5. Ejecución Directa de MPI
Para ejecutar el script del motor paralelo de manera aislada utilizando directamente `mpirun`:
```bash
mpirun -n 4 python -m src.core.paralelo
```
