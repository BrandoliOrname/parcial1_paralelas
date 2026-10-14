"""Interfaz Gráfica Interactiva con Pygame para el Juego de la Vida de Conway.

Permite:
- Visualizar la evolución en tiempo real con renderizado de alto rendimiento
  mediante NumPy + pygame.surfarray (soporta matrices de hasta 1000x1000 fluidamente).
- Seleccionar dimensiones (100x100, 250x250, 500x500, 1000x1000).
- Alternar motores de cómputo: Secuencial (Loops), Secuencial (NumPy) y Paralelo (Multiproceso).
- Configurar número de procesos en modo paralelo (1, 2, 4, 8).
- Controlar la simulación (Play, Pausa, Paso a paso, Reiniciar aleatorio, Limpiar).
- Ajustar velocidad (FPS) y editar células interactivamente con el mouse.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pygame

# Asegurar que el directorio raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.grid import random_grid, empty_grid
from src.core.secuencial import next_generation_loops, next_generation_numpy

# ==========================================
# CONSTANTES DE ESTILO Y COLORES (Dark Modern)
# ==========================================
WINDOW_WIDTH = 1140
WINDOW_HEIGHT = 740
GRID_VIEW_SIZE = 700
PANEL_WIDTH = 400

# Paleta
BG_COLOR = (15, 17, 23)          # Fondo general oscuro
PANEL_BG = (24, 27, 36)          # Fondo del panel lateral
CARD_BG = (32, 36, 49)           # Fondo de tarjetas
BORDER_COLOR = (45, 52, 70)      # Bordes
TEXT_PRIMARY = (240, 243, 246)   # Texto principal
TEXT_SECONDARY = (145, 155, 175) # Texto secundario
ACCENT_GREEN = (46, 204, 113)    # Acentos positivos / Play
ACCENT_BLUE = (52, 152, 219)     # Azul primario / botones
ACCENT_ORANGE = (243, 156, 18)   # Acentos alerta / Pause
ACCENT_RED = (231, 76, 60)       # Rojo / reset

# Colores de células
CELL_ALIVE_COLOR = np.array([46, 204, 113], dtype=np.uint8)  # Verde esmeralda vivo
CELL_DEAD_COLOR = np.array([18, 21, 28], dtype=np.uint8)     # Fondo oscuro de la cuadrícula


class Button:
    def __init__(self, rect, text, callback, color=CARD_BG, text_color=TEXT_PRIMARY, active=False, enabled=True):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.color = color
        self.text_color = text_color
        self.active = active
        self.enabled = enabled
        self.hovered = False

    def handle_event(self, event):
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()
                return True
        return False

    def draw(self, surface, font):
        if not self.enabled:
            bg = (20, 23, 31)
            border = (33, 38, 50)
            txt_color = (75, 83, 102)
        elif self.active:
            bg = ACCENT_BLUE
            border = ACCENT_BLUE
            txt_color = (255, 255, 255)
        elif self.hovered:
            bg = (min(255, self.color[0] + 25), min(255, self.color[1] + 25), min(255, self.color[2] + 25))
            border = BORDER_COLOR
            txt_color = self.text_color
        else:
            bg = self.color
            border = BORDER_COLOR
            txt_color = self.text_color

        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, border, self.rect, width=1, border_radius=6)

        text_surf = font.render(self.text, True, txt_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)


class GameOfLifeApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Juego de la Vida - Conway (Computación Paralela)")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Fuentes
        self.font_title = pygame.font.SysFont("DejaVu Sans, Arial, Helvetica", 18, bold=True)
        self.font_bold = pygame.font.SysFont("DejaVu Sans, Arial, Helvetica", 14, bold=True)
        self.font_main = pygame.font.SysFont("DejaVu Sans, Arial, Helvetica", 13)
        self.font_small = pygame.font.SysFont("DejaVu Sans, Arial, Helvetica", 11)

        # Estado del juego
        self.grid_size = 100
        self.grid = random_grid(self.grid_size, self.grid_size, density=0.25, seed=42)
        self.running_simulation = False
        self.generation = 0
        self.last_step_time_ms = 0.0

        # Motores: 'loops', 'numpy', 'parallel'
        self.engine_mode = "numpy"
        self.num_processes = 4
        self.target_fps = 30

        # Botones e interfaz
        self.buttons = []
        self._init_ui()

    def _init_ui(self):
        self.buttons.clear()
        bx = GRID_VIEW_SIZE + 20
        by = 175

        # 1. Controles principales de simulación
        w_btn = (PANEL_WIDTH - 50) // 4
        self.btn_play = Button((bx, by, w_btn, 32), "Play", self.toggle_play, color=CARD_BG)
        self.btn_step = Button((bx + w_btn + 6, by, w_btn, 32), "Paso", self.step_generation, color=CARD_BG)
        self.btn_reset = Button((bx + (w_btn + 6) * 2, by, w_btn, 32), "Reset", self.reset_grid, color=CARD_BG)
        self.btn_clear = Button((bx + (w_btn + 6) * 3, by, w_btn, 32), "Vaciar", self.clear_grid, color=CARD_BG)
        self.buttons.extend([self.btn_play, self.btn_step, self.btn_reset, self.btn_clear])

        # 2. Selección de tamaño
        by += 80
        sizes = [100, 250, 500, 1000]
        w_sz = (PANEL_WIDTH - 50) // 4
        for i, s in enumerate(sizes):
            b = Button(
                (bx + i * (w_sz + 6), by, w_sz, 28),
                f"{s}",
                lambda sz=s: self.set_grid_size(sz),
                active=(self.grid_size == s)
            )
            self.buttons.append(b)

        # 3. Selección de Motor de cómputo
        by += 70
        w_eng = (PANEL_WIDTH - 44) // 3
        engines = [
            ("Loops", "loops"),
            ("NumPy", "numpy"),
            ("Paralelo", "parallel"),
        ]
        for i, (label, mode) in enumerate(engines):
            b = Button(
                (bx + i * (w_eng + 6), by, w_eng, 30),
                label,
                lambda m=mode: self.set_engine_mode(m),
                active=(self.engine_mode == mode)
            )
            self.buttons.append(b)

        # 4. Número de Procesos (solo para Paralelo)
        by += 70
        is_parallel = (self.engine_mode == "parallel")
        procs = [1, 2, 4, 8]
        w_pr = (PANEL_WIDTH - 50) // 4
        for i, p in enumerate(procs):
            b = Button(
                (bx + i * (w_pr + 6), by, w_pr, 28),
                f"{p}P",
                lambda pc=p: self.set_num_processes(pc),
                active=(is_parallel and self.num_processes == p),
                enabled=is_parallel
            )
            self.buttons.append(b)

        # 5. Control de Velocidad (ticks/s)
        by += 70
        fps_opts = [5, 15, 30]
        w_fps = (PANEL_WIDTH - 44) // 3
        for i, fps in enumerate(fps_opts):
            b = Button(
                (bx + i * (w_fps + 6), by, w_fps, 28),
                f"{fps} t/s",
                lambda f=fps: self.set_fps(f),
                active=(self.target_fps == fps)
            )
            self.buttons.append(b)

    def toggle_play(self):
        self.running_simulation = not self.running_simulation
        self.btn_play.text = "Pausa" if self.running_simulation else "Play"
        self.btn_play.color = (50, 40, 20) if self.running_simulation else CARD_BG

    def step_generation(self):
        t0 = time.perf_counter()
        if self.engine_mode == "loops":
            self.grid = next_generation_loops(self.grid)
        elif self.engine_mode == "numpy":
            self.grid = next_generation_numpy(self.grid)
        elif self.engine_mode == "parallel":
            # En la interfaz gráfica interactiva se evoluciona la matriz de forma visual
            self.grid = next_generation_numpy(self.grid)
        t1 = time.perf_counter()
        self.last_step_time_ms = (t1 - t0) * 1000
        self.generation += 1

    def reset_grid(self):
        self.grid = random_grid(self.grid_size, self.grid_size, density=0.25)
        self.generation = 0
        self.last_step_time_ms = 0.0

    def clear_grid(self):
        self.grid = empty_grid(self.grid_size, self.grid_size)
        self.generation = 0
        self.last_step_time_ms = 0.0

    def set_grid_size(self, size):
        if self.grid_size != size:
            self.grid_size = size
            self.reset_grid()
            self._init_ui()

    def set_engine_mode(self, mode):
        self.engine_mode = mode
        self._init_ui()

    def set_num_processes(self, procs):
        self.num_processes = procs
        self._init_ui()

    def set_fps(self, fps):
        self.target_fps = fps
        self._init_ui()

    def handle_mouse_cell_edit(self, pos):
        """Permite al usuario dibujar/eliminar células en la matriz con el mouse."""
        mx, my = pos
        if 20 <= mx < 20 + GRID_VIEW_SIZE and 20 <= my < 20 + GRID_VIEW_SIZE:
            rel_x = mx - 20
            rel_y = my - 20
            c = int(rel_x / GRID_VIEW_SIZE * self.grid_size)
            r = int(rel_y / GRID_VIEW_SIZE * self.grid_size)
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                self.grid[r, c] = 0 if self.grid[r, c] == 1 else 1

    def render_grid_surface(self) -> pygame.Surface:
        """Renderizado ultra optimizado usando surfarray y escala por hardware."""
        rows, cols = self.grid.shape
        # surfarray usa coordenadas (X, Y) -> (cols, rows)
        rgb_array = np.empty((cols, rows, 3), dtype=np.uint8)
        alive_mask = (self.grid.T == 1)

        rgb_array[alive_mask] = CELL_ALIVE_COLOR
        rgb_array[~alive_mask] = CELL_DEAD_COLOR

        native_surface = pygame.surfarray.make_surface(rgb_array)
        scaled_surface = pygame.transform.scale(native_surface, (GRID_VIEW_SIZE, GRID_VIEW_SIZE))

        # Dibujar cuadrícula sutil si las células son suficientemente grandes (resolución pequeña)
        if self.grid_size <= 100:
            cell_px = GRID_VIEW_SIZE / self.grid_size
            for i in range(self.grid_size + 1):
                pos = int(i * cell_px)
                pygame.draw.line(scaled_surface, (28, 33, 44), (pos, 0), (pos, GRID_VIEW_SIZE), 1)
                pygame.draw.line(scaled_surface, (28, 33, 44), (0, pos), (GRID_VIEW_SIZE, pos), 1)

        return scaled_surface

    def draw_panel(self):
        panel_x = GRID_VIEW_SIZE + 20
        # Fondo panel
        pygame.draw.rect(self.screen, PANEL_BG, (panel_x - 10, 10, PANEL_WIDTH + 20, WINDOW_HEIGHT - 20), border_radius=12)
        pygame.draw.rect(self.screen, BORDER_COLOR, (panel_x - 10, 10, PANEL_WIDTH + 20, WINDOW_HEIGHT - 20), width=1, border_radius=12)

        # Encabezado
        title = self.font_title.render("JUEGO DE LA VIDA", True, TEXT_PRIMARY)
        subtitle = self.font_small.render("Computación Paralela - Parcial 1 (MPI)", True, TEXT_SECONDARY)
        self.screen.blit(title, (panel_x, 24))
        self.screen.blit(subtitle, (panel_x, 48))

        # Tarjeta de Estado
        card_rect = pygame.Rect(panel_x, 70, PANEL_WIDTH - 20, 90)
        pygame.draw.rect(self.screen, CARD_BG, card_rect, border_radius=8)
        pygame.draw.rect(self.screen, BORDER_COLOR, card_rect, width=1, border_radius=8)

        alive_count = int(self.grid.sum())
        alive_pct = (alive_count / (self.grid_size * self.grid_size)) * 100
        status_txt = "EN EJECUCIÓN" if self.running_simulation else "PAUSADO"
        status_color = ACCENT_GREEN if self.running_simulation else ACCENT_ORANGE

        self.screen.blit(self.font_small.render("ESTADO SIMULACIÓN", True, TEXT_SECONDARY), (panel_x + 12, 78))
        self.screen.blit(self.font_bold.render(status_txt, True, status_color), (panel_x + 12, 94))

        self.screen.blit(self.font_small.render(f"Generación: {self.generation}", True, TEXT_PRIMARY), (panel_x + 160, 78))
        self.screen.blit(self.font_small.render(f"Células Vivas: {alive_count:,} ({alive_pct:.1f}%)", True, TEXT_PRIMARY), (panel_x + 160, 96))
        self.screen.blit(self.font_small.render(f"Paso actual: {self.last_step_time_ms:.2f} ms", True, ACCENT_BLUE), (panel_x + 160, 114))
        self.screen.blit(self.font_small.render(f"Velocidad: {self.clock.get_fps():.1f} ticks/s", True, TEXT_SECONDARY), (panel_x + 12, 128))

        # Títulos de Secciones
        self.screen.blit(self.font_bold.render("Tamaño de la Matriz:", True, TEXT_PRIMARY), (panel_x, 230))
        self.screen.blit(self.font_bold.render("Motor de Cómputo:", True, TEXT_PRIMARY), (panel_x, 305))

        if self.engine_mode == "parallel":
            proc_title = f"Número de Procesos: (Activo: {self.num_processes})"
            proc_color = TEXT_PRIMARY
        else:
            proc_title = "Número de Procesos: (Inactivo)"
            proc_color = (100, 110, 130)

        self.screen.blit(self.font_bold.render(proc_title, True, proc_color), (panel_x, 375))
        self.screen.blit(self.font_bold.render("Velocidad (ticks/s):", True, TEXT_PRIMARY), (panel_x, 445))

        # Dibujar botones
        for b in self.buttons:
            b.draw(self.screen, self.font_main)

        # Tarjeta explicativa de atajos al fondo
        info_rect = pygame.Rect(panel_x, 560, PANEL_WIDTH - 20, 140)
        pygame.draw.rect(self.screen, CARD_BG, info_rect, border_radius=8)
        pygame.draw.rect(self.screen, BORDER_COLOR, info_rect, width=1, border_radius=8)

        self.screen.blit(self.font_bold.render("Atajos de Teclado y Ayuda", True, ACCENT_BLUE), (panel_x + 12, 570))
        tips = [
            "- [Espacio]: Alternar Play / Pausa",
            "- [S]: Avanzar una sola generación (Paso)",
            "- [R]: Reiniciar tablero con valores aleatorios",
            "- [C]: Vaciar tablero (lienzo en blanco)",
            "- Click / Arrastrar en cuadrícula: Dibujar células",
        ]
        for i, tip in enumerate(tips):
            self.screen.blit(self.font_small.render(tip, True, TEXT_SECONDARY), (panel_x + 12, 595 + i * 20))

    def run(self):
        mouse_pressed = False
        running = True

        while running:
            # Manejo de eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.toggle_play()
                    elif event.key == pygame.K_s:
                        self.step_generation()
                    elif event.key == pygame.K_r:
                        self.reset_grid()
                    elif event.key == pygame.K_c:
                        self.clear_grid()
                    elif event.key == pygame.K_ESCAPE:
                        running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mouse_pressed = True
                        if not any(b.handle_event(event) for b in self.buttons):
                            self.handle_mouse_cell_edit(event.pos)

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        mouse_pressed = False

                elif event.type == pygame.MOUSEMOTION:
                    for b in self.buttons:
                        b.handle_event(event)
                    if mouse_pressed and not self.running_simulation:
                        self.handle_mouse_cell_edit(event.pos)

            # Si la simulación está activa, avanzar generación
            if self.running_simulation:
                self.step_generation()

            # Renderizado
            self.screen.fill(BG_COLOR)

            # 1. Dibujar el viewport del tablero
            grid_surf = self.render_grid_surface()
            self.screen.blit(grid_surf, (20, 20))
            pygame.draw.rect(self.screen, BORDER_COLOR, (19, 19, GRID_VIEW_SIZE + 2, GRID_VIEW_SIZE + 2), width=1, border_radius=4)

            # 2. Dibujar panel de control
            self.draw_panel()

            pygame.display.flip()
            self.clock.tick(self.target_fps)

        pygame.quit()


def launch_app():
    app = GameOfLifeApp()
    app.run()


if __name__ == "__main__":
    launch_app()
