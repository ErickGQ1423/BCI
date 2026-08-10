#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pygame
import socket
import sys
import time
import random
import os
import serial  # <--- AGREGADO: Comunicación Serial
from pathlib import Path
from pylsl import StreamInlet, resolve_stream

# Personal modules
from Utils.visualization import draw_arrow_fill, draw_ball_fill, draw_fixation_cross, draw_time_balls
from Utils.experiment_utils import generate_trial_sequence
from Utils.networking import send_udp_message, display_multiple_messages_with_udp
import config
from Utils.logging_manager import LoggerManager

# ============================================================
# CONFIG
# ============================================================
NEXT_INDICATOR_POS = (0.50, 0.28)
NEXT_INDICATOR_SCALE = 1.00

# UDP Sockets
udp_socket_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_robot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
fes_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

FES_toggle = config.FES_toggle
OFFLINE_FILL_ALPHA = getattr(config, "OFFLINE_FEEDBACK_FILL_ALPHA", getattr(config, "FEEDBACK_FILL_ALPHA", 180))

# --- ARDUINO CONFIG (AGREGADO) ---
# OPCIÓN A: Env var tiene prioridad; si no está definida, usa config.ARDUINO_PORT
ARDUINO_PORT = os.environ.get("ARDUINO_PORT", config.ARDUINO_PORT if config.USE_ARDUINO else "")

# OPCIÓN B: Usar Manual desde Terminal (Descomentada AHORA)
# ARDUINO_PORT = "/dev/ttyACM0"

ARDUINO_BAUD = int(os.environ.get("ARDUINO_BAUD", 9600))
arduino_ser = None  # Variable para la conexión
# ---------------------------------

# Logging
recording_subject = getattr(config, "RECORDING_SUBJECT", config.TRAINING_SUBJECT)
recording_data_dir = Path(getattr(config, "RECORDING_DATA_DIR", config.DATA_DIR))
logger = LoggerManager.auto_detect_from_subject(
    subject=recording_subject,
    base_path=recording_data_dir
)

# Config snapshot
loggable_fields = [
    "UDP_MARKER", "UDP_ROBOT", "UDP_FES", "ARM_SIDE", "TOTAL_TRIALS",
    "TIME_MI", "FES_toggle", "TRAINING_SUBJECT", "DATA_DIR",
    "RECORDING_SUBJECT", "RECORDING_DATA_DIR", "OFFLINE_FEEDBACK_FILL_ALPHA"
]
config_log_subset = {k: getattr(config, k) for k in loggable_fields if hasattr(config, k)}
logger.save_config_snapshot(config_log_subset)

pygame.init()

# if config.BIG_BROTHER_MODE:
#     os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
#     screen = pygame.display.set_mode((1920, 1080), pygame.NOFRAME)
# else:
#     screen = pygame.display.set_mode((0, 0), pygame.NOFRAME)

# screen_width, screen_height = pygame.display.Info().current_w, pygame.display.Info().current_h

# Reemplaza tu bloque actual por este:
if config.BIG_BROTHER_MODE:
    # Fuerza una resolución específica sin bordes (útil para setups de monitores múltiples)
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    screen = pygame.display.set_mode((3840, 2160), pygame.NOFRAME)
else:
    # MODO PANTALLA COMPLETA REAL
    # (0,0) detecta la resolución nativa de tu monitor actual
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

# Esto obtiene las dimensiones reales después de activar el modo
screen_width, screen_height = screen.get_size()


# ============================================================
# ARDUINO SETUP (AGREGADO)
# ============================================================
if ARDUINO_PORT:
    try:
        logger.log_event(f"Connecting to Glove (Arduino) on {ARDUINO_PORT}...")
        arduino_ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.1)
        # Espera de seguridad para el reinicio del Arduino
        time.sleep(2)
        logger.log_event("Glove connected successfully.")
    except Exception as e:
        logger.log_event(f"ERROR connecting to Glove: {e}")
        arduino_ser = None
else:
    logger.log_event("No Arduino port configured (Visual Only Mode).")


def arduino_write(cmd: bytes):
    if arduino_ser is None:
        return
    try:
        arduino_ser.write(cmd)
    except Exception as e:
        logger.log_event(f"⚠️ Arduino write failed: {e}")


def glove_cmd_for_mode(mode: int) -> bytes:
    """Return configured glove command for MI/opening vs REST baseline."""
    return (
        getattr(config, "ARDUINO_CMD_MI", b"0")
        if mode == 0 else
        getattr(config, "ARDUINO_CMD_REST", b"1")
    )


def glove_cmd_rest() -> bytes:
    """Configured closed/baseline glove command."""
    return getattr(config, "ARDUINO_CMD_REST", b"1")


def glove_cmd_mi() -> bytes:
    """Configured open glove command."""
    return getattr(config, "ARDUINO_CMD_MI", b"0")


# Force glove to start every experiment in the configured closed/baseline state.
if arduino_ser and arduino_ser.is_open:
    arduino_write(glove_cmd_rest())
    logger.log_event("🤚 Glove initialized to closed/baseline state.")
    init_settle = float(getattr(config, "ARDUINO_INIT_SETTLE_SECONDS", 3.0))
    if init_settle > 0:
        logger.log_event(
            f"⏳ Waiting {init_settle:.1f}s for glove to reach closed/baseline state."
        )
        time.sleep(init_settle)


# ============================================================
# VISUAL FUNCTIONS (TUS FUNCIONES EXACTAS)
# ============================================================

def draw_arrow_directional(screen, pos_x, pos_y, size, color, direction="right"):
    """
    Draws a complete arrow (line + triangle tip) with offset correction.
    The line end is adjusted to stay behind the triangle's tip.
    """
    # 1. Geometry Setup
    line_len = size * 0.8
    tri_size = size // 2
    
    # OFFSET CORRECTION: Move the line end point slightly 'inwards' 
    # so it doesn't poke out of the triangle's tip.
    offset = 5  # pixels
    
    if direction == "right":
        line_start = (pos_x - line_len, pos_y)
        line_end = (pos_x + line_len - offset, pos_y) # Pulled back
        
        # Tip points (Right)
        points = [
            (pos_x + line_len, pos_y),                  # Tip
            (pos_x + line_len - tri_size, pos_y - tri_size), # Top back
            (pos_x + line_len - tri_size, pos_y + tri_size)  # Bottom back
        ]
    else:
        line_start = (pos_x + line_len, pos_y)
        line_end = (pos_x - line_len + offset, pos_y) # Pulled back
        
        # Tip points (Left)
        points = [
            (pos_x - line_len, pos_y),                  # Tip
            (pos_x - line_len + tri_size, pos_y - tri_size), # Top back
            (pos_x - line_len + tri_size, pos_y + tri_size)  # Bottom back
        ]

    # 2. Draw Body (Line)
    pygame.draw.line(screen, color, line_start, line_end, 12)

    # 3. Draw Tip (Triangle)
    pygame.draw.polygon(screen, color, points)

def draw_neutral_intertrial_screen():
    """Neutral visual state used during intertrial and hidden REST preparation."""
    pygame.display.get_surface().fill(config.black)
    draw_fixation_cross(screen_width, screen_height)
    draw_ball_fill(0, screen_width, screen_height, show_threshold=False, fill_alpha=OFFLINE_FILL_ALPHA)
    draw_arrow_fill(0, screen_width, screen_height, show_threshold=False, fill_alpha=OFFLINE_FILL_ALPHA)
    draw_time_balls(0, screen_width, screen_height, mode="single", single_pos=NEXT_INDICATOR_POS)
    pygame.display.flip()


def display_fixation_period(duration=3):
    start_time = time.time()
    clock = pygame.time.Clock()
    while time.time() - start_time < duration:
        draw_neutral_intertrial_screen()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
        clock.tick(60)

def draw_pretrial_screen(next_color, time_ball_state, elapsed_ms=0, total_ms=2000):
    """Pre-trial with geometric outline and countdown bar."""
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)

    is_mi = (next_color == (255, 50, 50) or
             next_color == getattr(config, 'red', (255, 50, 50)))
    rest_neutral_visual = (
        (not is_mi)
        and bool(getattr(config, "OFFLINE_REST_NEUTRAL_VISUAL", False))
    )
    if rest_neutral_visual:
        # Keep REST preparation visually identical to the intertrial screen.
        # Triggers/timing remain unchanged; only the cue is hidden.
        draw_neutral_intertrial_screen()
        return

    # ── Figura geométrica correspondiente ────────────────────
    if is_mi:
        draw_arrow_fill(0, screen_width, screen_height, show_threshold=False, fill_alpha=OFFLINE_FILL_ALPHA)
    else:
        draw_ball_fill(0, screen_width, screen_height, show_threshold=False, fill_alpha=OFFLINE_FILL_ALPHA)

# ── Countdown bar ────────────────────────────────────────
    bar_w     = int(screen_width * 0.2)
    bar_h     = 12
    bar_x     = screen_width // 2 - bar_w // 2
    bar_y     = screen_height // 2 + 245
    progress  = min(elapsed_ms / total_ms, 1.0)  # crece de 0 a 1
    fill_w    = int(bar_w * progress)
    bar_color = (255, 50, 50) if is_mi else (0, 120, 255)

    # Fondo gris
    if not rest_neutral_visual:
        pygame.draw.rect(screen, (60, 60, 60),
                         (bar_x, bar_y, bar_w, bar_h), border_radius=6)

    if fill_w > 0 and not rest_neutral_visual:
        if is_mi:
            # Rojo: se llena de izquierda a derecha
            pygame.draw.rect(screen, bar_color,
                             (bar_x, bar_y, fill_w, bar_h),
                             border_radius=6)
        else:
            # Azul: se llena de derecha a izquierda
            pygame.draw.rect(screen, bar_color,
                             (bar_x + bar_w - fill_w, bar_y, fill_w, bar_h),
                             border_radius=6)

    # ── Indicador pequeño arriba (NEXT) ──────────────────────
    pos_x = int(screen_width * NEXT_INDICATOR_POS[0])
    pos_y = int(screen_height * NEXT_INDICATOR_POS[1])
    base_size = int(min(screen_width, screen_height) * 0.08 * NEXT_INDICATOR_SCALE)

    if is_mi:
        bg_rect = pygame.Rect(pos_x - base_size//2, pos_y - base_size//2,
                              base_size, base_size)
        pygame.draw.rect(screen, (255, 50, 50), bg_rect)
    elif not rest_neutral_visual:
        pygame.draw.circle(screen, (0, 120, 255), (pos_x, pos_y), base_size // 2)

    draw_time_balls(time_ball_state, screen_width, screen_height,
                    mode="single",
                    indicator_color=(config.black if rest_neutral_visual else next_color),
                    single_pos=NEXT_INDICATOR_POS,
                    ball_radius=int(base_size * 0.4))

    # ── Texto ────────────────────────────────────────────────
    font_prep = pygame.font.SysFont(None, 72)
    mi_label = getattr(config, "ARDUINO_MI_LABEL", "Open")
    prep_msg  = f"Prepare: {mi_label} {config.ARM_SIDE.upper()} Hand" if is_mi else "Rest"
    if not rest_neutral_visual:
        txt_surface = font_prep.render(prep_msg, True, config.white)
        screen.blit(txt_surface,
                    (screen_width // 2 - txt_surface.get_width() // 2,
                     screen_height // 2 + 300))

    # ── Flecha direccional ───────────────────────────────────
    arrow_dir = "right" if is_mi else "left"
    if not rest_neutral_visual:
        draw_arrow_directional(screen, pos_x, pos_y,
                               base_size // 2.5, (255, 255, 255),
                               direction=arrow_dir)

    pygame.display.flip()
            
def show_feedback(duration, mode):
    """Feedback phase keeping the arrow for continuity."""
    start_time = time.time()
    pos_x, pos_y = int(screen_width * NEXT_INDICATOR_POS[0]), int(screen_height * NEXT_INDICATOR_POS[1])
    base_size = int(min(screen_width, screen_height) * 0.08 * NEXT_INDICATOR_SCALE)

    # --- CONTROL ARDUINO (AGREGADO) ---
    if arduino_ser and arduino_ser.is_open:
        try:
            command = glove_cmd_for_mode(mode)
            arduino_ser.write(command)
        except Exception as e:
            logger.log_event(f"Arduino Error: {e}")
    # ----------------------------------

    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        screen.fill(config.black)
        draw_fixation_cross(screen_width, screen_height)

        if mode == 0: # MI
            draw_arrow_fill(progress, screen_width, screen_height, False, fill_alpha=OFFLINE_FILL_ALPHA)
            # Maintain Visual Identity (Square)
            bg_rect = pygame.Rect(pos_x - base_size//2, pos_y - base_size//2, base_size, base_size)
            pygame.draw.rect(screen, (255, 50, 50), bg_rect)
            pygame.draw.rect(screen, (255, 50, 50), pygame.Rect(pos_x - int(base_size*0.35), 
                             pos_y - int(base_size*0.35), int(base_size*0.7), int(base_size*0.7)))
            # Keep Arrow
            draw_arrow_directional(screen, pos_x, pos_y, base_size // 2.5, (255, 255, 255), "right")
            mi_label = getattr(config, "ARDUINO_MI_LABEL", "Open")
            msg = f"{mi_label} {config.ARM_SIDE.upper()} Hand"

        else: # REST
            draw_ball_fill(progress, screen_width, screen_height, False, fill_alpha=OFFLINE_FILL_ALPHA)
            # Maintain Visual Identity (Circle)
            pygame.draw.circle(screen, (0, 120, 255), (pos_x, pos_y), base_size // 2)
            pygame.draw.circle(screen, (0, 120, 255), (pos_x, pos_y), int(base_size * 0.35))
            # Keep Arrow
            draw_arrow_directional(screen, pos_x, pos_y, base_size // 2.5, (255, 255, 255), "left")
            msg = "Rest"

        if msg:
            txt = pygame.font.SysFont(None, 96).render(msg, True, config.white)
            screen.blit(txt, (screen_width//2 - txt.get_width()//2, screen_height//2 + 300))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
    return True

# ============================================================
# MAIN LOOP
# ============================================================
logger.log_event("Resolving EEG stream...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])

trial_sequence = generate_trial_sequence(config.TOTAL_TRIALS, config.MAX_REPEATS)
current_trial = 0
clock = pygame.time.Clock()

display_fixation_period(duration=3)

try:
    while current_trial < len(trial_sequence):
        next_mode = trial_sequence[current_trial]
        next_color = (255, 50, 50) if next_mode == 0 else (0, 120, 255)

        # PRE-TRIAL
        countdown_start = pygame.time.get_ticks()
        backdoor_mode, waiting = None, True
        prep_sensory_fes_active = False

        if next_mode == 0 and FES_toggle:
            send_udp_message(
                fes_socket,
                config.UDP_FES["IP"],
                config.UDP_FES["PORT"],
                "FES_SENS_GO",
                logger,
            )
            prep_sensory_fes_active = True
            logger.log_event(
                "⚡ FES_SENS_GO — inicio de barra de preparación (2.5 s)"
            )

        while waiting:
            elapsed = pygame.time.get_ticks() - countdown_start
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT: backdoor_mode = 0; waiting = False
                    if event.key == pygame.K_DOWN:  backdoor_mode = 1; waiting = False
            if config.TIMING and elapsed >= 2500:
                waiting = False
            draw_pretrial_screen(next_color, time_ball_state=1,
                                 elapsed_ms=elapsed, total_ms=2500)
            clock.tick(60)

        if prep_sensory_fes_active:
            send_udp_message(
                fes_socket,
                config.UDP_FES["IP"],
                config.UDP_FES["PORT"],
                "FES_STOP",
                logger,
            )
            logger.log_event("⚡ FES_STOP — fin de barra de preparación")

        mode = backdoor_mode if backdoor_mode is not None else next_mode

        # EXECUTION
        trig = config.TRIGGERS["MI_BEGIN"] if mode == 0 else config.TRIGGERS["REST_BEGIN"]
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], trig, logger)
        if mode == 0 and FES_toggle:
            send_udp_message(
                fes_socket,
                config.UDP_FES["IP"],
                config.UDP_FES["PORT"],
                "FES_MOTOR_GO",
                logger,
            )
            logger.log_event("⚡ FES_MOTOR_GO — inicio del llenado rojo")

        feedback_completed = show_feedback(config.TIME_MI, mode)

        if mode == 0 and FES_toggle:
            send_udp_message(
                fes_socket,
                config.UDP_FES["IP"],
                config.UDP_FES["PORT"],
                "FES_STOP",
                logger,
            )
            logger.log_event("⚡ FES_STOP — fin del llenado rojo")

        if not feedback_completed:
            break

        # Return glove to configured baseline immediately after feedback.
        arduino_write(getattr(config, "ARDUINO_CMD_REST", b"1"))

        # END TRIAL / ROBOT
        end_trig = config.TRIGGERS["MI_END"] if mode == 0 else config.TRIGGERS["REST_END"]
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], end_trig, logger)

        if mode == 0:
            sel_traj = random.choice(config.ROBOT_TRAJECTORY)
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_BEGIN"], logger)
            display_multiple_messages_with_udp([" "], [config.green], [0], config.TIME_ROB, [sel_traj, "g"], udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], logger)
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_END"], logger)
            display_fixation_period(2)
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_HOME"], logger)
            send_udp_message(udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], config.ROBOT_OPCODES["HOME"], logger, expect_ack=True)
        else:
            display_multiple_messages_with_udp([" "], [config.white], [0], config.TIME_STATIONARY, None, udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], logger)

        # Baseline glove state between trials.
        arduino_write(getattr(config, "ARDUINO_CMD_REST", b"1"))

        send_udp_message(
            udp_socket_marker,
            config.UDP_MARKER["IP"],
            config.UDP_MARKER["PORT"],
            config.TRIGGERS["INTERTRIAL_BEGIN"],
            logger,
        )
        display_fixation_period(float(getattr(config, "INTERTRIAL_DURATION", 3.0)))
        send_udp_message(
            udp_socket_marker,
            config.UDP_MARKER["IP"],
            config.UDP_MARKER["PORT"],
            config.TRIGGERS["INTERTRIAL_END"],
            logger,
        )
        current_trial += 1

finally:
    # Final instruction after the offline experiment: leave the glove open.
    if arduino_ser and arduino_ser.is_open:
        arduino_write(glove_cmd_mi())
        logger.log_event("🤚 Glove opened at experiment end.")
        arduino_ser.close()
    pygame.quit()
    [s.close() for s in [udp_socket_marker, udp_socket_robot, fes_socket]]
