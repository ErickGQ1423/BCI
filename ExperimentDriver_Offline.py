#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pygame
import socket
import sys
import time
import random
import os
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

# Logging
logger = LoggerManager.auto_detect_from_subject(
    subject=config.TRAINING_SUBJECT,
    base_path=Path(config.DATA_DIR)
)

# Config snapshot
loggable_fields = ["UDP_MARKER", "UDP_ROBOT", "UDP_FES", "ARM_SIDE", "TOTAL_TRIALS", "TIME_MI", "FES_toggle"]
config_log_subset = {k: getattr(config, k) for k in loggable_fields if hasattr(config, k)}
logger.save_config_snapshot(config_log_subset)

pygame.init()

if config.BIG_BROTHER_MODE:
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    screen = pygame.display.set_mode((3840, 2160), pygame.NOFRAME)
else:
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

screen_width, screen_height = pygame.display.Info().current_w, pygame.display.Info().current_h

# ============================================================
# VISUAL FUNCTIONS
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

def display_fixation_period(duration=3):
    start_time = time.time()
    clock = pygame.time.Clock()
    while time.time() - start_time < duration:
        pygame.display.get_surface().fill(config.black)
        draw_fixation_cross(screen_width, screen_height)
        draw_ball_fill(0, screen_width, screen_height, show_threshold=False)
        draw_arrow_fill(0, screen_width, screen_height, show_threshold=False)
        draw_time_balls(0, screen_width, screen_height, mode="single", single_pos=NEXT_INDICATOR_POS)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
        clock.tick(60)

def draw_pretrial_screen(next_color, time_ball_state):
    """Pre-trial with Line+Triangle arrow."""
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)
    
    pos_x = int(screen_width * NEXT_INDICATOR_POS[0])
    pos_y = int(screen_height * NEXT_INDICATOR_POS[1])
    base_size = int(min(screen_width, screen_height) * 0.08 * NEXT_INDICATOR_SCALE)
    
    is_mi = (next_color == (255, 50, 50) or next_color == getattr(config, 'red', (255, 50, 50)))

    # 1. Outer Background (White)
    if is_mi:
        bg_rect = pygame.Rect(pos_x - base_size//2, pos_y - base_size//2, base_size, base_size)
        pygame.draw.rect(screen, (255, 50, 50), bg_rect)
    else:
        pygame.draw.circle(screen, (0, 120, 255), (pos_x, pos_y), base_size // 2)

    # 2. Middle Color
    draw_time_balls(time_ball_state, screen_width, screen_height, mode="single", 
                    indicator_color=next_color, single_pos=NEXT_INDICATOR_POS, ball_radius=int(base_size * 0.4))
    
    # ============================================================
    # AGREGAR TEXTO AQUÍ (NUEVA SECCIÓN)
    # ============================================================
    font_prep = pygame.font.SysFont(None, 72) # Tamaño un poco menor que el feedback para diferenciar
    if is_mi:
        prep_msg = f"Prepare: Flex {config.ARM_SIDE.upper()} Hand"
    else:
        prep_msg = "Prepare: Rest"
    
    txt_surface = font_prep.render(prep_msg, True, config.white)
    # Lo posicionamos en la misma coordenada X que el indicador, pero más abajo (ajusta +400 según necesites)
    screen.blit(txt_surface, (screen_width // 2 - txt_surface.get_width() // 2, screen_height // 2 + 300))
    # ============================================================

    # 3. Directional Arrow (Line + Triangle)
    arrow_dir = "right" if is_mi else "left"
    draw_arrow_directional(screen, pos_x, pos_y, base_size // 2.5, (255, 255, 255), direction=arrow_dir)
    
    pygame.display.flip()

def show_feedback(duration, mode):
    """Feedback phase keeping the arrow for continuity."""
    start_time = time.time()
    pos_x, pos_y = int(screen_width * NEXT_INDICATOR_POS[0]), int(screen_height * NEXT_INDICATOR_POS[1])
    base_size = int(min(screen_width, screen_height) * 0.08 * NEXT_INDICATOR_SCALE)

    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        screen.fill(config.black)
        draw_fixation_cross(screen_width, screen_height)

        if mode == 0: # MI
            draw_arrow_fill(progress, screen_width, screen_height, False)
            # Maintain Visual Identity (Square)
            bg_rect = pygame.Rect(pos_x - base_size//2, pos_y - base_size//2, base_size, base_size)
            pygame.draw.rect(screen, (255, 50, 50), bg_rect)
            pygame.draw.rect(screen, (255, 50, 50), pygame.Rect(pos_x - int(base_size*0.35), 
                             pos_y - int(base_size*0.35), int(base_size*0.7), int(base_size*0.7)))
            # Keep Arrow
            draw_arrow_directional(screen, pos_x, pos_y, base_size // 2.5, (255, 255, 255), "right")
            msg = f"Flex {config.ARM_SIDE.upper()} Hand"
        else: # REST
            draw_ball_fill(progress, screen_width, screen_height, False)
            # Maintain Visual Identity (Circle)
            pygame.draw.circle(screen, (0, 120, 255), (pos_x, pos_y), base_size // 2)
            pygame.draw.circle(screen, (0, 120, 255), (pos_x, pos_y), int(base_size * 0.35))
            # Keep Arrow
            draw_arrow_directional(screen, pos_x, pos_y, base_size // 2.5, (255, 255, 255), "left")
            msg = "Rest"

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
        draw_pretrial_screen(next_color, time_ball_state=1)
        
        backdoor_mode, waiting = None, True
        countdown_start = pygame.time.get_ticks()
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT: backdoor_mode = 0; waiting = False
                    if event.key == pygame.K_DOWN: backdoor_mode = 1; waiting = False
            if config.TIMING and (pygame.time.get_ticks() - countdown_start >= 1500):
                waiting = False
            draw_pretrial_screen(next_color, time_ball_state=1)
            clock.tick(60)

        mode = backdoor_mode if backdoor_mode is not None else next_mode

        # EXECUTION
        trig = config.TRIGGERS["MI_BEGIN"] if mode == 0 else config.TRIGGERS["REST_BEGIN"]
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], trig, logger)
        if mode == 0 and FES_toggle: send_udp_message(fes_socket, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_SENS_GO", logger)

        if not show_feedback(config.TIME_MI, mode): break

        # END TRIAL / ROBOT
        end_trig = config.TRIGGERS["MI_END"] if mode == 0 else config.TRIGGERS["REST_END"]
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], end_trig, logger)

        if mode == 0:
            sel_traj = random.choice(config.ROBOT_TRAJECTORY)
            if FES_toggle: send_udp_message(fes_socket, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_MOTOR_GO", logger)
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_BEGIN"], logger)
            display_multiple_messages_with_udp([" "], [config.green], [0], config.TIME_ROB, [sel_traj, "g"], udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], logger)
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_END"], logger)
            display_fixation_period(2)
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_HOME"], logger)
            send_udp_message(udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], config.ROBOT_OPCODES["HOME"], logger, expect_ack=True)
        else:
            display_multiple_messages_with_udp([" "], [config.white], [0], config.TIME_STATIONARY, None, udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], logger)

        display_fixation_period(3)
        current_trial += 1

finally:
    pygame.quit()
    [s.close() for s in [udp_socket_marker, udp_socket_robot, fes_socket]]