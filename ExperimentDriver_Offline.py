#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pygame
import socket
import sys
import time
from Utils.visualization import draw_arrow_fill, draw_ball_fill, draw_fixation_cross, draw_time_balls
from Utils.experiment_utils import generate_trial_sequence
from Utils.networking import send_udp_message, display_multiple_messages_with_udp
import config
from pylsl import StreamInlet, resolve_stream
from pathlib import Path
from Utils.logging_manager import LoggerManager
import random
import os


# ============================================================
# CONFIG: Next-trial indicator placement (resolution independent)
# Move Y_RATIO down to move indicator DOWN. Up to move UP.
# ============================================================
NEXT_INDICATOR_POS = (0.50, 0.28)   # (x_ratio, y_ratio)
NEXT_INDICATOR_SCALE = 1.00        # 1.0 = default size from draw_time_balls auto-scale


# Initialize UDP sockets
udp_socket_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_robot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
fes_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

FES_toggle = config.FES_toggle

# Auto-detect active recording (or fallback if none)
logger = LoggerManager.auto_detect_from_subject(
    subject=config.TRAINING_SUBJECT,
    base_path=Path(config.DATA_DIR)
)

# Log config snapshot
loggable_fields = [
    "UDP_MARKER", "UDP_ROBOT", "UDP_FES",
    "ARM_SIDE", "TOTAL_TRIALS", "MAX_REPEATS",
    "TIME_MI", "TIME_ROB", "TIME_STATIONARY",
    "SHAPE_MAX", "SHAPE_MIN", "ROBOT_TRAJECTORY",
    "FES_toggle", "FES_CHANNEL", "FES_TIMING_OFFSET",
    "WORKING_DIR", "DATA_DIR", "MODEL_PATH",
    "DATA_FILE_PATH", "TRAINING_SUBJECT"
]
config_log_subset = {key: getattr(config, key) for key in loggable_fields if hasattr(config, key)}
logger.save_config_snapshot(config_log_subset)

logger.log_event("Initialized offline EEG processing pipeline.")

pygame.init()

if config.BIG_BROTHER_MODE:
    # External display at +0+0 (HDMI-1), force window origin (0,0)
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    # Use desktop resolution for the HDMI screen if you want borderless fullscreen feel
    screen = pygame.display.set_mode((1920, 1080), pygame.NOFRAME)
    logger.log_event("🎥 Big Brother Mode ON — window placed at (0,0) on external monitor (HDMI-1).")
else:
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    logger.log_event("👤 Big Brother Mode OFF — fullscreen on active display.")

pygame.display.set_caption("EEG Offline Interactive Loop")
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h


def display_fixation_period(duration=3):
    """Displays fixation cross for a given duration (seconds)."""
    logger.log_event(f"Fixation period started for {duration} seconds.")
    start_time = time.time()
    clock = pygame.time.Clock()

    while time.time() - start_time < duration:
        pygame.display.get_surface().fill(config.black)

        draw_fixation_cross(screen_width, screen_height)
        draw_ball_fill(0, screen_width, screen_height, show_threshold=False)
        draw_arrow_fill(0, screen_width, screen_height, show_threshold=False)

        # Keep baseline indicator
        draw_time_balls(
            0, screen_width, screen_height,
            mode="single",
            single_pos=NEXT_INDICATOR_POS
        )

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                logger.log_event("Fixation interrupted — experiment manually terminated.")
                return

        clock.tick(60)

    logger.log_event("Fixation period complete.")


def draw_pretrial_screen(next_color, time_ball_state):
    """
    Draws the pre-trial screen with a next-trial indicator.
    This version does NOT overlay a separate circle; it delegates positioning to draw_time_balls().
    """
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)
    draw_arrow_fill(0, screen_width, screen_height, show_threshold=False)
    draw_ball_fill(0, screen_width, screen_height, show_threshold=False)

    # Draw the single indicator ball using the same anchor every time
    # indicator_color paints it MI/REST without needing an overlay.
    draw_time_balls(
        time_ball_state,
        screen_width,
        screen_height,
        mode="single",
        indicator_color=next_color,
        single_pos=NEXT_INDICATOR_POS,
        ball_radius=int(min(screen_width, screen_height) * 0.035 * NEXT_INDICATOR_SCALE)
    )

    pygame.display.flip()


def show_feedback(duration=5, mode=0):
    """Displays feedback animation for the specified duration."""
    logger.log_event(f"Feedback display started — Mode: {'MI' if mode == 0 else 'REST'}, Duration: {duration}s")
    start_time = time.time()

    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time
        progress = elapsed_time / duration

        screen.fill(config.black)
        if mode == 0:
            draw_arrow_fill(progress, screen_width, screen_height, show_threshold=False)
            draw_ball_fill(0, screen_width, screen_height, show_threshold=False)
            draw_fixation_cross(screen_width, screen_height)

            # During MI, show MI color ball (red)
            draw_time_balls(2, screen_width, screen_height, mode="single", single_pos=NEXT_INDICATOR_POS)

            message = pygame.font.SysFont(None, 96).render(f"Flex {config.ARM_SIDE.upper()} Hand", True, config.white)
        else:
            draw_ball_fill(progress, screen_width, screen_height, show_threshold=False)
            draw_arrow_fill(0, screen_width, screen_height, show_threshold=False)
            draw_fixation_cross(screen_width, screen_height)

            # During REST, show REST color ball (blue)
            draw_time_balls(3, screen_width, screen_height, mode="single", single_pos=NEXT_INDICATOR_POS)

            message = pygame.font.SysFont(None, 96).render("Rest", True, config.white)

        screen.blit(
            message,
            (screen_width // 2 - message.get_width() // 2,
             screen_height // 2 - message.get_height() // 2 + 300)
        )

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                logger.log_event("Feedback interrupted — experiment manually terminated.")
                return False

    logger.log_event("Feedback display complete.")
    return True


# ============================================================
# Main Game Loop
# ============================================================
logger.log_event("Attempting to resolve EEG stream...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
logger.log_event("EEG data stream detected. Starting experiment...")

trial_sequence = generate_trial_sequence(config.TOTAL_TRIALS, config.MAX_REPEATS)
logger.log_event(f"Trial Sequence: {trial_sequence}")
mode_labels = ["MI" if t == 0 else "REST" for t in trial_sequence]
logger.log_event(f"Trial Sequence (labeled): {mode_labels}")

current_trial = 0
running = True
clock = pygame.time.Clock()

display_fixation_period(duration=3)

try:
    while running and current_trial < len(trial_sequence):

        # === (A) NEXT TRIAL INDICATOR (programmed) ===
        next_mode = trial_sequence[current_trial]  # 0=MI, 1=REST
        next_color = (255, 0, 0) if next_mode == 0 else (0, 120, 255)

        # === (B) PRE-TRIAL SCREEN (initial frame) ===
        draw_pretrial_screen(next_color, time_ball_state=1)

        backdoor_mode = None
        waiting_for_press = True
        countdown_start = None
        countdown_duration = 1500  # ms

        while waiting_for_press:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting_for_press = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        backdoor_mode = 0  # MI
                    elif event.key == pygame.K_DOWN:
                        backdoor_mode = 1  # REST
                    waiting_for_press = False

            if config.TIMING:
                if countdown_start is None:
                    countdown_start = pygame.time.get_ticks()

                elapsed_time = pygame.time.get_ticks() - countdown_start

                # Redraw during countdown to keep the indicator visible
                draw_pretrial_screen(next_color, time_ball_state=1)

                if elapsed_time >= countdown_duration:
                    logger.log_event("Timing mode: Countdown expired, proceeding automatically.")
                    waiting_for_press = False

            clock.tick(60)

        if not running:
            break

        # === Determine actual mode ===
        # If backdoor is used, it overrides the programmed trial for THIS iteration.
        if backdoor_mode is not None:
            mode = backdoor_mode
            logger.log_event(f"Backdoor override used: {'MI' if mode == 0 else 'REST'}")
        else:
            mode = trial_sequence[current_trial]

        logger.log_event(f"Starting trial {current_trial+1}/{len(trial_sequence)} — Mode: {'MI' if mode == 0 else 'REST'}")

        # === Triggers ===
        if mode == 0:
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_BEGIN"], logger=logger)
            logger.log_event("Sent MI_BEGIN trigger.")

            if FES_toggle == 1:
                send_udp_message(fes_socket, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_SENS_GO", logger=logger)
                logger.log_event("FES sensory stimulation sent.")
        else:
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_BEGIN"], logger=logger)
            logger.log_event("Sent REST_BEGIN trigger.")

        # === Feedback ===
        if not show_feedback(duration=config.TIME_MI, mode=mode):
            break

        # === Post-feedback ===
        if mode == 0:
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_END"], logger=logger)
            logger.log_event("Sent MI_END trigger.")

            messages = [" "]
            selected_trajectory = random.choice(config.ROBOT_TRAJECTORY)
            udp_messages = [selected_trajectory, "g"]
            colors = [config.green]
            duration = config.TIME_ROB

            if FES_toggle == 1:
                send_udp_message(fes_socket, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_MOTOR_GO", logger=logger)
                logger.log_event("FES motor stimulation sent.")

            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_BEGIN"], logger=logger)
            logger.log_event(f"Sent ROBOT_BEGIN trigger with trajectory: {selected_trajectory}")

        else:
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_END"], logger=logger)
            logger.log_event("Sent REST_END trigger.")

            messages = [" "]
            udp_messages = None
            colors = [config.white]
            duration = config.TIME_STATIONARY

        display_multiple_messages_with_udp(
            messages=messages,
            colors=colors,
            offsets=[0],
            duration=duration,
            udp_messages=udp_messages,
            udp_socket=udp_socket_robot,
            udp_ip=config.UDP_ROBOT["IP"],
            udp_port=config.UDP_ROBOT["PORT"],
            logger=logger
        )

        if mode == 0:
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_END"], logger=logger)
            logger.log_event("Sent ROBOT_END trigger.")

            display_fixation_period(duration=2)

            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_HOME"], logger=logger)

            send_udp_message(
                udp_socket_robot,
                config.UDP_ROBOT["IP"],
                config.UDP_ROBOT["PORT"],
                config.ROBOT_OPCODES["HOME"],
                logger=logger,
                expect_ack=True,
                ack_timeout=1.0,
                max_retries=1
            )

        display_fixation_period(duration=3)
        current_trial += 1
        clock.tick(60)

finally:
    pygame.quit()
    udp_socket_marker.close()
    udp_socket_robot.close()
    fes_socket.close()
    logger.log_event("Experiment terminated (cleanup complete).")
