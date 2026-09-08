# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import pygame
# import socket
# import pickle
# import datetime
# import os
# import random
# import time
# import serial
# from pylsl import StreamInlet, resolve_stream

# import mne
# mne.set_log_level("WARNING")

# from Utils.visualization import (
#     draw_arrow_fill, draw_ball_fill, draw_fixation_cross, 
#     draw_time_balls, draw_progress_bar
# )
# from Utils.experiment_utils import (
#     generate_trial_sequence, save_transform, load_transform
# )
# from Utils.EEGStreamState import EEGStreamState
# from Utils.networking import send_udp_message, display_multiple_messages_with_udp
# import config
# from pathlib import Path
# from Utils.logging_manager import LoggerManager

# # Import runtime_common
# from Utils.runtime_common import (
#     log_confusion_matrix_from_trial_summary,
#     append_trial_probabilities_to_csv,
#     display_fixation_period,
#     hold_messages_and_classify,
#     show_feedback,
# )
# import Utils.runtime_common as _RC

# # ============================================================
# # LOGGING & CONFIG
# # ============================================================
# logger = LoggerManager.auto_detect_from_subject(
#     subject=config.TRAINING_SUBJECT,
#     base_path=Path(config.DATA_DIR),
#     mode="online"
# )
# # Log config snapshot
# loggable_fields = [
#     "UDP_MARKER", "UDP_ROBOT", "UDP_FES", "ARM_SIDE", "TOTAL_TRIALS", 
#     "TIME_MI", "FES_toggle", "TRAINING_SUBJECT"
# ]
# config_log_subset = {k: getattr(config, k) for k in loggable_fields if hasattr(config, k)}
# logger.save_config_snapshot(config_log_subset)

# eeg_dir = logger.log_base / "eeg"
# adaptive_T_path = eeg_dir / "adaptive_T.pkl"

# Prev_T, counter = load_transform(adaptive_T_path)
# if Prev_T is None:
#     counter = 0
#     logger.log_event("ℹ️ No adaptive transform found — starting fresh.")
# else:
#     logger.log_event(f"✅ Loaded adaptive transform with counter = {counter}")

# pygame.init()

# # 1. Obtenemos la resolución actual del monitor ANTES de crear la ventana
# info_monitor = pygame.display.Info()
# monitor_w = info_monitor.current_w
# monitor_h = info_monitor.current_h

# if config.BIG_BROTHER_MODE:
#     os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
#     #   screen = pygame.display.set_mode((1920, 1080), pygame.NOFRAME)
#     screen = pygame.display.set_mode((monitor_w, monitor_h), pygame.FULLSCREEN | pygame.NOFRAME)
#     screen_width = 1920
#     screen_height = 1080
# else:
#     # 2. Forzamos la posición a la esquina superior izquierda
#     os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    
#     # 3. Creamos una ventana SIN BORDES con el tamaño exacto que detectamos
#     # Esto evita que Ubuntu intente escalar la ventana y la recorte
#     screen = pygame.display.set_mode((monitor_w, monitor_h), pygame.NOFRAME)
#     screen_width = monitor_w
#     screen_height = monitor_h

# pygame.display.set_caption("EEG Online Interactive Loop")
# info = pygame.display.Info()
# screen_width = info.current_w
# screen_height = info.current_h

# # UDP Settings
# udp_socket_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# udp_socket_robot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# udp_socket_fes = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# FES_toggle = config.FES_toggle

# # ============================================================
# # ARDUINO SETUP
# # ============================================================#!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import pygame
# import socket
# import pickle
# import datetime
# import os
# import random
# import time
# import serial
# from pylsl import StreamInlet, resolve_stream

# import mne
# mne.set_log_level("WARNING")

# from Utils.visualization import (
#     draw_arrow_fill, draw_ball_fill, draw_fixation_cross, 
#     draw_time_balls, draw_progress_bar
# )
# from Utils.experiment_utils import (
#     generate_trial_sequence, save_transform, load_transform
# )
# from Utils.EEGStreamState import EEGStreamState
# from Utils.networking import send_udp_message, display_multiple_messages_with_udp
# import config
# from pathlib import Path
# from Utils.logging_manager import LoggerManager

# # Import runtime_common
# from Utils.runtime_common import (
#     log_confusion_matrix_from_trial_summary,
#     append_trial_probabilities_to_csv,
#     display_fixation_period,
#     hold_messages_and_classify,
#     show_feedback,
# )
# import Utils.runtime_common as _RC

# # ============================================================
# # LOGGING & CONFIG
# # ============================================================
# logger = LoggerManager.auto_detect_from_subject(
#     subject=config.TRAINING_SUBJECT,
#     base_path=Path(config.DATA_DIR),
#     mode="online"
# )
# # Log config snapshot
# loggable_fields = [
#     "UDP_MARKER", "UDP_ROBOT", "UDP_FES", "ARM_SIDE", "TOTAL_TRIALS", 
#     "TIME_MI", "FES_toggle", "TRAINING_SUBJECT"
# ]
# config_log_subset = {k: getattr(config, k) for k in loggable_fields if hasattr(config, k)}
# logger.save_config_snapshot(config_log_subset)

# eeg_dir = logger.log_base / "eeg"
# adaptive_T_path = eeg_dir / "adaptive_T.pkl"

# Prev_T, counter = load_transform(adaptive_T_path)
# if Prev_T is None:
#     counter = 0
#     logger.log_event("ℹ️ No adaptive transform found — starting fresh.")
# else:
#     logger.log_event(f"✅ Loaded adaptive transform with counter = {counter}")

# pygame.init()

# # 1. Obtenemos la resolución actual del monitor ANTES de crear la ventana
# info_monitor = pygame.display.Info()
# monitor_w = info_monitor.current_w
# monitor_h = info_monitor.current_h

# if config.BIG_BROTHER_MODE:
#     os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
#     #   screen = pygame.display.set_mode((1920, 1080), pygame.NOFRAME)
#     screen = pygame.display.set_mode((monitor_w, monitor_h), pygame.FULLSCREEN | pygame.NOFRAME)
#     screen_width = 1920
#     screen_height = 1080
# else:
#     # 2. Forzamos la posición a la esquina superior izquierda
#     os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    
#     # 3. Creamos una ventana SIN BORDES con el tamaño exacto que detectamos
#     # Esto evita que Ubuntu intente escalar la ventana y la recorte
#     screen = pygame.display.set_mode((monitor_w, monitor_h), pygame.NOFRAME)
#     screen_width = monitor_w
#     screen_height = monitor_h

# pygame.display.set_caption("EEG Online Interactive Loop")
# info = pygame.display.Info()
# screen_width = info.current_w
# screen_height = info.current_h

# # UDP Settings
# udp_socket_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# udp_socket_robot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# udp_socket_fes = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# FES_toggle = config.FES_toggle

# # ============================================================
# # ARDUINO SETUP
# # ============================================================
# ARDUINO_PORT = os.environ.get("ARDUINO_PORT", "")
# ARDUINO_BAUD = int(os.environ.get("ARDUINO_BAUD", 9600))
# arduino = None

# if ARDUINO_PORT:
#     try:
#         logger.log_event(f"Connecting to Glove (Arduino) on {ARDUINO_PORT}...")
#         arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.1)
#         time.sleep(2)  # CRITICAL: Safety wait for Arduino reset
#         logger.log_event("✅ Glove connected successfully.")
#     except Exception as e:
#         logger.log_event(f"❌ Error connecting to Glove: {e}", level="error")
#         arduino = None
# else:
#     logger.log_event("ℹ️ No Arduino port configured.")

# # Load Model
# subject_model_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
# subject_model_path = os.path.join(subject_model_dir, f"sub-{config.TRAINING_SUBJECT}_model.pkl")

# try:
#     with open(subject_model_path, 'rb') as f:
#         model = pickle.load(f)
#     logger.log_event(f"✅ Model loaded: {subject_model_path}")
# except FileNotFoundError:
#     logger.log_event(f"❌ Model not found: {subject_model_path}", level="error")
#     sys.exit(1)

# # Wire runtime objects
# _RC.config = config
# _RC.logger = logger
# _RC.model = model
# _RC.screen = screen
# _RC.screen_width = screen_width
# _RC.screen_height = screen_height
# _RC.udp_socket_marker = udp_socket_marker
# _RC.udp_socket_robot  = udp_socket_robot
# _RC.udp_socket_fes    = udp_socket_fes
# _RC.FES_toggle = FES_toggle
# _RC.Prev_T = Prev_T
# _RC.counter = counter

# # NOTE: We do not pass '_RC.arduino' because runtime_common 
# # will not handle the glove. The glove is handled by this main script.

# def main():
#     logger.log_event("Resolving EEG data stream via LSL...")
#     streams = resolve_stream('type', 'EEG')
#     inlet = StreamInlet(streams[0])
#     eeg_state = EEGStreamState(inlet=inlet, config=config, logger=logger)
    
#     trial_sequence = generate_trial_sequence(total_trials=config.TOTAL_TRIALS, max_repeats=config.MAX_REPEATS)
#     current_trial = 0
#     running = True
#     clock = pygame.time.Clock()

#     display_fixation_period(duration=3, eeg_state=eeg_state)

#     # Ensure glove is open at start
#     if arduino: arduino_write(b'0')

#     while running and current_trial < len(trial_sequence):
#         logger.log_event(f"--- Trial {current_trial+1}/{len(trial_sequence)} START ---")

#         # 1. Obtenemos el modo AQUÍ ARRIBA (para saber qué texto de preparación poner)
#         mode = trial_sequence[current_trial] 

#         # 2. UI Setup (FASE DE PREPARACIÓN)
#         screen.fill(config.black)
#         draw_fixation_cross(screen_width, screen_height)
#         draw_arrow_fill(0, screen_width, screen_height)
#         draw_ball_fill(0, screen_width, screen_height)
#         draw_time_balls(0, screen_width, screen_height)
        
#         # ========================================================
#         # [NUEVO] TEXTO DE PREPARACIÓN DIRECTO EN EL DRIVER
#         # ========================================================
#         font_prep = pygame.font.SysFont(None, 96)
#         if mode == 0: # Preparando Imaginación Motora
#             prep_msg = f"Prepare: Imagine closing {config.ARM_SIDE.upper()} hand"
#             color_msg = (255, 255, 255)  # Amarillo clarito para diferenciar de la ejecución
#         else: # Preparando Descanso
#             prep_msg = "Prepare: Rest"
#             color_msg = (255, 255, 255)  # Azul clarito
            
#         txt_surface = font_prep.render(prep_msg, True, color_msg)
#         # Lo centramos en X, y lo ponemos abajo en Y
#         screen.blit(txt_surface, (screen_width // 2 - txt_surface.get_width() // 2, screen_height // 2 + 300))
#         # ========================================================

#         pygame.display.flip()

#         # 3. Waiting / Countdown
#         waiting_for_press = True
#         countdown_start = None
#         countdown_duration = 3000

#         while waiting_for_press:
#             eeg_state.update()
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     running = False; waiting_for_press = False
#                 elif event.type == pygame.KEYDOWN:
#                     if event.key == pygame.K_SPACE: waiting_for_press = False
            
#             if config.TIMING:
#                 if countdown_start is None: countdown_start = pygame.time.get_ticks()
#                 elapsed = pygame.time.get_ticks() - countdown_start
#                 draw_time_balls(1, screen_width, screen_height)
#                 pygame.display.flip()
#                 if elapsed >= countdown_duration: waiting_for_press = False

#         if not running: break

#         mode = trial_sequence[current_trial]
        
#         # 3. Baseline
#         try:
#             eeg_state.compute_baseline(duration_sec=config.BASELINE_DURATION)
#         except ValueError:
#             continue

#         # -----------------------------------------------------------
#         # PHASE 1: EFFORT (Sensory FES Only)
#         # -----------------------------------------------------------
#         # show_feedback handles the bar and Sensory FES (tingling)
#         prediction, confidence, leaky_integrator, trial_probs, earlystop_flag = show_feedback(
#             duration=config.TIME_MI,
#             mode=mode,
#             eeg_state=eeg_state
#         )

#         append_trial_probabilities_to_csv(
#             trial_probabilities=trial_probs, mode=mode, trial_number=current_trial + 1,
#             predicted_label=prediction, early_cutout=earlystop_flag,
#             mi_threshold=config.THRESHOLD_MI, rest_threshold=config.THRESHOLD_REST,
#             logger=logger, phase="MI" if mode == 0 else "REST"
#         )

#         # -----------------------------------------------------------
#         # PHASE 2: REWARD (Motor FES + Glove + Robot)
#         # -----------------------------------------------------------
#         if mode == 0: # MI Trial
#             if prediction == 200: # SUCCESS! (Threshold reached)
                
#                 # 1. CLOSE GLOVE (Reward Trigger)
#                 if arduino: 
#                     arduino_write(b'1')
#                     logger.log_event("✅ Prediction Success -> Closing Glove (Reward)")

#                 # 2. MOTOR FES
#                 if FES_toggle:
#                     send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_MOTOR_GO", logger=logger)
                
#                 # 3. ROBOT
#                 messages = ["Correct", "Hand close"]
#                 colors = [config.green, config.green]
#                 send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_BEGIN"], logger=logger)
                
#                 # Display message and robot (glove remains closed during this time)
#                 display_multiple_messages_with_udp(
#                     messages=messages, colors=colors, offsets=[-100, 100], duration=0.01,
#                     udp_messages=[random.choice(config.ROBOT_TRAJECTORY), config.ROBOT_OPCODES["GO"]],
#                     udp_socket=udp_socket_robot, udp_ip=config.UDP_ROBOT["IP"], udp_port=config.UDP_ROBOT["PORT"],
#                     logger=logger, eeg_state=eeg_state
#                 )
                
#                 # Maintain state (Glove closed) while robot moves (TIME_ROB)
#                 final_class, robot_probs, early = hold_messages_and_classify(
#                     messages, colors, [-100, 100], config.TIME_ROB, 0,
#                     udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"],
#                     eeg_state, leaky_integrator
#                 )
                
#                 # Robot home
#                 send_udp_message(udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], config.ROBOT_OPCODES["HOME"], logger=logger, expect_ack=True)
                
#             else: # FAIL (Threshold not reached)
#                 # Glove remains open
#                 if arduino: arduino_write(b'0')
#                 display_multiple_messages_with_udp(["Incorrect", "Hand Stationary"], [config.red, config.white], [-100, 100], config.TIME_STATIONARY, None, udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], logger, eeg_state)

#         else: # REST Trial
#             msg_txt = "Correct" if prediction == 100 else "Incorrect"
#             col = config.green if prediction == 100 else config.red
#             # Ensure glove is open
#             if arduino: arduino_write(b'0')
#             display_multiple_messages_with_udp([msg_txt, "Hand Stationary"], [col, config.white], [-100, 100], config.TIME_STATIONARY, None, udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], logger, eeg_state)

#         # -----------------------------------------------------------
#         # PHASE 3: RELAXATION (End of Trial)
#         # -----------------------------------------------------------
#         # Open glove for the next trial
#         if arduino: arduino_write(b'0')

#         display_fixation_period(duration=3, eeg_state=eeg_state)
#         current_trial += 1

#     # Cleanup
#     if current_trial == len(trial_sequence) and config.SAVE_ADAPTIVE_T:
#         save_transform(Prev_T, counter, adaptive_T_path)

#     log_confusion_matrix_from_trial_summary(logger)
    
#     if arduino: 
#         arduino_write(b'0')
#         arduino.close()
        
#     pygame.quit()

# if __name__ == "__main__":
#     main()

# ARDUINO_PORT = os.environ.get("ARDUINO_PORT", "")
# ARDUINO_BAUD = int(os.environ.get("ARDUINO_BAUD", 9600))
# arduino = None

# if ARDUINO_PORT:
#     try:
#         logger.log_event(f"Connecting to Glove (Arduino) on {ARDUINO_PORT}...")
#         arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.1)
#         time.sleep(2)  # CRITICAL: Safety wait for Arduino reset
#         logger.log_event("✅ Glove connected successfully.")
#     except Exception as e:
#         logger.log_event(f"❌ Error connecting to Glove: {e}", level="error")
#         arduino = None
# else:
#     logger.log_event("ℹ️ No Arduino port configured.")

# # Load Model
# subject_model_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
# subject_model_path = os.path.join(subject_model_dir, f"sub-{config.TRAINING_SUBJECT}_model.pkl")

# try:
#     with open(subject_model_path, 'rb') as f:
#         model = pickle.load(f)
#     logger.log_event(f"✅ Model loaded: {subject_model_path}")
# except FileNotFoundError:
#     logger.log_event(f"❌ Model not found: {subject_model_path}", level="error")
#     sys.exit(1)

# # Wire runtime objects
# _RC.config = config
# _RC.logger = logger
# _RC.model = model
# _RC.screen = screen
# _RC.screen_width = screen_width
# _RC.screen_height = screen_height
# _RC.udp_socket_marker = udp_socket_marker
# _RC.udp_socket_robot  = udp_socket_robot
# _RC.udp_socket_fes    = udp_socket_fes
# _RC.FES_toggle = FES_toggle
# _RC.Prev_T = Prev_T
# _RC.counter = counter

# # NOTE: We do not pass '_RC.arduino' because runtime_common 
# # will not handle the glove. The glove is handled by this main script.

# def main():
#     logger.log_event("Resolving EEG data stream via LSL...")
#     streams = resolve_stream('type', 'EEG')
#     inlet = StreamInlet(streams[0])
#     eeg_state = EEGStreamState(inlet=inlet, config=config, logger=logger)
    
#     trial_sequence = generate_trial_sequence(total_trials=config.TOTAL_TRIALS, max_repeats=config.MAX_REPEATS)
#     current_trial = 0
#     running = True
#     clock = pygame.time.Clock()

#     display_fixation_period(duration=3, eeg_state=eeg_state)

#     # Ensure glove is open at start
#     if arduino: arduino_write(b'0')

#     while running and current_trial < len(trial_sequence):
#         logger.log_event(f"--- Trial {current_trial+1}/{len(trial_sequence)} START ---")

#         # 1. Obtenemos el modo AQUÍ ARRIBA (para saber qué texto de preparación poner)
#         mode = trial_sequence[current_trial] 

#         # 2. UI Setup (FASE DE PREPARACIÓN)
#         screen.fill(config.black)
#         draw_fixation_cross(screen_width, screen_height)
#         draw_arrow_fill(0, screen_width, screen_height)
#         draw_ball_fill(0, screen_width, screen_height)
#         draw_time_balls(0, screen_width, screen_height)
        
#         # ========================================================
#         # [NUEVO] TEXTO DE PREPARACIÓN DIRECTO EN EL DRIVER
#         # ========================================================
#         font_prep = pygame.font.SysFont(None, 96)
#         if mode == 0: # Preparando Imaginación Motora
#             prep_msg = f"Prepare: Imagine closing {config.ARM_SIDE.upper()} hand"
#             color_msg = (255, 255, 255)  # Amarillo clarito para diferenciar de la ejecución
#         else: # Preparando Descanso
#             prep_msg = "Prepare: Rest"
#             color_msg = (255, 255, 255)  # Azul clarito
            
#         txt_surface = font_prep.render(prep_msg, True, color_msg)
#         # Lo centramos en X, y lo ponemos abajo en Y
#         screen.blit(txt_surface, (screen_width // 2 - txt_surface.get_width() // 2, screen_height // 2 + 300))
#         # ========================================================

#         pygame.display.flip()

#         # 3. Waiting / Countdown
#         waiting_for_press = True
#         countdown_start = None
#         countdown_duration = 3000

#         while waiting_for_press:
#             eeg_state.update()
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     running = False; waiting_for_press = False
#                 elif event.type == pygame.KEYDOWN:
#                     if event.key == pygame.K_SPACE: waiting_for_press = False
            
#             if config.TIMING:
#                 if countdown_start is None: countdown_start = pygame.time.get_ticks()
#                 elapsed = pygame.time.get_ticks() - countdown_start
#                 draw_time_balls(1, screen_width, screen_height)
#                 pygame.display.flip()
#                 if elapsed >= countdown_duration: waiting_for_press = False

#         if not running: break

#         mode = trial_sequence[current_trial]
        
#         # 3. Baseline
#         try:
#             eeg_state.compute_baseline(duration_sec=config.BASELINE_DURATION)
#         except ValueError:
#             continue

#         # -----------------------------------------------------------
#         # PHASE 1: EFFORT (Sensory FES Only)
#         # -----------------------------------------------------------
#         # show_feedback handles the bar and Sensory FES (tingling)
#         prediction, confidence, leaky_integrator, trial_probs, earlystop_flag = show_feedback(
#             duration=config.TIME_MI,
#             mode=mode,
#             eeg_state=eeg_state
#         )

#         append_trial_probabilities_to_csv(
#             trial_probabilities=trial_probs, mode=mode, trial_number=current_trial + 1,
#             predicted_label=prediction, early_cutout=earlystop_flag,
#             mi_threshold=config.THRESHOLD_MI, rest_threshold=config.THRESHOLD_REST,
#             logger=logger, phase="MI" if mode == 0 else "REST"
#         )

#         # -----------------------------------------------------------
#         # PHASE 2: REWARD (Motor FES + Glove + Robot)
#         # -----------------------------------------------------------
#         if mode == 0: # MI Trial
#             if prediction == 200: # SUCCESS! (Threshold reached)
                
#                 # 1. CLOSE GLOVE (Reward Trigger)
#                 if arduino: 
#                     arduino_write(b'1')
#                     logger.log_event("✅ Prediction Success -> Closing Glove (Reward)")

#                 # 2. MOTOR FES
#                 if FES_toggle:
#                     send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_MOTOR_GO", logger=logger)
                
#                 # 3. ROBOT
#                 messages = ["Correct", "Hand close"]
#                 colors = [config.green, config.green]
#                 send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_BEGIN"], logger=logger)
                
#                 # Display message and robot (glove remains closed during this time)
#                 display_multiple_messages_with_udp(
#                     messages=messages, colors=colors, offsets=[-100, 100], duration=0.01,
#                     udp_messages=[random.choice(config.ROBOT_TRAJECTORY), config.ROBOT_OPCODES["GO"]],
#                     udp_socket=udp_socket_robot, udp_ip=config.UDP_ROBOT["IP"], udp_port=config.UDP_ROBOT["PORT"],
#                     logger=logger, eeg_state=eeg_state
#                 )
                
#                 # Maintain state (Glove closed) while robot moves (TIME_ROB)
#                 final_class, robot_probs, early = hold_messages_and_classify(
#                     messages, colors, [-100, 100], config.TIME_ROB, 0,
#                     udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"],
#                     eeg_state, leaky_integrator
#                 )
                
#                 # Robot home
#                 send_udp_message(udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], config.ROBOT_OPCODES["HOME"], logger=logger, expect_ack=True)
                
#             else: # FAIL (Threshold not reached)
#                 # Glove remains open
#                 if arduino: arduino_write(b'0')
#                 display_multiple_messages_with_udp(["Incorrect", "Hand Stationary"], [config.red, config.white], [-100, 100], config.TIME_STATIONARY, None, udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], logger, eeg_state)

#         else: # REST Trial
#             msg_txt = "Correct" if prediction == 100 else "Incorrect"
#             col = config.green if prediction == 100 else config.red
#             # Ensure glove is open
#             if arduino: arduino_write(b'0')
#             display_multiple_messages_with_udp([msg_txt, "Hand Stationary"], [col, config.white], [-100, 100], config.TIME_STATIONARY, None, udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], logger, eeg_state)

#         # -----------------------------------------------------------
#         # PHASE 3: RELAXATION (End of Trial)
#         # -----------------------------------------------------------
#         # Open glove for the next trial
#         if arduino: arduino_write(b'0')

#         display_fixation_period(duration=3, eeg_state=eeg_state)
#         current_trial += 1

#     # Cleanup
#     if current_trial == len(trial_sequence) and config.SAVE_ADAPTIVE_T:
#         save_transform(Prev_T, counter, adaptive_T_path)

#     log_confusion_matrix_from_trial_summary(logger)
    
#     if arduino: 
#         arduino_write(b'0')
#         arduino.close()
        
#     pygame.quit()

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pygame
import socket
import pickle
import datetime
import csv
import hashlib
import os
import random
import re
import time
import serial
import sys  # ✅ faltaba (lo usas en sys.exit)
import numpy as np
from pylsl import StreamInlet, resolve_stream
from sklearn.metrics import roc_auc_score

import bci_runtime_env
import mne
mne.set_log_level("WARNING")

from Utils.visualization import (
    draw_arrow_fill, draw_ball_fill, draw_fixation_cross,
    draw_time_balls, draw_progress_bar
)
from Utils.experiment_utils import (
    generate_trial_sequence, save_transform, load_transform
)
from Utils.EEGStreamState import EEGStreamState
from Utils.networking import send_udp_message, display_multiple_messages_with_udp
import config
from pathlib import Path
from Utils.logging_manager import LoggerManager

# Import runtime_common
from Utils.runtime_common import (
    log_confusion_matrix_from_trial_summary,
    append_trial_probabilities_to_csv,
    display_fixation_period,
    hold_messages_and_classify,
    show_feedback,
)
import Utils.runtime_common as _RC


# ============================================================
# LOGGING & CONFIG
# ============================================================
recording_subject = getattr(config, "RECORDING_SUBJECT", config.TRAINING_SUBJECT)
recording_data_dir = Path(getattr(config, "RECORDING_DATA_DIR", config.DATA_DIR))
logger = LoggerManager.auto_detect_from_subject(
    subject=recording_subject,
    base_path=recording_data_dir,
    mode="online"
)
# Log config snapshot
loggable_fields = [
    "UDP_MARKER", "UDP_ROBOT", "UDP_FES", "ARM_SIDE", "TOTAL_TRIALS",
    "TIME_MI", "FES_toggle", "TRAINING_SUBJECT", "DATA_DIR",
    "RECORDING_SUBJECT", "RECORDING_DATA_DIR", "PREP_DECODER_MODE",
    "PREP_CONTROL_MODEL", "WARMUP_OBSERVER_ENABLED", "WARMUP_MODEL_PATH",
    "SHADOW_MODEL_ANALYSIS_ENABLED", "ENDPOINT_VALIDATION_ENABLED",
    "ONLINE_PREP_FEEDBACK_FILL_ALPHA", "ONLINE_EXEC_FEEDBACK_FILL_ALPHA",
    "FS", "LOWCUT", "HIGHCUT", "MOTOR_CHANNEL_NAMES",
    "EEG_STREAM_MAX_AGE_S", "ONLINE_MODEL_PATH",
    "RECENTERING", "RECENTERING_ALPHA", "RECENTERING_MIN_TRIALS",
    "RECENTERING_REQUIRE_NON_AMBIGUOUS", "RECENTERING_REQUIRE_CORRECT",
    "RECENTERING_MIN_CONFIDENCE", "M2_INIT_RECENTER_FROM_TRAINING",
    "M2_USE_SAVED_ADAPTIVE_RECENTER", "ADAPTIVE_CONTINUITY_MODE",
    "ADAPTIVE_RECENTER_LOAD_PATH", "SAVE_ADAPTIVE_T",
    "THRESHOLD_MI", "THRESHOLD_REST", "ENDPOINT_MDM_MI_THRESHOLD",
    "ENDPOINT_MDM_REST_THRESHOLD", "PREP_CONTROL_ENDPOINT",
    "MIN_PREDICTIONS", "EARLYSTOP_CONSECUTIVE_PREDICTIONS",
    "INTEGRATOR_ALPHA",
    "EEG_QUALITY_GATE", "EEG_QUALITY_MIN_PTP_UV", "EEG_QUALITY_MAX_PTP_UV",
    "EEG_QUALITY_MAX_RMS_UV", "EEG_QUALITY_MAX_ABS_UV"
]
config_log_subset = {k: getattr(config, k) for k in loggable_fields if hasattr(config, k)}
logger.save_config_snapshot(config_log_subset)

eeg_dir = logger.log_base / "eeg"
adaptive_T_path = eeg_dir / "adaptive_T.pkl"

def _previous_same_condition_adaptive_path(current_session_dir: Path) -> Path | None:
    """Find the latest previous session with the same condition and adaptive_T.pkl.

    Example:
        ses-S002_ONLINE_FES -> ses-S001_ONLINE_FES/eeg/adaptive_T.pkl

    This keeps NoFES and FES adaptive states separated while allowing day-to-day
    continuity without hardcoding full paths per subject.
    """
    session_name = current_session_dir.name
    match = re.match(r"ses-S(\d+)_(.+)$", session_name)
    if not match:
        return None

    current_idx = int(match.group(1))
    condition_suffix = match.group(2)
    subject_dir = current_session_dir.parent

    candidates = []
    for session_dir in subject_dir.glob(f"ses-S*_{condition_suffix}"):
        candidate_match = re.match(r"ses-S(\d+)_(.+)$", session_dir.name)
        if not candidate_match:
            continue
        candidate_idx = int(candidate_match.group(1))
        if candidate_idx >= current_idx:
            continue
        candidate_path = session_dir / "eeg" / "adaptive_T.pkl"
        if candidate_path.exists():
            candidates.append((candidate_idx, candidate_path))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


adaptive_load_path = adaptive_T_path
manual_adaptive_load_path = getattr(config, "ADAPTIVE_RECENTER_LOAD_PATH", None)
adaptive_continuity_mode = getattr(config, "ADAPTIVE_CONTINUITY_MODE", "current_session")

if manual_adaptive_load_path:
    adaptive_load_path = Path(manual_adaptive_load_path)
    logger.log_event(f"🧭 Adaptive load path override: {adaptive_load_path}")
elif adaptive_continuity_mode == "fresh":
    adaptive_load_path = None
    logger.log_event("🧭 Adaptive continuity mode: fresh — not loading saved adaptive transform.")
elif adaptive_continuity_mode == "previous_same_condition" and not adaptive_T_path.exists():
    previous_path = _previous_same_condition_adaptive_path(logger.log_base)
    if previous_path is not None:
        adaptive_load_path = previous_path
        logger.log_event(f"🧭 Adaptive continuity: loading previous same-condition state from {adaptive_load_path}")
    else:
        logger.log_event("🧭 Adaptive continuity: no previous same-condition adaptive_T.pkl found.")

Prev_T, counter = load_transform(adaptive_load_path) if adaptive_load_path is not None else (None, 0)
if Prev_T is None:
    counter = 0
    logger.log_event("ℹ️ No adaptive transform found — starting fresh.")
else:
    logger.log_event(f"✅ Loaded adaptive transform with counter = {counter}")

pygame.init()

# 1) Resolución actual del monitor ANTES de crear la ventana
info_monitor = pygame.display.Info()
monitor_w = info_monitor.current_w
monitor_h = info_monitor.current_h

if config.BIG_BROTHER_MODE:
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    screen = pygame.display.set_mode((3840, 2160),pygame.NOFRAME)
    # Si tú quieres forzar 1920x1080 aquí, lo puedes hacer,
    # pero para que el indicador se vea proporcional, lo dejamos dinámico:
    screen_width = monitor_w
    screen_height = monitor_h
else:
    os.environ["SDL_VIDEO_WINDOW_POS"] = "1920,0"
    screen = pygame.display.set_mode((monitor_w, monitor_h), pygame.NOFRAME)
    screen_width = monitor_w
    screen_height = monitor_h

pygame.display.set_caption("EEG Online Interactive Loop")
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h

# UDP Settings
udp_socket_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_robot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_fes = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
FES_toggle = config.FES_toggle
ONLINE_PREP_FILL_ALPHA = getattr(
    config,
    "ONLINE_PREP_FEEDBACK_FILL_ALPHA",
    getattr(config, "ONLINE_FEEDBACK_FILL_ALPHA", getattr(config, "FEEDBACK_FILL_ALPHA", 180))
)
ONLINE_EXEC_FILL_ALPHA = getattr(
    config,
    "ONLINE_EXEC_FEEDBACK_FILL_ALPHA",
    getattr(config, "ONLINE_FEEDBACK_FILL_ALPHA", getattr(config, "FEEDBACK_FILL_ALPHA", 180))
)


# ============================================================
# ARDUINO SETUP
# ============================================================
ARDUINO_PORT = os.environ.get(
    "ARDUINO_PORT",
    config.ARDUINO_PORT if getattr(config, "USE_ARDUINO", False) else ""
)
ARDUINO_BAUD = int(os.environ.get("ARDUINO_BAUD", getattr(config, "ARDUINO_BAUD", 9600)))
arduino = None

if ARDUINO_PORT:
    try:
        logger.log_event(f"Connecting to Glove (Arduino) on {ARDUINO_PORT}...")
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.1)
        time.sleep(2)  # CRITICAL: Safety wait for Arduino reset
        logger.log_event("✅ Glove connected successfully.")
    except Exception as e:
        logger.log_event(f"❌ Error connecting to Glove: {e}", level="error")
        arduino = None
else:
    logger.log_event("ℹ️ No Arduino port configured.")

def arduino_write(cmd: bytes):
    if arduino is None:
        return
    try:
        arduino.write(cmd)
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
    """Configured safe/baseline glove command."""
    return getattr(config, "ARDUINO_CMD_REST", b"1")


def glove_cmd_mi() -> bytes:
    """Configured MI/reward glove command."""
    return getattr(config, "ARDUINO_CMD_MI", b"0")


# Force glove to start every experiment in the configured closed/baseline state.
if arduino:
    arduino_write(glove_cmd_rest())
    logger.log_event("🤚 Glove initialized to closed/baseline state.")
    init_settle = float(getattr(config, "ARDUINO_INIT_SETTLE_SECONDS", 3.0))
    if init_settle > 0:
        logger.log_event(
            f"⏳ Waiting {init_settle:.1f}s for glove to reach closed/baseline state."
        )
        time.sleep(init_settle)

# Load Model
subject_model_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
subject_model_path = getattr(config, "ONLINE_MODEL_PATH", None) or os.path.join(
    subject_model_dir,
    f"sub-{config.TRAINING_SUBJECT}_model.pkl",
)

try:
    with open(subject_model_path, 'rb') as f:
        model_pkg = pickle.load(f)
    model_type = model_pkg.get('model_type', 'unknown')

    if model_type == 'M2_LDA_shrink_MDM':
        model    = None
        template = None
        _calib = model_pkg.get('subject_calib')
        _calib_sess = model_pkg.get('session_calib', '')
        _calib_info = f"+ {_calib.split('_')[-1]}/{_calib_sess}" if _calib else "(MAESTRO — sin calibración)"
        logger.log_event(f"✅ Model loaded: {subject_model_path}")
        logger.log_event(f"   Tipo: M2 cross-subject | "
                         f"pasos={model_pkg['n_timepoints']} | "
                         f"canales={model_pkg['picks']} | "
                         f"entrenado con: {[s.split('_')[-1] for s in model_pkg['subjects_train']]} "
                         f"{_calib_info}")
    else:
        model    = model_pkg['model']
        template = model_pkg.get('template', None)
        logger.log_event(f"✅ Model loaded: {subject_model_path}")
        logger.log_event(f"   Model type: {model_type} | classes: {model.classes_}")
except FileNotFoundError:
    logger.log_event(f"❌ Model not found: {subject_model_path}", level="error")
    sys.exit(1)

warmup_model_pkg = None
if getattr(config, "WARMUP_OBSERVER_ENABLED", False):
    warmup_model_path = Path(getattr(config, "WARMUP_MODEL_PATH", ""))
    try:
        with warmup_model_path.open("rb") as f:
            _pkg = pickle.load(f)
        if _pkg.get("model_type") != "M2_LDA_shrink_MDM":
            logger.log_event(
                f"⚠️ Warmup observer ignored — unsupported model_type={_pkg.get('model_type')}"
            )
        elif _pkg.get("picks") != model_pkg.get("picks") or _pkg.get("n_timepoints") != model_pkg.get("n_timepoints"):
            logger.log_event(
                "⚠️ Warmup observer ignored — picks/n_timepoints do not match master model"
            )
        else:
            warmup_model_pkg = _pkg
            logger.log_event(f"✅ Warmup observer loaded: {warmup_model_path}")
            logger.log_event(
                f"   Warmup: sujeto={_pkg.get('subject_calib')} "
                f"sesión={_pkg.get('session_calib')} "
                f"n_total={_pkg.get('n_total')}"
            )
    except FileNotFoundError:
        logger.log_event(f"⚠️ Warmup observer not found: {warmup_model_path}")
    except Exception as e:
        logger.log_event(f"⚠️ Warmup observer load failed: {e}")

predictions_list = []
ground_truth_list = []
raw_predictions_list = []
raw_ground_truth_list = []
raw_early_predictions_list = []
raw_early_ground_truth_list = []
mdm_operational_decisions_original = []
final_validated_decisions = []
shadow_model_names = ("MDM", "LDA", "LDA3", "LR", "SVM")
shadow_earlystop_results = {name: [] for name in shadow_model_names}
shadow_stability_results = {name: [] for name in shadow_model_names}
endpoint_validation_stats = {
    "n_endpoint_fallbacks": 0,
    "accepted_by_lda": 0,
    "accepted_by_lr": 0,
    "accepted_by_both": 0,
    "rejected_to_ambiguous": 0,
    "errors_prevented": 0,
    "correct_mdm_rejected": 0,
    "mdm_already_ambiguous": 0,
}
full_window_targets = []
full_window_probabilities = {
    "MDM": [],
    "LDA_shrink": [],
    "LDA_shrink_3ch": [],
    "LR": [],
    "SVM": [],
}
# ============================================================
# WIRE RUNTIME OBJECTS
# ============================================================
_RC.config    = config
_RC.logger    = logger
_RC.model     = model
_RC.model_pkg = model_pkg if model_type == 'M2_LDA_shrink_MDM' else None
_RC.observer_model_pkg = warmup_model_pkg
_RC.screen    = screen
_RC.template  = template
_RC.screen_width = screen_width
_RC.screen_height = screen_height
_RC.udp_socket_marker = udp_socket_marker
_RC.udp_socket_robot  = udp_socket_robot
_RC.udp_socket_fes    = udp_socket_fes
_RC.FES_toggle = FES_toggle
_RC.Prev_T = Prev_T
_RC.counter = counter

if model_type == 'M2_LDA_shrink_MDM':
    n_steps = int(model_pkg.get("n_timepoints", 0))

    def _valid_m2_recenter_refs(refs):
        if not isinstance(refs, (list, tuple)) or len(refs) != n_steps:
            return False
        for ref in refs:
            arr = np.asarray(ref)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                return False
            if not np.isfinite(arr).all():
                return False
        return True

    saved_refs = Prev_T
    training_refs = model_pkg.get("mdm_recenter_refs")

    if (
        getattr(config, "M2_USE_SAVED_ADAPTIVE_RECENTER", True)
        and _valid_m2_recenter_refs(saved_refs)
    ):
        _RC.m2_prev_T = [np.asarray(ref, dtype=float).copy() for ref in saved_refs]
        _RC.m2_rec_counter = int(counter)
        logger.log_event(
            f"✅ M2 adaptive whitening loaded from saved transform "
            f"({len(_RC.m2_prev_T)} steps, counter={_RC.m2_rec_counter})."
        )
    elif (
        getattr(config, "M2_INIT_RECENTER_FROM_TRAINING", True)
        and _valid_m2_recenter_refs(training_refs)
    ):
        _RC.m2_prev_T = [np.asarray(ref, dtype=float).copy() for ref in training_refs]
        _RC.m2_rec_counter = 0
        logger.log_event(
            f"✅ M2 whitening initialized from training Riemannian mean "
            f"({len(_RC.m2_prev_T)} steps)."
        )
    else:
        _RC.m2_prev_T = None
        _RC.m2_rec_counter = 0
        if getattr(config, "RECENTERING", 0):
            logger.log_event(
                "ℹ️ M2 whitening reference unavailable — adaptive recentering "
                "will initialize from accepted online trials."
            )

# NOTE: We do not pass '_RC.arduino' because runtime_common
# will not handle the glove. The glove is handled by this main script.


# ============================================================
# ✅ PRE-TRIAL INDICATOR (MATCH OFFLINE LOOK)
# ============================================================
NEXT_INDICATOR_POS = (0.50, 0.28)
NEXT_INDICATOR_SCALE = 1.00


def resolve_prep_decision(prep_predictions, prep_all_probs, mode):
    """Return (prediction_label_or_none, reason) for the preparation phase."""
    n_valid = len(prep_predictions)
    min_valid = int(getattr(config, "MIN_FINAL_PREDICTIONS", config.MIN_PREDICTIONS))
    if n_valid < min_valid:
        return None, f"ambiguous_insufficient_predictions n={n_valid}/{min_valid}"

    mi_label = 200
    rest_label = 100
    n_mi = sum(1 for pred in prep_predictions if pred == mi_label)
    n_rest = sum(1 for pred in prep_predictions if pred == rest_label)
    vote_fraction = max(n_mi, n_rest) / n_valid if n_valid else 0.0
    min_vote_fraction = float(getattr(config, "FINAL_DECISION_MIN_VOTE_FRACTION", 0.60))

    if vote_fraction < min_vote_fraction:
        return None, (
            f"ambiguous_split_vote mi={n_mi} rest={n_rest} "
            f"vote_fraction={vote_fraction:.2f}"
        )

    winning_label = mi_label if n_mi >= n_rest else rest_label
    if prep_all_probs:
        p_mi = [row[2] for row in prep_all_probs if len(row) >= 3]
        p_rest = [row[1] for row in prep_all_probs if len(row) >= 3]
        mean_mi = sum(p_mi) / len(p_mi) if p_mi else 0.5
        mean_rest = sum(p_rest) / len(p_rest) if p_rest else 0.5
    else:
        mean_mi = n_mi / n_valid
        mean_rest = n_rest / n_valid

    target_label = mi_label if mode == 0 else rest_label
    if winning_label != target_label:
        target_name = "MI" if target_label == mi_label else "REST"
        winning_name = "MI" if winning_label == mi_label else "REST"
        return None, (
            f"ambiguous_opposite_evidence target={target_name} "
            f"winner={winning_name} mi={n_mi} rest={n_rest} "
            f"vote_fraction={vote_fraction:.2f}"
        )

    if winning_label == mi_label:
        if mean_mi < config.THRESHOLD_MI:
            return None, f"ambiguous_low_mi_mean mean_mi={mean_mi:.3f}"
        return mi_label, (
            f"accepted_mi mi={n_mi} rest={n_rest} "
            f"vote_fraction={vote_fraction:.2f} mean_mi={mean_mi:.3f}"
        )

    if mean_rest < config.THRESHOLD_REST:
        return None, f"ambiguous_low_rest_mean mean_rest={mean_rest:.3f}"
    return rest_label, (
        f"accepted_rest mi={n_mi} rest={n_rest} "
        f"vote_fraction={vote_fraction:.2f} mean_rest={mean_rest:.3f}"
    )


def resolve_prep_decision_raw(prep_predictions, prep_all_probs):
    """Return final prep decision without using the trial target."""
    n_valid = len(prep_predictions)
    min_valid = int(getattr(config, "MIN_FINAL_PREDICTIONS", config.MIN_PREDICTIONS))
    if n_valid < min_valid:
        return None, f"ambiguous_insufficient_predictions n={n_valid}/{min_valid}"

    mi_label = 200
    rest_label = 100
    n_mi = sum(1 for pred in prep_predictions if pred == mi_label)
    n_rest = sum(1 for pred in prep_predictions if pred == rest_label)
    vote_fraction = max(n_mi, n_rest) / n_valid if n_valid else 0.0
    min_vote_fraction = float(getattr(config, "FINAL_DECISION_MIN_VOTE_FRACTION", 0.60))

    if vote_fraction < min_vote_fraction:
        return None, (
            f"ambiguous_split_vote mi={n_mi} rest={n_rest} "
            f"vote_fraction={vote_fraction:.2f}"
        )

    winning_label = mi_label if n_mi >= n_rest else rest_label
    if prep_all_probs:
        p_mi = [row[2] for row in prep_all_probs if len(row) >= 3]
        p_rest = [row[1] for row in prep_all_probs if len(row) >= 3]
        mean_mi = sum(p_mi) / len(p_mi) if p_mi else 0.5
        mean_rest = sum(p_rest) / len(p_rest) if p_rest else 0.5
    else:
        mean_mi = n_mi / n_valid
        mean_rest = n_rest / n_valid

    if winning_label == mi_label:
        if mean_mi < config.THRESHOLD_MI:
            return None, f"ambiguous_low_mi_mean mean_mi={mean_mi:.3f}"
        return mi_label, (
            f"accepted_mi mi={n_mi} rest={n_rest} "
            f"vote_fraction={vote_fraction:.2f} mean_mi={mean_mi:.3f}"
        )

    if mean_rest < config.THRESHOLD_REST:
        return None, f"ambiguous_low_rest_mean mean_rest={mean_rest:.3f}"
    return rest_label, (
        f"accepted_rest mi={n_mi} rest={n_rest} "
        f"vote_fraction={vote_fraction:.2f} mean_rest={mean_rest:.3f}"
    )


def resolve_endpoint_control_decision(
    control_probabilities,
    last_step,
    model_package,
    shadow_records=None,
):
    """Decisión primaria target-independent del modelo de control configurado."""
    if model_package is None:
        return None, "ambiguous_endpoint_unavailable"

    t_points = np.asarray(model_package["t_points"], dtype=float)
    endpoint = float(
        getattr(config, "PREP_CONTROL_ENDPOINT", t_points[-1])
    )
    endpoint_step = int(np.argmin(np.abs(t_points - endpoint)))
    if last_step != endpoint_step:
        return None, (
            f"ambiguous_endpoint_missing "
            f"last_step={last_step + 1 if last_step >= 0 else 0} "
            f"required_step={endpoint_step + 1}"
        )

    control_model = str(getattr(config, "PREP_CONTROL_MODEL", "MDM")).upper()
    control_probability_key = {
        "MDM": "MDM",
        "LDA": "LDA",
        "LDA_SHRINK": "LDA",
        "LDA_SHRINKAGE": "LDA",
        "LDA3": "LDA3",
        "LDA_3CH": "LDA3",
        "LDA_SHRINK_3CH": "LDA3",
        "COMPACT_LDA": "LDA3",
        "LR": "LR",
        "SVM": "SVM",
    }.get(control_model, "MDM")

    # Prefer the exact, validity-aware value captured at the endpoint. The
    # legacy lists are rounded for display and can move values across a boundary.
    p_mi = None
    if shadow_records:
        endpoint_record = next(
            (
                record for record in shadow_records
                if int(record["step_index"]) == endpoint_step
            ),
            None,
        )
        if endpoint_record is not None:
            p_mi = endpoint_record.get("probabilities", {}).get(
                control_probability_key
            )
    if p_mi is None and control_probabilities:
        candidate = float(control_probabilities[-1])
        if np.isfinite(candidate):
            p_mi = candidate
    if p_mi is None or not np.isfinite(p_mi):
        return None, f"ambiguous_endpoint_invalid_{control_probability_key}"
    p_mi = float(p_mi)
    mi_threshold = float(
        getattr(config, "ENDPOINT_MDM_MI_THRESHOLD", 0.60)
    )
    rest_threshold = float(
        getattr(config, "ENDPOINT_MDM_REST_THRESHOLD", 0.40)
    )
    if p_mi >= mi_threshold:
        return 200, (
            f"accepted_mi control={control_probability_key} "
            f"endpoint={endpoint:+.2f}s "
            f"p_mi={p_mi:.3f} threshold={mi_threshold:.2f}"
        )

    if p_mi <= rest_threshold:
        return 100, (
            f"accepted_rest control={control_probability_key} "
            f"endpoint={endpoint:+.2f}s "
            f"p_mi={p_mi:.3f} threshold={rest_threshold:.2f}"
        )

    if (
        control_probability_key == "MDM"
        and bool(getattr(config, "MDM_WEIGHTED_RESCUE_ENABLED", False))
        and shadow_records
    ):
        weighted_values = []
        for record in sorted(shadow_records, key=lambda item: int(item["step_index"])):
            record_time = float(record.get("time", np.nan))
            if record_time > endpoint + 1e-9:
                continue
            record_p_mi = record.get("probabilities", {}).get("MDM")
            if record_p_mi is None or not np.isfinite(record_p_mi):
                continue
            weighted_values.append(float(record_p_mi))

        if weighted_values:
            weights = np.arange(1, len(weighted_values) + 1, dtype=float)
            weighted_p_mi = float(
                np.average(np.asarray(weighted_values), weights=weights)
            )
            weighted_mi_threshold = float(
                getattr(config, "MDM_WEIGHTED_MI_THRESHOLD", mi_threshold)
            )
            weighted_rest_threshold = float(
                getattr(config, "MDM_WEIGHTED_REST_THRESHOLD", rest_threshold)
            )
            if weighted_p_mi >= weighted_mi_threshold:
                return 200, (
                    f"accepted_mi_mdm_weighted endpoint={endpoint:+.2f}s "
                    f"endpoint_p_mi={p_mi:.3f} weighted_p_mi={weighted_p_mi:.3f} "
                    f"n={len(weighted_values)} threshold={weighted_mi_threshold:.2f}"
                )
            if weighted_p_mi <= weighted_rest_threshold:
                return 100, (
                    f"accepted_rest_mdm_weighted endpoint={endpoint:+.2f}s "
                    f"endpoint_p_mi={p_mi:.3f} weighted_p_mi={weighted_p_mi:.3f} "
                    f"n={len(weighted_values)} threshold={weighted_rest_threshold:.2f}"
                )

    if (
        control_probability_key == "MDM"
        and bool(getattr(config, "VIEWER_TEMPORAL_RESCUE_ENABLED", False))
        and shadow_records
    ):
        viewer_models = list(
            getattr(
                config,
                "VIEWER_TEMPORAL_RESCUE_MODELS",
                ["LDA", "LDA3", "LR", "SVM"],
            )
        )
        required_votes = int(
            getattr(config, "VIEWER_TEMPORAL_REQUIRED_VOTES", 3)
        )
        min_vote_fraction = float(
            getattr(config, "VIEWER_TEMPORAL_MIN_VOTE_FRACTION", 0.60)
        )

        viewer_predictions = {}
        for model_name in viewer_models:
            model_votes = []
            for record in sorted(
                shadow_records, key=lambda item: int(item["step_index"])
            ):
                record_time = float(record.get("time", np.nan))
                if record_time > endpoint + 1e-9:
                    continue
                model_p_mi = record.get("probabilities", {}).get(model_name)
                if model_p_mi is None or not np.isfinite(model_p_mi):
                    continue
                model_votes.append(200 if float(model_p_mi) >= 0.5 else 100)

            if not model_votes:
                continue
            mi_votes = sum(vote == 200 for vote in model_votes)
            rest_votes = len(model_votes) - mi_votes
            winner_votes = max(mi_votes, rest_votes)
            if winner_votes / len(model_votes) < min_vote_fraction:
                continue
            viewer_predictions[model_name] = (
                200 if mi_votes > rest_votes else 100
            )

        mi_viewers = [
            name for name, prediction in viewer_predictions.items()
            if prediction == 200
        ]
        rest_viewers = [
            name for name, prediction in viewer_predictions.items()
            if prediction == 100
        ]
        if len(mi_viewers) >= required_votes:
            return 200, (
                f"accepted_mi_viewer_temporal endpoint={endpoint:+.2f}s "
                f"endpoint_p_mi={p_mi:.3f} viewers={mi_viewers} "
                f"required={required_votes}/{len(viewer_models)}"
            )
        if len(rest_viewers) >= required_votes:
            return 100, (
                f"accepted_rest_viewer_temporal endpoint={endpoint:+.2f}s "
                f"endpoint_p_mi={p_mi:.3f} viewers={rest_viewers} "
                f"required={required_votes}/{len(viewer_models)}"
            )

    return None, (
        f"ambiguous_endpoint control={control_probability_key} "
        f"endpoint={endpoint:+.2f}s "
        f"p_mi={p_mi:.3f} band=({rest_threshold:.2f},{mi_threshold:.2f})"
    )


def evaluate_shadow_early_stop(records, model_name, target_label):
    """Simulate target-independent early stop without affecting online control."""
    valid_records = []
    for record in records:
        p_mi = record.get("probabilities", {}).get(model_name)
        if p_mi is not None and np.isfinite(p_mi):
            valid_records.append((record, float(p_mi)))

    result = {
        "model": model_name,
        "triggered": False,
        "step": None,
        "time": None,
        "prediction": None,
        "confidence": None,
        "correct": None,
        "target": target_label,
        "n_valid": len(valid_records),
    }
    if not getattr(config, "SHADOW_MODEL_ANALYSIS_ENABLED", True):
        return result

    alpha = float(config.INTEGRATOR_ALPHA)
    min_predictions = int(config.MIN_PREDICTIONS)
    consecutive_required = int(config.EARLYSTOP_CONSECUTIVE_PREDICTIONS)
    labels = []
    integrated_mi = None
    integrated_rest = None

    for observation_number, (record, p_mi) in enumerate(valid_records, start=1):
        p_rest = 1.0 - p_mi
        if integrated_mi is None:
            # Matches the real controller: the first valid value seeds the
            # integrator directly rather than being attenuated from zero.
            integrated_mi = p_mi
            integrated_rest = p_rest
        else:
            integrated_mi = alpha * integrated_mi + (1.0 - alpha) * p_mi
            integrated_rest = (
                alpha * integrated_rest + (1.0 - alpha) * p_rest
            )

        prediction = 200 if p_mi >= 0.5 else 100
        labels.append(prediction)
        recent = labels[-consecutive_required:]
        sustained = (
            len(recent) == consecutive_required
            and all(label == prediction for label in recent)
        )
        confidence = integrated_mi if prediction == 200 else integrated_rest
        threshold = (
            float(config.THRESHOLD_MI)
            if prediction == 200
            else float(config.THRESHOLD_REST)
        )
        if (
            observation_number >= min_predictions
            and sustained
            and confidence >= threshold
        ):
            result.update({
                "triggered": True,
                "step": int(record["step"]),
                "time": float(record["time"]),
                "prediction": prediction,
                "confidence": float(confidence),
                "correct": prediction == target_label,
            })
            break

    return result


def find_shadow_stabilization(records, model_name, target_label, model_package):
    """Find the first contiguous step whose class persists to the endpoint."""
    result = {
        "model": model_name,
        "available": False,
        "step": None,
        "time": None,
        "prediction": None,
        "correct": None,
        "target": target_label,
        "n_valid": 0,
        "reason": "endpoint_unavailable",
    }
    if model_package is None:
        return result

    t_points = np.asarray(model_package["t_points"], dtype=float)
    endpoint = float(getattr(config, "PREP_CONTROL_ENDPOINT", t_points[-1]))
    endpoint_index = int(np.argmin(np.abs(t_points - endpoint)))

    valid = []
    for record in records:
        p_mi = record.get("probabilities", {}).get(model_name)
        if p_mi is not None and np.isfinite(p_mi):
            valid.append({
                "step_index": int(record["step_index"]),
                "step": int(record["step"]),
                "time": float(record["time"]),
                "prediction": 200 if float(p_mi) >= 0.5 else 100,
            })
    valid.sort(key=lambda item: item["step_index"])
    result["n_valid"] = len(valid)

    if not valid or valid[-1]["step_index"] != endpoint_index:
        return result

    final_prediction = valid[-1]["prediction"]
    for position, candidate in enumerate(valid):
        suffix = valid[position:]
        expected_indices = list(
            range(candidate["step_index"], endpoint_index + 1)
        )
        observed_indices = [item["step_index"] for item in suffix]
        stable_class = all(
            item["prediction"] == final_prediction for item in suffix
        )
        if stable_class and observed_indices == expected_indices:
            result.update({
                "available": True,
                "step": candidate["step"],
                "time": candidate["time"],
                "prediction": final_prediction,
                "correct": final_prediction == target_label,
                "reason": "stable_through_endpoint",
            })
            break

    return result


def validate_mdm_endpoint_with_observers(
    mdm_prediction, shadow_records, model_package
):
    """Validate an MDM endpoint class; observers can only accept or abstain."""
    return validate_mdm_decision_with_observers(
        mdm_prediction,
        shadow_records,
        model_package,
        step_index=None,
        context_label="endpoint",
    )


def validate_mdm_decision_with_observers(
    mdm_prediction,
    shadow_records,
    model_package,
    step_index=None,
    context_label="decision",
):
    """Validate an MDM class at one step; LDA/LR can only accept or abstain.

    This keeps MDM as the only proposing controller. LDA and LR never replace
    its class: if at least one agrees, MDM is accepted; if both available
    validators disagree, the operational output becomes AMBIGUOUS.
    """
    if mdm_prediction is None:
        return None, f"mdm_{context_label}_ambiguous", {}
    if model_package is None:
        return None, "model_package_unavailable", {}

    if step_index is None:
        t_points = np.asarray(model_package["t_points"], dtype=float)
        endpoint = float(getattr(config, "PREP_CONTROL_ENDPOINT", t_points[-1]))
        step_index = int(np.argmin(np.abs(t_points - endpoint)))

    decision_record = next(
        (
            record for record in shadow_records
            if int(record["step_index"]) == int(step_index)
        ),
        None,
    )
    if decision_record is None:
        return None, f"validator_{context_label}_unavailable", {}

    validator_predictions = {}
    for model_name in ("LDA", "LR"):
        p_mi = decision_record.get("probabilities", {}).get(model_name)
        if p_mi is not None and np.isfinite(p_mi):
            validator_predictions[model_name] = (
                200 if float(p_mi) >= 0.5 else 100
            )

    agreeing = [
        name for name, prediction in validator_predictions.items()
        if prediction == mdm_prediction
    ]
    if agreeing:
        return (
            mdm_prediction,
            "accepted_by_" + "+".join(agreeing),
            validator_predictions,
        )

    if len(validator_predictions) == 2:
        return None, "both_validators_disagree", validator_predictions
    return None, "insufficient_validator_agreement", validator_predictions


def log_shadow_model_summary(logger, earlystop_results, stability_results):
    """Log session-level diagnostic tables for all shadow models."""
    logger.log_event("[SHADOW_EARLYSTOP_SUMMARY]")
    logger.log_event(
        "  model  n_trials  early_stop_count  mean_early_stop_step  "
        "median_early_stop_step  early_stop_accuracy  false_MI  false_REST"
    )

    class_step_means = {200: {}, 100: {}}
    summary_metrics = {}
    for model_name in shadow_model_names:
        triggered = [
            result for result in earlystop_results[model_name]
            if result["triggered"]
        ]
        correct = sum(bool(result["correct"]) for result in triggered)
        accuracy = 100.0 * correct / len(triggered) if triggered else float("nan")
        avg_step = (
            float(np.mean([result["step"] for result in triggered]))
            if triggered else float("nan")
        )
        median_step = (
            float(np.median([result["step"] for result in triggered]))
            if triggered else float("nan")
        )
        mi_results = [
            result for result in triggered
            if result["prediction"] == 200 and result["target"] == 200
        ]
        rest_results = [
            result for result in triggered
            if result["prediction"] == 100 and result["target"] == 100
        ]
        avg_mi = (
            float(np.mean([result["step"] for result in mi_results]))
            if mi_results else float("nan")
        )
        avg_rest = (
            float(np.mean([result["step"] for result in rest_results]))
            if rest_results else float("nan")
        )
        class_step_means[200][model_name] = avg_mi
        class_step_means[100][model_name] = avg_rest
        false_mi = sum(
            result["prediction"] == 200 and result["target"] == 100
            for result in triggered
        )
        false_rest = sum(
            result["prediction"] == 100 and result["target"] == 200
            for result in triggered
        )
        summary_metrics[model_name] = {
            "stops": len(triggered),
            "accuracy": accuracy,
            "false_total": false_mi + false_rest,
        }
        logger.log_event(
            f"  {model_name:<6} {len(earlystop_results[model_name]):>8}  "
            f"{len(triggered):>16}  {avg_step:>20.2f}  "
            f"{median_step:>22.2f}  {accuracy:>18.1f}%  "
            f"{false_mi:>8}  {false_rest:>10}"
        )

    for label, class_name in ((200, "MI"), (100, "REST")):
        finite = {
            model: value
            for model, value in class_step_means[label].items()
            if np.isfinite(value)
        }
        if finite:
            fastest_step = min(finite.values())
            fastest_models = [
                model for model, value in finite.items()
                if np.isclose(value, fastest_step)
            ]
            logger.log_event(
                f"[SHADOW_FASTEST_{class_name}] "
                f"models={'+'.join(fastest_models)} avg_step={fastest_step:.2f}"
            )

    if summary_metrics:
        most_stops = max(
            metric["stops"] for metric in summary_metrics.values()
        )
        most_stops_models = [
            model for model, metric in summary_metrics.items()
            if metric["stops"] == most_stops
        ]
        finite_accuracy = {
            model: metric["accuracy"]
            for model, metric in summary_metrics.items()
            if np.isfinite(metric["accuracy"])
        }
        logger.log_event(
            f"[SHADOW_MOST_EARLY_STOPS] "
            f"models={'+'.join(most_stops_models)} count={most_stops}"
        )
        if finite_accuracy:
            best_accuracy = max(finite_accuracy.values())
            best_accuracy_models = [
                model for model, value in finite_accuracy.items()
                if np.isclose(value, best_accuracy)
            ]
            logger.log_event(
                f"[SHADOW_BEST_EARLYSTOP_ACCURACY] "
                f"models={'+'.join(best_accuracy_models)} "
                f"accuracy={best_accuracy:.1f}%"
            )
        most_false = max(
            metric["false_total"] for metric in summary_metrics.values()
        )
        most_false_models = [
            model for model, metric in summary_metrics.items()
            if metric["false_total"] == most_false
        ]
        logger.log_event(
            f"[SHADOW_MOST_FALSE_ACTIVATIONS] "
            f"models={'+'.join(most_false_models)} count={most_false}"
        )

    logger.log_event("[SHADOW_STABILITY_SUMMARY]")
    logger.log_event(
        "  model  n_trials  mean_stabilization_step  "
        "median_stabilization_step  stable_accuracy"
    )
    stable_class_means = {200: {}, 100: {}}
    for model_name in shadow_model_names:
        available = [
            result for result in stability_results[model_name]
            if result["available"]
        ]
        avg_step = (
            float(np.mean([result["step"] for result in available]))
            if available else float("nan")
        )
        median_step = (
            float(np.median([result["step"] for result in available]))
            if available else float("nan")
        )
        accuracy = (
            100.0 * sum(bool(result["correct"]) for result in available)
            / len(available)
            if available else float("nan")
        )
        stable_mi = [
            result for result in available
            if result["prediction"] == 200 and result["target"] == 200
        ]
        stable_rest = [
            result for result in available
            if result["prediction"] == 100 and result["target"] == 100
        ]
        avg_mi = (
            float(np.mean([result["step"] for result in stable_mi]))
            if stable_mi else float("nan")
        )
        avg_rest = (
            float(np.mean([result["step"] for result in stable_rest]))
            if stable_rest else float("nan")
        )
        stable_class_means[200][model_name] = avg_mi
        stable_class_means[100][model_name] = avg_rest
        logger.log_event(
            f"  {model_name:<6} {len(stability_results[model_name]):>8}  "
            f"{avg_step:>23.2f}  {median_step:>25.2f}  "
            f"{accuracy:>14.1f}%"
        )

    for label, class_name in ((200, "MI"), (100, "REST")):
        finite = {
            model: value
            for model, value in stable_class_means[label].items()
            if np.isfinite(value)
        }
        if finite:
            fastest_step = min(finite.values())
            fastest_models = [
                model for model, value in finite.items()
                if np.isclose(value, fastest_step)
            ]
            logger.log_event(
                f"[SHADOW_FASTEST_STABLE_{class_name}] "
                f"models={'+'.join(fastest_models)} avg_step={fastest_step:.2f}"
            )


def log_endpoint_validation_summary(logger, stats):
    """Log aggregate effects of the LDA/LR endpoint validation layer."""
    logger.log_event("[ENDPOINT_VALIDATION_SUMMARY]")
    for field in (
        "n_endpoint_fallbacks",
        "accepted_by_lda",
        "accepted_by_lr",
        "accepted_by_both",
        "rejected_to_ambiguous",
        "errors_prevented",
        "correct_mdm_rejected",
    ):
        logger.log_event(f"  {field}={stats[field]}")
    # Additional accounting: these trials were already ambiguous at MDM and
    # therefore never reached the LDA/LR agreement test.
    logger.log_event(
        f"  mdm_already_ambiguous={stats['mdm_already_ambiguous']}"
    )


def log_raw_decision_summary(logger, title, predictions, targets):
    if not targets:
        return

    total = len(targets)
    correct = sum(
        1 for pred, target in zip(predictions, targets)
        if pred is not None and pred == target
    )
    errors = sum(
        1 for pred, target in zip(predictions, targets)
        if pred is not None and pred != target
    )
    ambiguous = sum(1 for pred in predictions if pred is None)
    decided = correct + errors
    total_acc = (correct / total) * 100
    decision_acc = (correct / decided) * 100 if decided else 0.0

    logger.log_event(f"{title}:")
    logger.log_event(f"{title} total accuracy = {total_acc:.1f}%")
    logger.log_event(f"{correct} correctos / {total} trials = {total_acc:.1f}%")
    logger.log_event(f"{title} decision accuracy = {decision_acc:.1f}%")
    logger.log_event(
        f"{correct} correctos / {decided} decisiones no ambiguas = "
        f"{decision_acc:.1f}%"
    )
    logger.log_event(
        f"{title} counts: correct={correct} | incorrect={errors} "
        f"| ambiguous={ambiguous}"
    )


def _decision_metrics_for_csv(predictions, targets):
    """Return compact target-independent metrics without changing decisions."""
    total = len(targets)
    if total == 0:
        return {
            "n_trials": 0,
            "n_decided": 0,
            "n_ambiguous": 0,
            "correct": 0,
            "incorrect": 0,
            "total_accuracy": 0.0,
            "decision_accuracy": 0.0,
            "mi_recall": 0.0,
            "rest_recall": 0.0,
        }

    correct = sum(
        1 for pred, target in zip(predictions, targets)
        if pred is not None and pred == target
    )
    incorrect = sum(
        1 for pred, target in zip(predictions, targets)
        if pred is not None and pred != target
    )
    ambiguous = sum(1 for pred in predictions if pred is None)
    decided = correct + incorrect

    mi_total = sum(1 for target in targets if target == 200)
    rest_total = sum(1 for target in targets if target == 100)
    mi_correct = sum(
        1 for pred, target in zip(predictions, targets)
        if target == 200 and pred == 200
    )
    rest_correct = sum(
        1 for pred, target in zip(predictions, targets)
        if target == 100 and pred == 100
    )

    return {
        "n_trials": total,
        "n_decided": decided,
        "n_ambiguous": ambiguous,
        "correct": correct,
        "incorrect": incorrect,
        "total_accuracy": correct / total if total else 0.0,
        "decision_accuracy": correct / decided if decided else 0.0,
        "mi_recall": mi_correct / mi_total if mi_total else 0.0,
        "rest_recall": rest_correct / rest_total if rest_total else 0.0,
    }


def _as_valid_spd_list(refs, expected_len=None):
    """Return a copied list of square finite SPD-like matrices, else None."""
    if not isinstance(refs, (list, tuple)):
        return None
    if expected_len is not None and len(refs) != expected_len:
        return None
    out = []
    for ref in refs:
        arr = np.asarray(ref, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            return None
        if not np.isfinite(arr).all():
            return None
        out.append(arr.copy())
    return out


def _spd_riemann_distance(A, B):
    """
    Affine-invariant Riemannian distance between two SPD matrices.

    This diagnostic is passive and independent of the online decoder decision.
    Eigenvalues are clipped only for numerical robustness.
    """
    A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
    B = 0.5 * (np.asarray(B, dtype=float) + np.asarray(B, dtype=float).T)
    eps = 1e-12

    evals_a, evecs_a = np.linalg.eigh(A)
    evals_a = np.clip(evals_a, eps, None)
    invsqrt_a = (evecs_a * (1.0 / np.sqrt(evals_a))) @ evecs_a.T

    C = invsqrt_a @ B @ invsqrt_a
    C = 0.5 * (C + C.T)
    evals_c = np.linalg.eigvalsh(C)
    evals_c = np.clip(evals_c, eps, None)
    return float(np.linalg.norm(np.log(evals_c)))


def _riemann_reference_distances(train_refs, current_refs):
    """Per-step distance between training references and current references."""
    if train_refs is None or current_refs is None or len(train_refs) != len(current_refs):
        return []
    distances = []
    for train_ref, current_ref in zip(train_refs, current_refs):
        try:
            distances.append(_spd_riemann_distance(train_ref, current_ref))
        except Exception:
            distances.append(np.nan)
    return distances


def _reference_change_magnitude(refs_before, refs_after):
    """Euclidean/Frobenius magnitude of the adaptive reference change."""
    refs_before = _as_valid_spd_list(refs_before)
    refs_after = _as_valid_spd_list(
        refs_after, expected_len=len(refs_before) if refs_before is not None else None
    )
    if refs_before is None or refs_after is None:
        return np.nan, []

    per_step = []
    total_sq = 0.0
    for before, after in zip(refs_before, refs_after):
        try:
            value = float(np.linalg.norm(np.asarray(after) - np.asarray(before), ord="fro"))
        except Exception:
            value = np.nan
        per_step.append(value)
        if np.isfinite(value):
            total_sq += value * value
    return float(np.sqrt(total_sq)), per_step


def _summarize_distances(distances):
    arr = np.asarray(distances, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": np.nan, "median": np.nan, "max": np.nan, "min": np.nan}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def _hash_reference_list(refs):
    """Short reproducibility fingerprint for a list of reference matrices."""
    valid_refs = _as_valid_spd_list(refs)
    if valid_refs is None:
        return "NA"
    h = hashlib.sha1()
    for ref in valid_refs:
        arr = np.ascontiguousarray(ref, dtype=np.float64)
        h.update(arr.tobytes())
    return h.hexdigest()[:12]


def _append_dict_csv(path, row, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def _run_label_from_logger(logger):
    parts = logger.log_dir.name.split("_")
    run_label = next((part for part in parts if part.startswith("run-")), "")
    return run_label or logger.log_dir.name


def _run_number_from_label(run_label):
    match = re.search(r"run-(\d+)", str(run_label))
    return int(match.group(1)) if match else ""


def _class_label_from_code(value):
    if value == 200:
        return "MI"
    if value == 100:
        return "REST"
    return "AMBIGUOUS"


def _decision_source_label(used_endpoint_fallback, prep_earlystop):
    if prep_earlystop and not used_endpoint_fallback:
        return "EARLY_STOP"
    if used_endpoint_fallback:
        return "ENDPOINT"
    return "OTHER"


def _valid_m2_steps_from_shadow_records(shadow_records):
    valid_steps = set()
    for record in shadow_records or []:
        try:
            p_mdm = record.get("probabilities", {}).get("MDM")
            if p_mdm is not None and np.isfinite(float(p_mdm)):
                valid_steps.add(int(record.get("step", len(valid_steps) + 1)))
        except Exception:
            continue
    return len(valid_steps)


def _infer_recenter_rejection_reason(
    *,
    prediction_label,
    target_label,
    prep_confidence,
    bad_eeg,
    model_available,
    prep_epoch_available,
    valid_m2_steps,
    seen_before,
    seen_after,
    updated_after,
    updated_before,
):
    if not getattr(config, "RECENTERING", 0):
        return "RECENTERING_DISABLED"
    if (
        getattr(config, "RECENTERING_REQUIRE_NON_AMBIGUOUS", True)
        and prediction_label is None
    ):
        return "AMBIGUOUS_DECISION"
    if (
        getattr(config, "RECENTERING_REQUIRE_CORRECT", True)
        and target_label is not None
        and prediction_label != target_label
    ):
        return "DECISION_DID_NOT_MATCH_TARGET"
    min_conf = float(getattr(config, "RECENTERING_MIN_CONFIDENCE", 0.0))
    if prep_confidence is not None and prep_confidence < min_conf:
        return "LOW_CONFIDENCE"
    if bad_eeg:
        return "BAD_EEG"
    if not model_available or not prep_epoch_available:
        return "MISSING_MODEL_OR_EPOCH"
    min_trials = int(getattr(config, "RECENTERING_MIN_TRIALS", 0))
    if seen_after > seen_before and seen_after <= min_trials:
        return "WARMUP"
    if valid_m2_steps <= 0:
        return "NO_VALID_M2_STEPS"
    if int(updated_after or 0) <= int(updated_before or 0):
        return "NO_STEPS_UPDATED"
    return "UNKNOWN"


def log_riemann_update_event(
    logger,
    *,
    subject,
    session,
    run_label,
    condition,
    trial_number,
    update_class,
    train_refs,
    refs_before,
    refs_after,
    updates_at_session_start,
    updates_before,
    updates_after,
    prep_confidence,
    target_label,
    prediction_label,
    update_reason,
    decision_source,
    valid_m2_steps,
):
    """Passive per-update adaptive recentering log.

    This records scientific diagnostics only. It does not participate in the
    decoder decision, recentering update, FES, robot, or feedback logic.
    """
    train_refs = _as_valid_spd_list(train_refs)
    refs_before = _as_valid_spd_list(
        refs_before, expected_len=len(train_refs) if train_refs is not None else None
    )
    refs_after = _as_valid_spd_list(
        refs_after, expected_len=len(train_refs) if train_refs is not None else None
    )
    if train_refs is None or refs_before is None or refs_after is None:
        return

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    run_number = _run_number_from_label(run_label)
    updates_at_session_start = int(updates_at_session_start or 0)
    updates_before = int(updates_before or 0)
    updates_after = int(updates_after or 0)
    update_index = max(0, updates_after - updates_at_session_start)
    global_update_index = updates_after

    dist_before = _riemann_reference_distances(train_refs, refs_before)
    dist_after = _riemann_reference_distances(train_refs, refs_after)
    before_summary = _summarize_distances(dist_before)
    after_summary = _summarize_distances(dist_after)
    adaptation_magnitude, per_step_magnitude = _reference_change_magnitude(
        refs_before, refs_after
    )

    means_dir = logger.log_base / "riemann_adaptive_means"
    means_dir.mkdir(parents=True, exist_ok=True)
    mean_npy = means_dir / (
        f"{run_label}_update-{update_index:04d}_"
        f"global-{global_update_index:04d}_trial-{trial_number:03d}.npy"
    )
    np.save(mean_npy, np.stack(refs_after, axis=0))

    update_csv = logger.log_base / "riemann_adaptation_updates.csv"
    fields = [
        "timestamp", "subject", "session", "run", "run_number", "condition",
        "trial_number", "update_index", "global_update_index",
        "update_class", "target_class", "prediction_class", "update_reason",
        "decision_source", "prep_confidence", "valid_m2_steps",
        "updates_before", "updates_after", "updates_at_session_start",
        "distance_to_training_mean", "distance_to_training_median",
        "distance_to_training_max", "distance_to_training_min",
        "distance_before_mean", "distance_delta_mean",
        "adaptation_magnitude", "adaptation_magnitude_mean_step",
        "adaptation_magnitude_max_step",
        "n_steps", "adaptive_mean_npy",
        "train_mean_hash", "mean_before_hash", "mean_after_hash",
    ]
    finite_step_mag = np.asarray(per_step_magnitude, dtype=float)
    finite_step_mag = finite_step_mag[np.isfinite(finite_step_mag)]
    _append_dict_csv(
        update_csv,
        {
            "timestamp": timestamp,
            "subject": subject,
            "session": session,
            "run": run_label,
            "run_number": run_number,
            "condition": condition,
            "trial_number": trial_number,
            "update_index": update_index,
            "global_update_index": global_update_index,
            "update_class": update_class,
            "target_class": _class_label_from_code(target_label),
            "prediction_class": _class_label_from_code(prediction_label),
            "update_reason": update_reason,
            "decision_source": decision_source,
            "prep_confidence": prep_confidence if prep_confidence is not None else "",
            "valid_m2_steps": valid_m2_steps,
            "updates_before": updates_before,
            "updates_after": updates_after,
            "updates_at_session_start": updates_at_session_start,
            "distance_to_training_mean": after_summary["mean"],
            "distance_to_training_median": after_summary["median"],
            "distance_to_training_max": after_summary["max"],
            "distance_to_training_min": after_summary["min"],
            "distance_before_mean": before_summary["mean"],
            "distance_delta_mean": after_summary["mean"] - before_summary["mean"],
            "adaptation_magnitude": adaptation_magnitude,
            "adaptation_magnitude_mean_step": (
                float(np.mean(finite_step_mag)) if finite_step_mag.size else np.nan
            ),
            "adaptation_magnitude_max_step": (
                float(np.max(finite_step_mag)) if finite_step_mag.size else np.nan
            ),
            "n_steps": len(refs_after),
            "adaptive_mean_npy": str(mean_npy),
            "train_mean_hash": _hash_reference_list(train_refs),
            "mean_before_hash": _hash_reference_list(refs_before),
            "mean_after_hash": _hash_reference_list(refs_after),
        },
        fields,
    )
    logger.log_event(
        "[RIEMANN_ADAPT_UPDATE] "
        f"trial={trial_number} run={run_label} update_index={update_index} "
        f"class={update_class} source={decision_source} "
        f"reason={update_reason} valid_steps={valid_m2_steps} "
        f"dist_train={after_summary['mean']:.6f} "
        f"magnitude={adaptation_magnitude:.6f} npy={mean_npy}"
    )


def log_riemann_rejection_event(
    logger,
    *,
    subject,
    session,
    run_label,
    condition,
    trial_number,
    target_label,
    prediction_label,
    reason,
    prep_confidence,
    valid_m2_steps,
    bad_eeg,
    early_stop,
):
    """Passive per-trial recentering rejection log."""
    rejection_csv = logger.log_base / "riemann_adaptation_rejections.csv"
    fields = [
        "timestamp", "subject", "session", "run", "run_number", "condition",
        "trial_number", "target", "mdm_prediction_original", "reason",
        "prep_confidence", "valid_m2_steps", "bad_eeg", "early_stop",
    ]
    _append_dict_csv(
        rejection_csv,
        {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "subject": subject,
            "session": session,
            "run": run_label,
            "run_number": _run_number_from_label(run_label),
            "condition": condition,
            "trial_number": trial_number,
            "target": _class_label_from_code(target_label),
            "mdm_prediction_original": _class_label_from_code(prediction_label),
            "reason": reason,
            "prep_confidence": prep_confidence if prep_confidence is not None else "",
            "valid_m2_steps": valid_m2_steps,
            "bad_eeg": bool(bad_eeg),
            "early_stop": bool(early_stop),
        },
        fields,
    )
    logger.log_event(
        "[RIEMANN_ADAPT_REJECT] "
        f"trial={trial_number} run={run_label} reason={reason} "
        f"target={_class_label_from_code(target_label)} "
        f"mdm={_class_label_from_code(prediction_label)} "
        f"conf={prep_confidence if prep_confidence is not None else 'NA'} "
        f"valid_steps={valid_m2_steps} bad_eeg={bool(bad_eeg)} "
        f"early_stop={bool(early_stop)}"
    )


def log_riemann_adaptation_csv(
    logger,
    *,
    subject,
    session,
    run_label,
    condition,
    train_refs,
    refs_before,
    refs_after,
    updates_before,
    updates_after,
    final_predictions,
    mdm_predictions,
    targets,
):
    """
    Passive CSV-only Riemannian adaptation diagnostics.

    This does not modify feature extraction, MDM decisions, FES, robot, or
    adaptive recentering. It only records how far the current adaptive
    reference moved away from the training reference and how that relates to
    run-level performance.
    """
    train_refs = _as_valid_spd_list(train_refs)
    refs_before = _as_valid_spd_list(
        refs_before, expected_len=len(train_refs) if train_refs is not None else None
    )
    refs_after = _as_valid_spd_list(
        refs_after, expected_len=len(train_refs) if train_refs is not None else None
    )

    dist_before = _riemann_reference_distances(train_refs, refs_before)
    dist_after = _riemann_reference_distances(train_refs, refs_after)
    before_summary = _summarize_distances(dist_before)
    after_summary = _summarize_distances(dist_after)
    final_metrics = _decision_metrics_for_csv(final_predictions, targets)
    mdm_metrics = _decision_metrics_for_csv(mdm_predictions, targets)

    updates_before = int(updates_before or 0)
    updates_after = int(updates_after or 0)
    updates_this_run = max(0, updates_after - updates_before)

    run_csv = logger.log_base / "riemann_adaptation_log.csv"
    step_csv = logger.log_base / "riemann_adaptation_steps.csv"
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    run_fields = [
        "timestamp", "subject", "session", "run", "condition",
        "model_type", "n_steps",
        "train_mean_hash", "mean_before_hash", "mean_after_hash",
        "distance_before_mean", "distance_before_median",
        "distance_before_max", "distance_before_min",
        "distance_after_mean", "distance_after_median",
        "distance_after_max", "distance_after_min",
        "distance_delta_mean", "distance_delta_max",
        "updates_before", "updates_after", "updates_this_run",
        "n_trials", "n_decided", "n_ambiguous",
        "final_total_accuracy", "final_decision_accuracy",
        "final_mi_recall", "final_rest_recall",
        "final_correct", "final_incorrect",
        "mdm_total_accuracy", "mdm_decision_accuracy",
        "mdm_mi_recall", "mdm_rest_recall",
        "mdm_correct", "mdm_incorrect", "log_dir",
    ]
    run_row = {
        "timestamp": timestamp,
        "subject": subject,
        "session": session,
        "run": run_label,
        "condition": condition,
        "model_type": "M2_LDA_shrink_MDM",
        "n_steps": len(train_refs) if train_refs is not None else 0,
        "train_mean_hash": _hash_reference_list(train_refs),
        "mean_before_hash": _hash_reference_list(refs_before),
        "mean_after_hash": _hash_reference_list(refs_after),
        "distance_before_mean": before_summary["mean"],
        "distance_before_median": before_summary["median"],
        "distance_before_max": before_summary["max"],
        "distance_before_min": before_summary["min"],
        "distance_after_mean": after_summary["mean"],
        "distance_after_median": after_summary["median"],
        "distance_after_max": after_summary["max"],
        "distance_after_min": after_summary["min"],
        "distance_delta_mean": after_summary["mean"] - before_summary["mean"],
        "distance_delta_max": after_summary["max"] - before_summary["max"],
        "updates_before": updates_before,
        "updates_after": updates_after,
        "updates_this_run": updates_this_run,
        "n_trials": final_metrics["n_trials"],
        "n_decided": final_metrics["n_decided"],
        "n_ambiguous": final_metrics["n_ambiguous"],
        "final_total_accuracy": final_metrics["total_accuracy"],
        "final_decision_accuracy": final_metrics["decision_accuracy"],
        "final_mi_recall": final_metrics["mi_recall"],
        "final_rest_recall": final_metrics["rest_recall"],
        "final_correct": final_metrics["correct"],
        "final_incorrect": final_metrics["incorrect"],
        "mdm_total_accuracy": mdm_metrics["total_accuracy"],
        "mdm_decision_accuracy": mdm_metrics["decision_accuracy"],
        "mdm_mi_recall": mdm_metrics["mi_recall"],
        "mdm_rest_recall": mdm_metrics["rest_recall"],
        "mdm_correct": mdm_metrics["correct"],
        "mdm_incorrect": mdm_metrics["incorrect"],
        "log_dir": str(logger.log_dir),
    }
    _append_dict_csv(run_csv, run_row, run_fields)

    step_fields = [
        "timestamp", "subject", "session", "run", "condition",
        "step", "distance_before", "distance_after", "distance_delta",
        "updates_before", "updates_after", "updates_this_run",
        "train_mean_hash", "mean_before_hash", "mean_after_hash",
    ]
    n_steps = max(len(dist_before), len(dist_after))
    for step_idx in range(n_steps):
        before = dist_before[step_idx] if step_idx < len(dist_before) else np.nan
        after = dist_after[step_idx] if step_idx < len(dist_after) else np.nan
        _append_dict_csv(
            step_csv,
            {
                "timestamp": timestamp,
                "subject": subject,
                "session": session,
                "run": run_label,
                "condition": condition,
                "step": step_idx + 1,
                "distance_before": before,
                "distance_after": after,
                "distance_delta": after - before,
                "updates_before": updates_before,
                "updates_after": updates_after,
                "updates_this_run": updates_this_run,
                "train_mean_hash": _hash_reference_list([train_refs[step_idx]]) if train_refs else "NA",
                "mean_before_hash": _hash_reference_list([refs_before[step_idx]]) if refs_before else "NA",
                "mean_after_hash": _hash_reference_list([refs_after[step_idx]]) if refs_after else "NA",
            },
            step_fields,
        )

    logger.log_event(
        "[RIEMANN_ADAPT_RUN] "
        f"csv={run_csv} condition={condition} "
        f"updates={updates_before}->{updates_after} "
        f"dist_mean={before_summary['mean']:.6f}->{after_summary['mean']:.6f} "
        f"final_acc={final_metrics['total_accuracy'] * 100:.1f}% "
        f"decision_acc={final_metrics['decision_accuracy'] * 100:.1f}%"
    )


def log_full_window_observer_summary(logger, probabilities, targets):
    """Report target performance and pairwise P(MI) correlations."""
    if not targets:
        return

    y = np.asarray(targets, dtype=int)
    y_binary = (y == 200).astype(int)
    logger.log_event("FULL-WINDOW Observer Summary (2.5 s, target-independent):")
    logger.log_event(
        "  Model          N     AUC   Accuracy   Corr(target)"
    )

    valid_scores = {}
    for model_name, values in probabilities.items():
        scores = np.asarray(values, dtype=float)
        valid = np.isfinite(scores)
        scores_valid = scores[valid]
        targets_valid = y[valid]
        y_binary_valid = y_binary[valid]
        n_valid = int(np.sum(valid))

        if n_valid:
            predictions = np.where(scores_valid >= 0.5, 200, 100)
            accuracy = float(np.mean(predictions == targets_valid))
        else:
            accuracy = np.nan

        auc = (
            float(roc_auc_score(y_binary_valid, scores_valid))
            if len(np.unique(y_binary_valid)) == 2
            else np.nan
        )
        corr_target = (
            float(np.corrcoef(scores_valid, y_binary_valid)[0, 1])
            if n_valid >= 2
            and np.std(scores_valid) > 0
            and np.std(y_binary_valid) > 0
            else np.nan
        )
        valid_scores[model_name] = scores
        logger.log_event(
            f"  {model_name:<12} {n_valid:>3}   {auc:>5.3f}   "
            f"{accuracy * 100:>7.1f}%   {corr_target:>+11.3f}"
        )

    model_names = list(valid_scores)
    logger.log_event("  Correlación Pearson entre P(MI) de observadores:")
    for idx, left_name in enumerate(model_names):
        for right_name in model_names[idx + 1:]:
            left = valid_scores[left_name]
            right = valid_scores[right_name]
            valid = np.isfinite(left) & np.isfinite(right)
            corr = (
                float(np.corrcoef(left[valid], right[valid])[0, 1])
                if np.sum(valid) >= 2
                and np.std(left[valid]) > 0
                and np.std(right[valid]) > 0
                else np.nan
            )
            logger.log_event(
                f"    {left_name:<10} vs {right_name:<10}: "
                f"r={corr:+.3f} (n={int(np.sum(valid))})"
            )


def resolve_observer_pmi_decision(p_mi_values, mode):
    """Return (prediction_label_or_none, reason) from observer P(MI) steps."""
    p_mi = [
        float(p) for p in p_mi_values
        if p is not None and np.isfinite(p)
    ]
    n_valid = len(p_mi)
    min_valid = max(1, int(getattr(config, "OBSERVER_MIN_STEPS", 3)))
    if n_valid < min_valid:
        return None, f"ambiguous_insufficient_steps n={n_valid}/{min_valid}"

    mi_label = 200
    rest_label = 100
    step_predictions = [mi_label if p >= 0.5 else rest_label for p in p_mi]
    n_mi = sum(1 for pred in step_predictions if pred == mi_label)
    n_rest = n_valid - n_mi
    vote_fraction = max(n_mi, n_rest) / n_valid
    min_vote_fraction = float(getattr(config, "FINAL_DECISION_MIN_VOTE_FRACTION", 0.60))
    mean_mi = sum(p_mi) / n_valid
    mean_rest = 1.0 - mean_mi

    raw_winner = mi_label if n_mi >= n_rest else rest_label
    raw_name = "MI" if raw_winner == mi_label else "REST"
    target_label = mi_label if mode == 0 else rest_label
    target_name = "MI" if target_label == mi_label else "REST"

    if vote_fraction < min_vote_fraction:
        return None, (
            f"ambiguous_split_vote target={target_name} raw_winner={raw_name} "
            f"mi={n_mi} rest={n_rest} vote_fraction={vote_fraction:.2f} "
            f"mean_mi={mean_mi:.3f}"
        )

    if raw_winner != target_label:
        return None, (
            f"ambiguous_opposite_evidence target={target_name} raw_winner={raw_name} "
            f"mi={n_mi} rest={n_rest} vote_fraction={vote_fraction:.2f} "
            f"mean_mi={mean_mi:.3f}"
        )

    if raw_winner == mi_label:
        if mean_mi < config.THRESHOLD_MI:
            return None, (
                f"ambiguous_low_mi_mean target={target_name} raw_winner={raw_name} "
                f"mean_mi={mean_mi:.3f}"
            )
        return mi_label, (
            f"accepted_mi mi={n_mi} rest={n_rest} "
            f"vote_fraction={vote_fraction:.2f} mean_mi={mean_mi:.3f}"
        )

    if mean_rest < config.THRESHOLD_REST:
        return None, (
            f"ambiguous_low_rest_mean target={target_name} raw_winner={raw_name} "
            f"mean_rest={mean_rest:.3f}"
        )
    return rest_label, (
        f"accepted_rest mi={n_mi} rest={n_rest} "
        f"vote_fraction={vote_fraction:.2f} mean_rest={mean_rest:.3f}"
    )


def draw_arrow_directional(screen, pos_x, pos_y, size, color, direction="right"):
    """
    Flecha completa: línea + triángulo (igual que offline).
    """
    line_len = size * 0.8
    tri_size = size // 2
    offset = 5  # px

    if direction == "right":
        line_start = (pos_x - line_len, pos_y)
        line_end = (pos_x + line_len - offset, pos_y)
        points = [
            (pos_x + line_len, pos_y),
            (pos_x + line_len - tri_size, pos_y - tri_size),
            (pos_x + line_len - tri_size, pos_y + tri_size),
        ]
    else:
        line_start = (pos_x + line_len, pos_y)
        line_end = (pos_x - line_len + offset, pos_y)
        points = [
            (pos_x - line_len, pos_y),
            (pos_x - line_len + tri_size, pos_y - tri_size),
            (pos_x - line_len + tri_size, pos_y + tri_size),
        ]

    pygame.draw.line(screen, color, line_start, line_end, 12)
    pygame.draw.polygon(screen, color, points)

# def draw_pretrial_screen_online(mode, time_ball_state=1):
#     """
#     Replica el look de OFFLINE en preparación:
#       - MI: cuadro rojo + flecha derecha
#       - REST: círculo azul + flecha izquierda
#       - time_balls en mode='single' en el indicador NEXT
#     """

#     screen.fill(config.black)
#     draw_fixation_cross(screen_width, screen_height)

#     pos_x = int(screen_width * NEXT_INDICATOR_POS[0])
#     pos_y = int(screen_height * NEXT_INDICATOR_POS[1])
#     base_size = int(min(screen_width, screen_height) * 0.08 * NEXT_INDICATOR_SCALE)

#     is_mi = (mode == 0)
#     next_color = (255, 50, 50) if is_mi else (0, 120, 255)

#     # 1) Shape background
#     if is_mi:
#         bg_rect = pygame.Rect(pos_x - base_size // 2, pos_y - base_size // 2, base_size, base_size)
#         pygame.draw.rect(screen, next_color, bg_rect)
#     else:
#         pygame.draw.circle(screen, next_color, (pos_x, pos_y), base_size // 2)

#     # 2) Single time-ball indicator (igual al offline)
#     draw_time_balls(
#         time_ball_state,
#         screen_width,
#         screen_height,
#         mode="single",
#         indicator_color=next_color,
#         single_pos=NEXT_INDICATOR_POS,
#         ball_radius=int(base_size * 0.4),
#     )

#     # 3) Texto de preparación
#     font_prep = pygame.font.SysFont(None, 96)
#     if is_mi:
#         prep_msg = f"Prepare to close {config.ARM_SIDE.upper()} hand"
#     else:
#         prep_msg = "Rest"

#     txt_surface = font_prep.render(prep_msg, True, config.white)
#     screen.blit(
#         txt_surface,
#         (screen_width // 2 - txt_surface.get_width() // 2, screen_height // 2 + 300),
#     )

#     # 4) Flecha direccional
#     arrow_dir = "right" if is_mi else "left"
#     draw_arrow_directional(screen, pos_x, pos_y, base_size // 2.5, (255, 255, 255), direction=arrow_dir)

#     pygame.display.flip()

# def draw_pretrial_screen_online(mode, elapsed_ms=0, total_ms=2500):
#     screen.fill(config.black)
#     draw_fixation_cross(screen_width, screen_height)

#     is_mi = (mode == 0)

#     # ── Figura geométrica grande (igual que offline) ──────────
#     if is_mi:
#         draw_arrow_fill(0, screen_width, screen_height, show_threshold=True)
#     else:
#         draw_ball_fill(0, screen_width, screen_height, show_threshold=True)

#     # ── Countdown bar ─────────────────────────────────────────
#     bar_w     = int(screen_width * 0.2)
#     bar_h     = 12
#     bar_x     = screen_width // 2 - bar_w // 2
#     bar_y     = screen_height // 2 + 245
#     progress  = min(elapsed_ms / total_ms, 1.0)
#     fill_w    = int(bar_w * progress)
#     bar_color = (255, 50, 50) if is_mi else (0, 120, 255)

#     pygame.draw.rect(screen, (60, 60, 60),
#                      (bar_x, bar_y, bar_w, bar_h), border_radius=6)
#     if fill_w > 0:
#         if is_mi:
#             pygame.draw.rect(screen, bar_color,
#                              (bar_x, bar_y, fill_w, bar_h), border_radius=6)
#         else:
#             pygame.draw.rect(screen, bar_color,
#                              (bar_x + bar_w - fill_w, bar_y, fill_w, bar_h),
#                              border_radius=6)

#     # ── Indicador pequeño arriba ───────────────────────────────
#     pos_x = int(screen_width * NEXT_INDICATOR_POS[0])
#     pos_y = int(screen_height * NEXT_INDICATOR_POS[1])
#     base_size = int(min(screen_width, screen_height) * 0.08 * NEXT_INDICATOR_SCALE)
#     next_color = (255, 50, 50) if is_mi else (0, 120, 255)

#     if is_mi:
#         bg_rect = pygame.Rect(pos_x - base_size // 2, pos_y - base_size // 2,
#                               base_size, base_size)
#         pygame.draw.rect(screen, next_color, bg_rect)
#     else:
#         pygame.draw.circle(screen, next_color, (pos_x, pos_y), base_size // 2)

#     draw_time_balls(1, screen_width, screen_height, mode="single",
#                     indicator_color=next_color, single_pos=NEXT_INDICATOR_POS,
#                     ball_radius=int(base_size * 0.4))

#     # ── Texto ─────────────────────────────────────────────────
#     font_prep = pygame.font.SysFont(None, 96)
#     prep_msg  = f"Prepare to close {config.ARM_SIDE.upper()} hand" if is_mi else "Rest"
#     txt_surface = font_prep.render(prep_msg, True, config.white)
#     screen.blit(txt_surface,
#                 (screen_width // 2 - txt_surface.get_width() // 2,
#                  screen_height // 2 + 300))

#     # ── Flecha direccional ────────────────────────────────────
#     arrow_dir = "right" if is_mi else "left"
#     draw_arrow_directional(screen, pos_x, pos_y,
#                            base_size // 2.5, (255, 255, 255),
#                            direction=arrow_dir)

#     pygame.display.flip()

def draw_neutral_intertrial_screen_online():
    """Neutral visual state used to hide REST preparation without changing logic."""
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)
    draw_ball_fill(0, screen_width, screen_height)
    draw_arrow_fill(0, screen_width, screen_height)
    draw_time_balls(0, screen_width, screen_height)
    pygame.display.flip()

def draw_pretrial_screen_online(mode, elapsed_ms=0, total_ms=2500, fill_progress=0.0):
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)

    is_mi = (mode == 0)
    if (
        not is_mi
        and bool(getattr(config, "ONLINE_REST_NEUTRAL_PREP_VISUAL", False))
    ):
        # REST preparation remains fully active for predictions/triggers/FES logic,
        # but visually matches the intertrial screen so it is not an explicit cue.
        draw_neutral_intertrial_screen_online()
        return

    # === Llenado complementario — igual que show_feedback ===
    prob = max(0, min(1, fill_progress))
    fill_correct   = prob          # figura de la clase correcta
    fill_incorrect = 1 - prob      # figura de la clase incorrecta

    # === Llenado BINARIO — solo una figura a la vez ===
    if is_mi:
        mi_fill   = fill_progress  # confianza MI
        rest_fill = 0.0            # REST siempre vacío en trial MI
        draw_arrow_fill(mi_fill,   screen_width, screen_height, show_threshold=True, fill_alpha=ONLINE_PREP_FILL_ALPHA)
        draw_ball_fill(rest_fill,  screen_width, screen_height, show_threshold=True, fill_alpha=ONLINE_PREP_FILL_ALPHA)
    else:
        rest_fill = fill_progress  # confianza REST
        mi_fill   = 0.0            # MI siempre vacío en trial REST
        draw_ball_fill(rest_fill,  screen_width, screen_height, show_threshold=True, fill_alpha=ONLINE_PREP_FILL_ALPHA)
        draw_arrow_fill(mi_fill,   screen_width, screen_height, show_threshold=True, fill_alpha=ONLINE_PREP_FILL_ALPHA)

    # ── Countdown bar ─────────────────────────────────────────
    bar_w    = int(screen_width * 0.3)
    bar_h    = int(getattr(config, "PREP_COUNTDOWN_BAR_HEIGHT", 28))
    bar_x    = screen_width // 2 - bar_w // 2
    bar_y    = screen_height // 2 + 245
    progress = min(elapsed_ms / total_ms, 1.0) if total_ms > 0 else 0
    fill_w   = int(bar_w * progress)
    bar_color = (255, 50, 50) if is_mi else (0, 120, 255)

    bar_radius = max(6, bar_h // 2)
    pygame.draw.rect(screen, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h), border_radius=bar_radius)
    if fill_w > 0:
        if is_mi:
            pygame.draw.rect(screen, bar_color, (bar_x, bar_y, fill_w, bar_h), border_radius=bar_radius)
        else:
            pygame.draw.rect(screen, bar_color, (bar_x + bar_w - fill_w, bar_y, fill_w, bar_h), border_radius=bar_radius)

    # ── Indicador pequeño arriba ───────────────────────────────
    pos_x = int(screen_width * NEXT_INDICATOR_POS[0])
    pos_y = int(screen_height * NEXT_INDICATOR_POS[1])
    base_size = int(min(screen_width, screen_height) * 0.08 * NEXT_INDICATOR_SCALE)
    next_color = (255, 50, 50) if is_mi else (0, 120, 255)

    if is_mi:
        bg_rect = pygame.Rect(pos_x - base_size // 2, pos_y - base_size // 2, base_size, base_size)
        pygame.draw.rect(screen, next_color, bg_rect)
    else:
        pygame.draw.circle(screen, next_color, (pos_x, pos_y), base_size // 2)

    draw_time_balls(1, screen_width, screen_height, mode="single",
                    indicator_color=next_color, single_pos=NEXT_INDICATOR_POS,
                    ball_radius=int(base_size * 0.4))

    # ── Texto ─────────────────────────────────────────────────
    font_prep = pygame.font.SysFont(None, 96)
    mi_label = getattr(config, "ARDUINO_MI_LABEL", "Open")
    prep_msg  = f"Prepare to {mi_label.lower()} {config.ARM_SIDE.upper()} hand" if is_mi else "Rest"
    txt_surface = font_prep.render(prep_msg, True, config.white)
    screen.blit(txt_surface, (screen_width // 2 - txt_surface.get_width() // 2, screen_height // 2 + 300))

    # ── Flecha direccional ────────────────────────────────────
    arrow_dir = "right" if is_mi else "left"
    draw_arrow_directional(screen, pos_x, pos_y, base_size // 2.5, (255, 255, 255), direction=arrow_dir)

    pygame.display.flip()

def main():
    logger.log_event("Resolving EEG data stream via LSL...")
    streams = resolve_stream('type', 'EEG')
    if not streams:
        logger.log_event("❌ No EEG stream found via LSL.", level="error")
        return
    inlet = StreamInlet(streams[0])
    eeg_state = EEGStreamState(inlet=inlet, config=config, logger=logger)

    nominal_srate = float(inlet.info().nominal_srate())
    if nominal_srate > 0 and not np.isclose(
        nominal_srate,
        float(config.FS),
        rtol=0.0,
        atol=0.1,
    ):
        logger.log_event(
            f"❌ LSL sampling rate mismatch: stream={nominal_srate:.3f} Hz "
            f"config={float(config.FS):.3f} Hz.",
            level="error",
        )
        return
    allowed_control_models = {
        "MDM", "LDA", "LDA_SHRINK", "LDA_SHRINKAGE",
        "LDA3", "LDA_3CH", "LDA_SHRINK_3CH", "COMPACT_LDA",
        "LR", "SVM",
    }
    configured_control = str(getattr(config, "PREP_CONTROL_MODEL", "MDM")).upper()
    if configured_control not in allowed_control_models:
        logger.log_event(
            "❌ Pilot preflight failed: unsupported PREP_CONTROL_MODEL="
            f"{configured_control}.",
            level="error",
        )
        return
    if model_type != "M2_LDA_shrink_MDM":
        logger.log_event(
            f"❌ Pilot preflight failed: unsupported model_type={model_type}.",
            level="error",
        )
        return

    trial_sequence = generate_trial_sequence(total_trials=config.TOTAL_TRIALS, max_repeats=config.MAX_REPEATS)
    current_trial = 0
    running = True
    clock = pygame.time.Clock()

    display_fixation_period(duration=12, eeg_state=eeg_state)
    try:
        eeg_state.assert_stream_fresh()
    except RuntimeError as exc:
        logger.log_event(f"❌ Pilot preflight failed: {exc}", level="error")
        return

    live_channels = list(eeg_state.channel_names or [])
    expected_channels = list(model_pkg.get("picks", []))
    missing_channels = [
        channel for channel in expected_channels
        if channel not in live_channels
    ]
    if missing_channels:
        logger.log_event(
            f"❌ Pilot preflight failed: missing model channels "
            f"{missing_channels}; live={live_channels}.",
            level="error",
        )
        return
    logger.log_event(
        f"✅ Pilot preflight passed: fs={nominal_srate:.1f} Hz "
        f"channels={expected_channels} control={configured_control}."
    )

    # Passive Riemannian adaptation diagnostics requested for the pilot:
    # capture the training reference and adaptive reference at run start.
    # These values are only used for CSV logging at the end of the run.
    riemann_train_refs_run = _as_valid_spd_list(model_pkg.get("mdm_recenter_refs"))
    riemann_refs_before_run = _as_valid_spd_list(getattr(_RC, "m2_prev_T", None))
    riemann_updates_before_run = int(getattr(_RC, "m2_rec_counter", 0) or 0)
    if riemann_train_refs_run is not None and riemann_refs_before_run is not None:
        _d0 = _summarize_distances(
            _riemann_reference_distances(
                riemann_train_refs_run,
                riemann_refs_before_run,
            )
        )
        logger.log_event(
            "[RIEMANN_ADAPT_START] "
            f"updates={riemann_updates_before_run} "
            f"dist_mean={_d0['mean']:.6f} dist_max={_d0['max']:.6f}"
        )
    else:
        logger.log_event(
            "[RIEMANN_ADAPT_START] unavailable — missing training/adaptive references"
        )

    # Ensure glove starts from the configured baseline state.
    if arduino:
        arduino_write(glove_cmd_rest())

    while running and current_trial < len(trial_sequence):
        # Compute the baseline from the fixation period immediately preceding
        # this trial. Never reuse preparation/FES activity from the prior trial.
        try:
            eeg_state.assert_stream_fresh()
            baseline_duration = float(
                getattr(
                    config,
                    "ONLINE_BASELINE_DURATION",
                    getattr(config, "BASELINE_DURATION", 1.0),
                )
            )
            baseline_end_offset = float(
                getattr(config, "ONLINE_BASELINE_END_OFFSET", 0.0)
            )
            eeg_state.compute_baseline(
                duration_sec=baseline_duration,
                end_offset_sec=baseline_end_offset,
            )
            logger.log_event(
                f"✅ Fresh EEG baseline computed before Trial {current_trial+1} "
                f"(duration={baseline_duration:.1f}s, "
                f"end_offset={baseline_end_offset:.1f}s)."
            )
        except (ValueError, RuntimeError) as exc:
            logger.log_event(
                f"❌ Cannot start Trial {current_trial+1}: {exc}",
                level="error",
            )
            running = False
            break

        logger.log_event(f"--- Trial {current_trial+1}/{len(trial_sequence)} START ---")

        # 1) Decide modo del trial
        mode = trial_sequence[current_trial]

        # 2) Trigger de cue PRIMERO — antes de dibujar
        cue_trig = config.TRIGGERS["MI_PREPARE"] if mode == 0 else config.TRIGGERS["REST_PREPARE"]
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"],
                        config.UDP_MARKER["PORT"], cue_trig, logger)

        # 3) Pantalla de preparación DESPUÉS del trigger
        draw_pretrial_screen_online(mode=mode, fill_progress=0.0)

        # 4) Waiting / Countdown + Clasificación durante preparación
        waiting_for_press = True
        countdown_start = None
        countdown_duration = 2500  # ms

        from Utils.experiment_utils import LeakyIntegrator
        prep_leaky      = LeakyIntegrator(alpha=config.INTEGRATOR_ALPHA)
        prep_predictions = []
        prep_all_probs   = []
        prep_confidence  = 0.0
        prep_prediction  = None
        prep_decision_reason = "no_decision_yet"
        prep_earlystop   = False
        prep_earlystop_elapsed = None
        prep_earlystop_step = None
        prep_earlystop_display_start = None
        prep_activation_confidence = None
        prep_display_confidence = 0.0
        prep_display_last_elapsed = 0
        next_classify_tick = None
        window_size_samples = int((config.CLASSIFY_WINDOW / 1000) * config.FS)
        accuracy_threshold  = config.THRESHOLD_MI if mode == 0 else config.THRESHOLD_REST
        prep_fes_active = False
        _RC.reset_m2_quality_state()

        # S012 fue entrenada con FES sensorial durante toda la preparación MI.
        # La prueba online conserva esa misma distribución desde el cue.
        if (
            mode == 0
            and FES_toggle == 1
            and getattr(config, "PREP_FEEDBACK_MODE", "FES") == "FES"
        ):
            send_udp_message(
                udp_socket_fes,
                config.UDP_FES["IP"],
                config.UDP_FES["PORT"],
                "FES_SENS_GO",
                logger=logger,
            )
            prep_fes_active = True
            logger.log_event(
                "⚡ FES_SENS_GO — inicio de preparación online "
                "(alineado con S012)"
            )


        while waiting_for_press:
            eeg_state.update()
            try:
                eeg_state.assert_stream_fresh()
            except RuntimeError as exc:
                logger.log_event(
                    f"❌ EEG stream lost during preparation: {exc}",
                    level="error",
                )
                running = False
                waiting_for_press = False
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting_for_press = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting_for_press = False

            if config.TIMING:
                if countdown_start is None:
                    countdown_start = pygame.time.get_ticks()
                    next_classify_tick = time.time() + config.STEP_SIZE
                    # M2_CUMULATIVE refreshes from the live EEG buffer on each classify tick.
                    # FROZEN_EPOCH_DEBUG keeps the legacy seed epoch for comparison only.
                    if _RC.model_pkg is not None:
                        if getattr(config, "PREP_DECODER_MODE", "M2_CUMULATIVE") == "FROZEN_EPOCH_DEBUG":
                            try:
                                _buf, _ = eeg_state.get_baseline_corrected_window(
                                    window_size_samples)
                            except Exception:
                                _buf = None
                        else:
                            _buf = None
                        _RC.prep_epoch    = _buf
                        _RC.m2_ch_idx     = None
                        _RC._m2_last_step = -1
                        _RC._m2_lda_probs = []
                        _RC._m2_lda_compact_probs = []
                        _RC._m2_mdm_probs = []
                        _RC._m2_lr_probs = []
                        _RC._m2_svm_probs = []
                        _RC._m2_shadow_records = []
                        _RC._m2_warmup_mdm_probs = []
                        _RC._m2_warmup_lda_probs = []

                elapsed = pygame.time.get_ticks() - countdown_start

                # ── Clasificar durante preparación ────────────────
                if not getattr(config, 'SIMULATION_MODE', False):
                    now = time.time()
                    if next_classify_tick and now >= next_classify_tick:
                        try:
                            if len(eeg_state.filtered_buffer) >= window_size_samples:
                                predictions_before = len(prep_predictions)
                                raw_confidence, prep_predictions, prep_all_probs = _RC.classify_real_time(
                                    eeg_state, window_size_samples,
                                    prep_all_probs, prep_predictions,
                                    mode, prep_leaky,
                                    elapsed_ms=elapsed
                                )
                                if len(prep_predictions) > predictions_before:
                                    if predictions_before == 0:
                                        prep_leaky.accumulated_probability = (
                                            raw_confidence
                                        )
                                        prep_confidence = raw_confidence
                                    else:
                                        prep_confidence = prep_leaky.update(
                                            raw_confidence
                                        )

                            # ── FORZAR PREDICCIÓN PARA PRUEBA SIN GORRA ──
                            # (fuera del if buffer, siempre se ejecuta)
                            if getattr(config, 'FORCE_MI_PREDICTION', False) and mode == 0:
                                prep_confidence = 0.7
                                prep_predictions.append(200)

                            correct_class = 200 if mode == 0 else 100
                            consecutive_required = int(getattr(config, "EARLYSTOP_CONSECUTIVE_PREDICTIONS", 1))
                            recent_predictions = prep_predictions[-consecutive_required:]
                            sustained_correct = (
                                len(recent_predictions) == consecutive_required and
                                all(pred == correct_class for pred in recent_predictions)
                            )

                            if (
                                mode == 0
                                and FES_toggle == 1
                                and getattr(config, "PREP_FEEDBACK_MODE", "FES") == "FES"
                                and not prep_fes_active
                            ):
                                if raw_confidence >= config.THRESHOLD_MI and sustained_correct:
                                    send_udp_message(udp_socket_fes, config.UDP_FES["IP"],
                                                     config.UDP_FES["PORT"], "FES_SENS_GO", logger=logger)
                                    prep_fes_active = True

                            if (
                               getattr(config, "PREP_EARLY_STOP_ENABLED", False)
                               and not prep_earlystop
                               and len(prep_predictions) >= config.MIN_PREDICTIONS and
                               prep_confidence >= accuracy_threshold
                               and sustained_correct
                            ):
                                prep_prediction = prep_predictions[-1]
                                prep_activation_confidence = prep_confidence
                                prep_decision_reason = (
                                    f"earlystop_sustained confidence={prep_confidence:.3f} "
                                    f"recent_n={consecutive_required}"
                                )
                                prep_earlystop  = True
                                prep_earlystop_step = _RC._m2_last_step + 1
                                if prep_earlystop_elapsed is None:
                                    prep_earlystop_elapsed = elapsed
                                    prep_earlystop_display_start = prep_display_confidence
                                logger.log_event(
                                    f"✅ Prep early stop — "
                                    f"confidence={prep_confidence:.2f}, "
                                    f"mode={'MI' if mode==0 else 'REST'}"
                                )
                            next_classify_tick += config.STEP_SIZE
                        except Exception as e:
                            logger.log_event(f"⚠️ Classify error during prep: {e}")

                # Suavizado exclusivamente visual: las decisiones siguen usando
                # prep_confidence sin retrasos ni modificaciones.
                visual_dt = max(0, elapsed - prep_display_last_elapsed)
                prep_display_last_elapsed = elapsed
                ramp_ms = max(
                    1,
                    int(getattr(config, "ONLINE_PREP_VISUAL_RAMP_MS", 250)),
                )
                max_visual_delta = visual_dt / ramp_ms
                visual_delta = prep_confidence - prep_display_confidence
                prep_display_confidence += max(
                    -max_visual_delta,
                    min(max_visual_delta, visual_delta),
                )

                # Animate bar to 1.0 after early stop (motivational)
                if prep_earlystop and prep_earlystop_elapsed is not None:
                    remaining = max(1, countdown_duration - prep_earlystop_elapsed)
                    t_anim = min(1.0, (elapsed - prep_earlystop_elapsed) / remaining)
                    animation_start = (
                        prep_earlystop_display_start
                        if prep_earlystop_display_start is not None
                        else prep_display_confidence
                    )
                    fill_display = animation_start + (1.0 - animation_start) * t_anim
                else:
                    fill_display = prep_display_confidence
                draw_pretrial_screen_online(
                    mode=mode,
                    elapsed_ms=elapsed,
                    total_ms=countdown_duration,
                    fill_progress=fill_display
                )

                # ── Feedback durante preparación (solo MI) ────────────
                if mode == 0 and getattr(config, 'PREP_FEEDBACK_MODE', 'FES') == 'GLOVE':
                    fes_should_be_on = prep_confidence >= config.THRESHOLD_MI
                    if fes_should_be_on:
                        if arduino:
                            arduino_write(glove_cmd_mi())
                            pygame.time.wait(150)
                            arduino_write(glove_cmd_rest())
                            logger.log_event("🤚 Glove pulse — prep MI feedback")
                        prep_fes_active = True
                    else:
                        prep_fes_active = False

                if elapsed >= countdown_duration:
                    waiting_for_press = False
            else:
                draw_pretrial_screen_online(mode=mode)

            clock.tick(60)

        # ── Apagar feedback al terminar preparación ───────────
        if mode == 0 and prep_fes_active:
            if getattr(config, 'PREP_FEEDBACK_MODE', 'FES') == 'FES' and FES_toggle == 1:
                send_udp_message(udp_socket_fes, config.UDP_FES["IP"],
                                config.UDP_FES["PORT"], "FES_STOP",
                                logger=logger, quiet=True)
            # GLOVE no necesita apagado — el pulso ya se cerró solo
        prep_fes_active = False
        if not running:
            logger.log_event(
                f"❌ Trial {current_trial+1} aborted before decision output."
            )
            break

        # Observadores de ventana completa: se evalúan siempre al terminar
        # los 2.5 s, incluso si el control hizo early stop.
        full_target = 200 if mode == 0 else 100
        full_probabilities = {}
        if _RC.model_pkg is not None:
            try:
                full_epoch, _ = eeg_state.get_baseline_corrected_window(
                    window_size_samples
                )
                channel_names = list(eeg_state.channel_names or [])
                full_indices = [
                    channel_names.index(channel)
                    for channel in model_pkg["picks"]
                ]
                full_epoch_ch = full_epoch[full_indices, :]
                quality_ok, quality_reason, _ = _RC._check_eeg_quality(
                    full_epoch_ch
                )
                if quality_ok:
                    full_probabilities = _RC.predict_m2_full_window(
                        model_pkg,
                        full_epoch_ch,
                        recenter=bool(getattr(config, "RECENTERING", 0)),
                    )
                else:
                    logger.log_event(
                        f"[FULL_WINDOW_OBSERVERS] trial={current_trial+1} "
                        f"skipped=BAD_EEG reason={quality_reason}"
                    )
            except Exception as exc:
                logger.log_event(
                    f"[FULL_WINDOW_OBSERVERS] trial={current_trial+1} "
                    f"skipped=ERROR reason={exc}"
                )

        full_window_targets.append(full_target)
        for observer_name in full_window_probabilities:
            full_window_probabilities[observer_name].append(
                full_probabilities.get(observer_name, np.nan)
            )

        if full_probabilities:
            full_parts = []
            for observer_name in full_window_probabilities:
                p_mi = full_probabilities.get(observer_name)
                if p_mi is None:
                    continue
                prediction_label = 200 if p_mi >= 0.5 else 100
                full_parts.append(
                    f"{observer_name}_PMI={p_mi:.3f} "
                    f"{observer_name}_pred={prediction_label}"
                )
            logger.log_event(
                f"[FULL_WINDOW_OBSERVERS] trial={current_trial+1} "
                f"target={full_target} " + " | ".join(full_parts)
            )

        # Diagnóstico contrafactual: cada modelo decide por sí mismo. La clase
        # real se usa únicamente después para calificar la decisión.
        shadow_records = list(
            getattr(_RC, "_m2_shadow_records", [])
        )
        trial_shadow_results = {}
        if getattr(config, "SHADOW_MODEL_ANALYSIS_ENABLED", True):
            for shadow_model_name in shadow_model_names:
                shadow_result = evaluate_shadow_early_stop(
                    shadow_records,
                    shadow_model_name,
                    full_target,
                )
                stability_result = find_shadow_stabilization(
                    shadow_records,
                    shadow_model_name,
                    full_target,
                    model_pkg,
                )
                shadow_earlystop_results[shadow_model_name].append(
                    shadow_result
                )
                shadow_stability_results[shadow_model_name].append(
                    stability_result
                )
                trial_shadow_results[shadow_model_name] = shadow_result

                shadow_prediction = (
                    "MI"
                    if shadow_result["prediction"] == 200
                    else "REST"
                    if shadow_result["prediction"] == 100
                    else "NONE"
                )
                shadow_correct = (
                    shadow_result["correct"]
                    if shadow_result["correct"] is not None
                    else "NA"
                )
                logger.log_event(
                    f"[SHADOW_EARLYSTOP] trial={current_trial+1} "
                    f"model={shadow_model_name} "
                    f"early_stop={shadow_result['triggered']} "
                    f"step={shadow_result['step'] or 'NA'} "
                    f"time={shadow_result['time'] if shadow_result['time'] is not None else 'NA'} "
                    f"predicted_class={shadow_prediction} "
                    f"confidence={shadow_result['confidence'] if shadow_result['confidence'] is not None else 'NA'} "
                    f"target={'MI' if full_target == 200 else 'REST'} "
                    f"correct={shadow_correct}"
                )
                stable_prediction = (
                    "MI"
                    if stability_result["prediction"] == 200
                    else "REST"
                    if stability_result["prediction"] == 100
                    else "NONE"
                )
                stable_correct = (
                    stability_result["correct"]
                    if stability_result["correct"] is not None
                    else "NA"
                )
                logger.log_event(
                    f"[SHADOW_STABILITY] trial={current_trial+1} "
                    f"model={shadow_model_name} "
                    f"stabilization_step={stability_result['step'] or 'NA'} "
                    f"time={stability_result['time'] if stability_result['time'] is not None else 'NA'} "
                    f"stable_class={stable_prediction} "
                    f"target={'MI' if full_target == 200 else 'REST'} "
                    f"correct={stable_correct} "
                    f"reason={stability_result['reason']}"
                )

            triggered_results = [
                result for result in trial_shadow_results.values()
                if result["triggered"]
            ]
            if triggered_results:
                fastest_step = min(
                    result["step"] for result in triggered_results
                )
                fastest = [
                    result["model"] for result in triggered_results
                    if result["step"] == fastest_step
                ]
                logger.log_event(
                    f"[SHADOW_FASTEST_MODEL] trial={current_trial+1} "
                    f"models={'+'.join(fastest)} step={fastest_step}"
                )
            else:
                logger.log_event(
                    f"[SHADOW_FASTEST_MODEL] trial={current_trial+1} "
                    "models=NONE step=NA"
                )

        # El endpoint siempre se calcula como métrica target-independent del
        # modelo de control configurado, incluso si la activación temprana ya
        # quedó bloqueada.
        configured_control_model = str(
            getattr(config, "PREP_CONTROL_MODEL", "MDM")
        ).upper()
        if configured_control_model in {
            "LDA3", "LDA_3CH", "LDA_SHRINK_3CH", "COMPACT_LDA",
        }:
            endpoint_control_probs = _RC._m2_lda_compact_probs
        elif configured_control_model in {"LDA", "LDA_SHRINK", "LDA_SHRINKAGE"}:
            endpoint_control_probs = _RC._m2_lda_probs
        elif configured_control_model == "LR":
            endpoint_control_probs = _RC._m2_lr_probs
        elif configured_control_model == "SVM":
            endpoint_control_probs = _RC._m2_svm_probs
        else:
            endpoint_control_probs = _RC._m2_mdm_probs

        endpoint_prediction, endpoint_reason = resolve_endpoint_control_decision(
            endpoint_control_probs,
            _RC._m2_last_step,
            model_pkg,
            shadow_records=shadow_records,
        )
        logger.log_event(
            f"[PREP_DECISION_ENDPOINT] trial={current_trial+1} "
            f"prediction={endpoint_prediction if endpoint_prediction is not None else 'AMBIGUOUS'} "
            f"reason={endpoint_reason}"
        )

        used_endpoint_fallback = not (
            prep_earlystop and prep_prediction is not None
        )
        decision_validation_applied = False
        if not used_endpoint_fallback:
            prep_decision_reason = (
                f"threshold_activation step={prep_earlystop_step} "
                f"confidence={prep_activation_confidence:.3f}"
            )
            mdm_operational_decision_original = prep_prediction
            if (
                getattr(config, "ENDPOINT_VALIDATION_ENABLED", True)
                and getattr(config, "EARLYSTOP_VALIDATION_ENABLED", True)
            ):
                decision_validation_applied = True
                (
                    final_validated_decision,
                    endpoint_validation_reason,
                    validator_predictions,
                ) = validate_mdm_decision_with_observers(
                    mdm_operational_decision_original,
                    shadow_records,
                    model_pkg,
                    step_index=prep_earlystop_step - 1,
                    context_label="earlystop",
                )
                prep_decision_reason = (
                    f"{prep_decision_reason}; "
                    f"validation={endpoint_validation_reason}"
                )
            else:
                final_validated_decision = mdm_operational_decision_original
                endpoint_validation_reason = "not_applied_mdm_early_stop"
                validator_predictions = {}
        else:
            mdm_operational_decision_original = endpoint_prediction
            if getattr(config, "ENDPOINT_VALIDATION_ENABLED", True):
                decision_validation_applied = True
                (
                    final_validated_decision,
                    endpoint_validation_reason,
                    validator_predictions,
                ) = validate_mdm_endpoint_with_observers(
                    mdm_operational_decision_original,
                    shadow_records,
                    model_pkg,
                )
            else:
                final_validated_decision = mdm_operational_decision_original
                endpoint_validation_reason = "validation_disabled"
                validator_predictions = {}
            prep_decision_reason = (
                f"endpoint_fallback {endpoint_reason}; "
                f"validation={endpoint_validation_reason}"
            )

        # Aggregate endpoint-validation outcomes without feeding target
        # information back into the decision itself.
        if used_endpoint_fallback:
            endpoint_validation_stats["n_endpoint_fallbacks"] += 1
            if mdm_operational_decision_original is None:
                endpoint_validation_stats["mdm_already_ambiguous"] += 1
            elif getattr(config, "ENDPOINT_VALIDATION_ENABLED", True):
                agreeing_validators = {
                    name for name, prediction in validator_predictions.items()
                    if prediction == mdm_operational_decision_original
                }
                if final_validated_decision == mdm_operational_decision_original:
                    if agreeing_validators == {"LDA", "LR"}:
                        endpoint_validation_stats["accepted_by_both"] += 1
                    elif "LDA" in agreeing_validators:
                        endpoint_validation_stats["accepted_by_lda"] += 1
                    elif "LR" in agreeing_validators:
                        endpoint_validation_stats["accepted_by_lr"] += 1
                elif final_validated_decision is None:
                    endpoint_validation_stats["rejected_to_ambiguous"] += 1
                    if mdm_operational_decision_original == full_target:
                        endpoint_validation_stats["correct_mdm_rejected"] += 1
                    else:
                        endpoint_validation_stats["errors_prevented"] += 1

        prep_prediction = final_validated_decision
        mdm_operational_decisions_original.append(
            mdm_operational_decision_original
        )
        final_validated_decisions.append(final_validated_decision)
        logger.log_event(
            f"[ENDPOINT_VALIDATION] trial={current_trial+1} "
            f"applied={decision_validation_applied} "
            f"context={'endpoint' if used_endpoint_fallback else 'earlystop'} "
            f"mdm_original={mdm_operational_decision_original if mdm_operational_decision_original is not None else 'AMBIGUOUS'} "
            f"lda={validator_predictions.get('LDA', 'NA')} "
            f"lr={validator_predictions.get('LR', 'NA')} "
            f"final={final_validated_decision if final_validated_decision is not None else 'AMBIGUOUS'} "
            f"reason={endpoint_validation_reason}"
        )
        logger.log_event(
            f"[DECISION_LAYERS] trial={current_trial+1} "
            f"mdm_operational_decision_original="
            f"{mdm_operational_decision_original if mdm_operational_decision_original is not None else 'AMBIGUOUS'} "
            f"final_validated_decision="
            f"{final_validated_decision if final_validated_decision is not None else 'AMBIGUOUS'}"
        )
        logger.log_event(
            f"[PREP_OPERATIONAL_DECISION] trial={current_trial+1} "
            f"prediction={prep_prediction if prep_prediction is not None else 'AMBIGUOUS'} "
            f"activated_early={prep_earlystop} reason={prep_decision_reason}"
        )

        # La votación histórica queda solo como observador para comparación.
        vote_prep_prediction, vote_prep_reason = resolve_prep_decision_raw(
            prep_predictions, prep_all_probs
        )
        logger.log_event(
            f"[PREP_DECISION_VOTE_OBSERVER] trial={current_trial+1} "
            f"prediction={vote_prep_prediction if vote_prep_prediction is not None else 'AMBIGUOUS'} "
            f"target={200 if mode == 0 else 100} reason={vote_prep_reason}"
        )
        raw_predictions_list.append(endpoint_prediction)
        raw_ground_truth_list.append(200 if mode == 0 else 100)
        early_max_steps = int(getattr(config, "EARLY_RAW_DECISION_MAX_STEPS", 8))
        early_prep_prediction, early_prep_reason = resolve_prep_decision_raw(
            prep_predictions[:early_max_steps],
            prep_all_probs[:early_max_steps]
        )
        logger.log_event(
            f"[PREP_DECISION_RAW_EARLY] trial={current_trial+1} "
            f"steps=1-{early_max_steps} "
            f"prediction={early_prep_prediction if early_prep_prediction is not None else 'AMBIGUOUS'} "
            f"target={200 if mode == 0 else 100} reason={early_prep_reason}"
        )
        raw_early_predictions_list.append(early_prep_prediction)
        raw_early_ground_truth_list.append(200 if mode == 0 else 100)

        if _RC.model_pkg is not None and _RC._m2_lda_probs:
            _label         = "MI" if mode == 0 else "REST"
            _earlystop_step = prep_earlystop_step if prep_earlystop else "—"
            logger.log_event(
                f"[TRIAL_SUMMARY] {_label} | trial={current_trial+1} | "
                f"early_stop={prep_earlystop} (paso {_earlystop_step})"
            )
            logger.log_event(
                f"  MDM P(MI)        : {_RC._m2_mdm_probs}"
            )
            logger.log_event(
                f"  LDA (validación) : {_RC._m2_lda_probs}"
            )
            if _RC._m2_lda_compact_probs:
                logger.log_event(
                    "  LDA 3ch observer : "
                    f"{_RC._m2_lda_compact_probs}"
                )
            logger.log_event(
                f"  LR observer P(MI): {_RC._m2_lr_probs}"
            )
            logger.log_event(
                f"  SVM observer P(MI): {_RC._m2_svm_probs}"
            )
            master_lda_prediction, master_lda_reason = resolve_observer_pmi_decision(
                _RC._m2_lda_probs, mode
            )
            logger.log_event(
                f"[LDA_OBSERVER_DECISION] trial={current_trial+1} "
                f"model=master_lda "
                f"prediction={master_lda_prediction if master_lda_prediction is not None else 'AMBIGUOUS'} "
                f"target={200 if mode == 0 else 100} reason={master_lda_reason}"
            )
            if _RC._m2_lda_compact_probs:
                compact_lda_prediction, compact_lda_reason = (
                    resolve_observer_pmi_decision(
                        _RC._m2_lda_compact_probs,
                        mode,
                    )
                )
                logger.log_event(
                    f"[LDA_3CH_OBSERVER_DECISION] trial={current_trial+1} "
                    "model=lda_shrink_3ch "
                    f"prediction={compact_lda_prediction if compact_lda_prediction is not None else 'AMBIGUOUS'} "
                    f"target={200 if mode == 0 else 100} "
                    f"reason={compact_lda_reason}"
                )
            lr_prediction, lr_reason = resolve_observer_pmi_decision(
                _RC._m2_lr_probs, mode
            )
            logger.log_event(
                f"[LR_OBSERVER_DECISION] trial={current_trial+1} "
                f"model=master_lr "
                f"prediction={lr_prediction if lr_prediction is not None else 'AMBIGUOUS'} "
                f"target={200 if mode == 0 else 100} reason={lr_reason}"
            )
            svm_prediction, svm_reason = resolve_observer_pmi_decision(
                _RC._m2_svm_probs, mode
            )
            logger.log_event(
                f"[SVM_OBSERVER_DECISION] trial={current_trial+1} "
                f"model=master_svm "
                f"prediction={svm_prediction if svm_prediction is not None else 'AMBIGUOUS'} "
                f"target={200 if mode == 0 else 100} reason={svm_reason}"
            )
            if _RC.observer_model_pkg is not None:
                logger.log_event(
                    f"  Warmup MDM P(MI) : {_RC._m2_warmup_mdm_probs}"
                )
                warmup_mdm_prediction, warmup_mdm_reason = resolve_observer_pmi_decision(
                    _RC._m2_warmup_mdm_probs, mode
                )
                logger.log_event(
                    f"[MDM_OBSERVER_DECISION] trial={current_trial+1} "
                    f"model=warmup_mdm "
                    f"prediction={warmup_mdm_prediction if warmup_mdm_prediction is not None else 'AMBIGUOUS'} "
                    f"target={200 if mode == 0 else 100} reason={warmup_mdm_reason}"
                )
                logger.log_event(
                    f"  Warmup LDA P(MI) : {_RC._m2_warmup_lda_probs}"
                )
                warmup_lda_prediction, warmup_lda_reason = resolve_observer_pmi_decision(
                    _RC._m2_warmup_lda_probs, mode
                )
                logger.log_event(
                    f"[LDA_OBSERVER_DECISION] trial={current_trial+1} "
                    f"model=warmup_lda "
                    f"prediction={warmup_lda_prediction if warmup_lda_prediction is not None else 'AMBIGUOUS'} "
                    f"target={200 if mode == 0 else 100} reason={warmup_lda_reason}"
                )
            _RC._m2_mdm_probs = []
            _RC._m2_lda_probs = []
            _RC._m2_lda_compact_probs = []
            _RC._m2_lr_probs = []
            _RC._m2_svm_probs = []
            _RC._m2_shadow_records = []
            _RC._m2_warmup_mdm_probs = []
            _RC._m2_warmup_lda_probs = []
            _RC._m2_last_step = -1

        if getattr(_RC, "_m2_quality_reject_count", 0):
            logger.log_event(
                f"[EEG_QUALITY_SUMMARY] rejected_windows={_RC._m2_quality_reject_count} "
                f"decision={'none' if not prep_predictions else 'partial'}"
            )

        # Actualizar recentering Riemanniano M2 con el epoch de este trial
        if _RC.model_pkg is not None and getattr(config, "RECENTERING", 0):
            _recenter_refs_before_trial = _as_valid_spd_list(getattr(_RC, "m2_prev_T", None))
            _recenter_counter_before_trial = int(getattr(_RC, "m2_rec_counter", 0) or 0)
            _recenter_seen_before_trial = int(getattr(_RC, "m2_rec_seen_trials", 0) or 0)
            _recenter_valid_m2_steps = _valid_m2_steps_from_shadow_records(
                shadow_records
            )
            _recenter_bad_eeg = bool(getattr(_RC, "_m2_quality_bad_trial", False))
            _recenter_target_label = 200 if mode == 0 else 100
            _recenter_decision_source = _decision_source_label(
                used_endpoint_fallback,
                prep_earlystop,
            )
            _RC.update_m2_recentering(
                prep_prediction=mdm_operational_decision_original,
                target_label=_recenter_target_label,
                prep_confidence=prep_confidence,
            )
            _recenter_counter_after_trial = int(getattr(_RC, "m2_rec_counter", 0) or 0)
            _recenter_seen_after_trial = int(getattr(_RC, "m2_rec_seen_trials", 0) or 0)
            _run_label_update = _run_label_from_logger(logger)
            _condition_update = (
                "FES" if int(getattr(config, "FES_toggle", 0)) else "NO_FES"
            )
            if _recenter_counter_after_trial > _recenter_counter_before_trial:
                try:
                    log_riemann_update_event(
                        logger,
                        subject=recording_subject,
                        session=logger.log_base.name,
                        run_label=_run_label_update,
                        condition=_condition_update,
                        trial_number=current_trial + 1,
                        update_class=_class_label_from_code(
                            mdm_operational_decision_original
                        ),
                        train_refs=riemann_train_refs_run,
                        refs_before=_recenter_refs_before_trial,
                        refs_after=getattr(_RC, "m2_prev_T", None),
                        updates_at_session_start=riemann_updates_before_run,
                        updates_before=_recenter_counter_before_trial,
                        updates_after=_recenter_counter_after_trial,
                        prep_confidence=prep_confidence,
                        target_label=_recenter_target_label,
                        prediction_label=mdm_operational_decision_original,
                        update_reason="ACCEPTED_CORRECT_CONFIDENT_VALID_EEG",
                        decision_source=_recenter_decision_source,
                        valid_m2_steps=_recenter_valid_m2_steps,
                    )
                except Exception as exc:
                    logger.log_event(
                        f"⚠️ [RIEMANN_ADAPT_UPDATE] logging failed: {exc}",
                        level="warning",
                    )
            else:
                try:
                    _recenter_rejection_reason = _infer_recenter_rejection_reason(
                        prediction_label=mdm_operational_decision_original,
                        target_label=_recenter_target_label,
                        prep_confidence=prep_confidence,
                        bad_eeg=_recenter_bad_eeg,
                        model_available=(_RC.model_pkg is not None),
                        prep_epoch_available=(getattr(_RC, "prep_epoch", None) is not None),
                        valid_m2_steps=_recenter_valid_m2_steps,
                        seen_before=_recenter_seen_before_trial,
                        seen_after=_recenter_seen_after_trial,
                        updated_before=_recenter_counter_before_trial,
                        updated_after=_recenter_counter_after_trial,
                    )
                    log_riemann_rejection_event(
                        logger,
                        subject=recording_subject,
                        session=logger.log_base.name,
                        run_label=_run_label_update,
                        condition=_condition_update,
                        trial_number=current_trial + 1,
                        target_label=_recenter_target_label,
                        prediction_label=mdm_operational_decision_original,
                        reason=_recenter_rejection_reason,
                        prep_confidence=prep_confidence,
                        valid_m2_steps=_recenter_valid_m2_steps,
                        bad_eeg=_recenter_bad_eeg,
                        early_stop=prep_earlystop,
                    )
                except Exception as exc:
                    logger.log_event(
                        f"⚠️ [RIEMANN_ADAPT_REJECT] logging failed: {exc}",
                        level="warning",
                    )

        # === Resumen de probabilidades de la fase de preparación ===
        if prep_all_probs:
            import numpy as _np
            _pa = _np.array(prep_all_probs)   # (N, 3): [time, P_rest, P_mi]
            _p_rest, _p_mi = _pa[:, 1], _pa[:, 2]
            _label = "MI" if mode == 0 else "REST"
            logger.log_event(
                f"[PROBS] {_label} | n={len(_p_mi)} | "
                f"P(MI):   mean={_p_mi.mean():.3f}  min={_p_mi.min():.3f}  max={_p_mi.max():.3f} | "
                f"P(REST): mean={_p_rest.mean():.3f}  min={_p_rest.min():.3f}  max={_p_rest.max():.3f} | "
                f"integrator={prep_confidence:.3f}  earlystop={prep_earlystop}"
            )

        if not running:
            break

        mode = trial_sequence[current_trial]

        # # -----------------------------------------------------------
        # # PHASE 1: EFFORT (Sensory FES Only)
        # # -----------------------------------------------------------
        # prediction, confidence, leaky_integrator, trial_probs, earlystop_flag = show_feedback(
        #     duration=config.TIME_MI,
        #     mode=mode,
        #     eeg_state=eeg_state
        # )

        # append_trial_probabilities_to_csv(
        #     trial_probabilities=trial_probs, mode=mode, trial_number=current_trial + 1,
        #     predicted_label=prediction, early_cutout=earlystop_flag,
        #     mi_threshold=config.THRESHOLD_MI, rest_threshold=config.THRESHOLD_REST,
        #     logger=logger, phase="MI" if mode == 0 else "REST"
        # )

        # PHASE 1: EFFORT (Sensory FES Only)
        # ── Trigger GO ───────────────────────────────────────────
        go_trig = config.TRIGGERS["MI_BEGIN"] if mode == 0 else config.TRIGGERS["REST_BEGIN"]
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"],
                         config.UDP_MARKER["PORT"], go_trig, logger)

        # ── Feedback visual 5s — llenar figura si correcto ───────
        correct_class = 200 if mode == 0 else 100
        prediction    = prep_prediction
        earlystop_flag = prep_earlystop
        motor_fes_active = False

        # ── Cerrar guante al inicio del feedback si fue correcto (MI) ──
        if prediction == correct_class and mode == 0:
            if arduino:
                arduino_write(glove_cmd_mi())
                logger.log_event(
                    f"🤚 Glove {getattr(config, 'ARDUINO_MI_LABEL', 'Open').lower()} — feedback reward (5s)"
                )
            if FES_toggle == 1:
                send_udp_message(udp_socket_fes, config.UDP_FES["IP"],
                                config.UDP_FES["PORT"], "FES_MOTOR_GO", logger=logger)
                motor_fes_active = True
                logger.log_event("⚡ FES_MOTOR_GO — feedback reward")    

        start_feedback = time.time()
        clock_fb = pygame.time.Clock()
        while time.time() - start_feedback < config.TIME_MI:
            eeg_state.update()
            progress = (time.time() - start_feedback) / config.TIME_MI
            screen.fill(config.black)
            draw_fixation_cross(screen_width, screen_height)

            if prediction == correct_class:
                if mode == 0:
                    draw_arrow_fill(progress, screen_width, screen_height, show_threshold=False, fill_alpha=ONLINE_EXEC_FILL_ALPHA)
                    draw_ball_fill(0, screen_width, screen_height, show_threshold=False, fill_alpha=ONLINE_EXEC_FILL_ALPHA)
                else:
                    draw_ball_fill(progress, screen_width, screen_height, show_threshold=False, fill_alpha=ONLINE_EXEC_FILL_ALPHA)
                    draw_arrow_fill(0, screen_width, screen_height, show_threshold=False, fill_alpha=ONLINE_EXEC_FILL_ALPHA)
            else:
                draw_arrow_fill(0, screen_width, screen_height, show_threshold=False, fill_alpha=ONLINE_EXEC_FILL_ALPHA)
                draw_ball_fill(0, screen_width, screen_height, show_threshold=False, fill_alpha=ONLINE_EXEC_FILL_ALPHA)

            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            clock_fb.tick(60)

        # ── Abrir guante al terminar feedback ──────────────────────────
        if arduino:
            arduino_write(glove_cmd_rest())

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            clock_fb.tick(60)

        if motor_fes_active:
            send_udp_message(
                udp_socket_fes,
                config.UDP_FES["IP"],
                config.UDP_FES["PORT"],
                "FES_STOP",
                logger=logger,
                quiet=True,
            )
            logger.log_event("⚡ FES_STOP — fin del feedback motor")

        # Trigger fin
        end_trig = config.TRIGGERS["MI_END"] if mode == 0 else config.TRIGGERS["REST_END"]
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"],
                         config.UDP_MARKER["PORT"], end_trig, logger)

        append_trial_probabilities_to_csv(
            trial_probabilities=prep_all_probs, mode=mode,
            trial_number=current_trial + 1,
            predicted_label=final_validated_decision, early_cutout=earlystop_flag,
            mi_threshold=config.THRESHOLD_MI, rest_threshold=config.THRESHOLD_REST,
            logger=logger, phase="MI" if mode == 0 else "REST"
        )

        logger.log_trial_summary(
            trial_number=current_trial + 1,
            true_label=correct_class,
            predicted_label=final_validated_decision,
            early_cutout=earlystop_flag,
            accuracy_threshold=accuracy_threshold,
            confidence=prep_confidence,
            num_predictions=len(prep_predictions)
        )

        # -----------------------------------------------------------
        # PHASE 2: REWARD (Motor FES + Glove + Robot)
        # -----------------------------------------------------------

        predictions_list.append(final_validated_decision)
        ground_truth_list.append(200 if mode == 0 else 100)

        if mode == 0:  # MI Trial
            if prediction == 200:  # SUCCESS (Threshold reached)

                # 1) CLOSE GLOVE (Reward Trigger)
                # if arduino:
                #     arduino_write(b'1')
                #     logger.log_event("✅ Prediction Success -> Closing Glove (Reward)")

                # 3) ROBOT
                messages = ["Correct", f"Hand {getattr(config, 'ARDUINO_MI_LABEL', 'Open').lower()}"]
                colors = [config.green, config.green]
                send_udp_message(
                    udp_socket_marker,
                    config.UDP_MARKER["IP"],
                    config.UDP_MARKER["PORT"],
                    config.TRIGGERS["ROBOT_BEGIN"],
                    logger=logger
                )

                display_multiple_messages_with_udp(
                    messages=messages,
                    colors=colors,
                    offsets=[-100, 100],
                    duration=0.01,
                    udp_messages=[random.choice(config.ROBOT_TRAJECTORY), config.ROBOT_OPCODES["GO"]],
                    udp_socket=udp_socket_robot,
                    udp_ip=config.UDP_ROBOT["IP"],
                    udp_port=config.UDP_ROBOT["PORT"],
                    logger=logger,
                    eeg_state=eeg_state
                )

                final_class, robot_probs, early = hold_messages_and_classify(
                    messages, colors, [-100, 100],
                    config.TIME_ROB, 0,
                    udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"],
                    eeg_state, prep_leaky
                )

                # Robot home
                send_udp_message(
                    udp_socket_robot,
                    config.UDP_ROBOT["IP"],
                    config.UDP_ROBOT["PORT"],
                    config.ROBOT_OPCODES["HOME"],
                    logger=logger,
                    expect_ack=True
                )

            else:  # FAIL (Threshold not reached)
                if arduino:
                    arduino_write(glove_cmd_rest())
                display_multiple_messages_with_udp(
                    ["Incorrect", "Hand Stationary"],
                    [config.red, config.white],
                    [-100, 100],
                    config.TIME_STATIONARY,
                    None,
                    udp_socket_robot,
                    config.UDP_ROBOT["IP"],
                    config.UDP_ROBOT["PORT"],
                    logger,
                    eeg_state
                )

        else:  # REST Trial
            msg_txt = "Correct" if prediction == 100 else "Incorrect"
            col = config.green if prediction == 100 else config.red
            if arduino:
                arduino_write(glove_cmd_rest())
            display_multiple_messages_with_udp(
                [msg_txt, "Hand Stationary"],
                [col, config.white],
                [-100, 100],
                config.TIME_STATIONARY,
                None,
                udp_socket_robot,
                config.UDP_ROBOT["IP"],
                config.UDP_ROBOT["PORT"],
                logger,
                eeg_state
            )

        # -----------------------------------------------------------
        # PHASE 3: RELAXATION (End of Trial)
        # -----------------------------------------------------------
        if arduino:
            arduino_write(glove_cmd_rest())

        display_fixation_period(
            duration=float(getattr(config, "INTERTRIAL_DURATION", 3.0)),
            eeg_state=eeg_state,
        )
        current_trial += 1

    # Cleanup / Save adaptive
    if current_trial == len(trial_sequence) and config.SAVE_ADAPTIVE_T:
        if model_type == "M2_LDA_shrink_MDM" and _RC.m2_prev_T is not None:
            save_transform(_RC.m2_prev_T, _RC.m2_rec_counter, adaptive_T_path)
        else:
            save_transform(_RC.Prev_T, _RC.counter, adaptive_T_path)

    log_raw_decision_summary(
        logger,
        "ENDPOINT MDM Decision Summary (target-independent)",
        raw_predictions_list,
        raw_ground_truth_list
    )
    log_raw_decision_summary(
        logger,
        "EARLY RAW Decision Summary (target-independent)",
        raw_early_predictions_list,
        raw_early_ground_truth_list
    )
    log_raw_decision_summary(
        logger,
        "MDM Operational Original Summary",
        mdm_operational_decisions_original,
        ground_truth_list,
    )
    log_raw_decision_summary(
        logger,
        "Final Validated Decision Summary",
        final_validated_decisions,
        ground_truth_list,
    )
    if getattr(config, "SHADOW_MODEL_ANALYSIS_ENABLED", True):
        log_shadow_model_summary(
            logger,
            shadow_earlystop_results,
            shadow_stability_results,
        )
    if getattr(config, "ENDPOINT_VALIDATION_ENABLED", True):
        log_endpoint_validation_summary(
            logger,
            endpoint_validation_stats,
        )
    log_full_window_observer_summary(
        logger,
        full_window_probabilities,
        full_window_targets,
    )

    log_confusion_matrix_from_trial_summary(logger)

    if model_type == "M2_LDA_shrink_MDM":
        try:
            run_label = _run_label_from_logger(logger)
            condition = "FES" if int(getattr(config, "FES_toggle", 0)) else "NO_FES"
            log_riemann_adaptation_csv(
                logger,
                subject=recording_subject,
                session=logger.log_base.name,
                run_label=run_label,
                condition=condition,
                train_refs=riemann_train_refs_run,
                refs_before=riemann_refs_before_run,
                refs_after=getattr(_RC, "m2_prev_T", None),
                updates_before=riemann_updates_before_run,
                updates_after=getattr(_RC, "m2_rec_counter", 0),
                final_predictions=final_validated_decisions,
                mdm_predictions=mdm_operational_decisions_original,
                targets=ground_truth_list,
            )
        except Exception as exc:
            logger.log_event(
                f"⚠️ [RIEMANN_ADAPT_RUN] CSV logging failed: {exc}",
                level="warning",
            )

    if arduino:
        # Final instruction after the online experiment: leave the glove open.
        arduino_write(glove_cmd_mi())
        logger.log_event("🤚 Glove opened at experiment end.")
        arduino.close()

    pygame.quit()


if __name__ == "__main__":
    main()
