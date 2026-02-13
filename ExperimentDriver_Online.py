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
#     if arduino: arduino.write(b'0')

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
#                     arduino.write(b'1')
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
#                 if arduino: arduino.write(b'0')
#                 display_multiple_messages_with_udp(["Incorrect", "Hand Stationary"], [config.red, config.white], [-100, 100], config.TIME_STATIONARY, None, udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], logger, eeg_state)

#         else: # REST Trial
#             msg_txt = "Correct" if prediction == 100 else "Incorrect"
#             col = config.green if prediction == 100 else config.red
#             # Ensure glove is open
#             if arduino: arduino.write(b'0')
#             display_multiple_messages_with_udp([msg_txt, "Hand Stationary"], [col, config.white], [-100, 100], config.TIME_STATIONARY, None, udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], logger, eeg_state)

#         # -----------------------------------------------------------
#         # PHASE 3: RELAXATION (End of Trial)
#         # -----------------------------------------------------------
#         # Open glove for the next trial
#         if arduino: arduino.write(b'0')

#         display_fixation_period(duration=3, eeg_state=eeg_state)
#         current_trial += 1

#     # Cleanup
#     if current_trial == len(trial_sequence) and config.SAVE_ADAPTIVE_T:
#         save_transform(Prev_T, counter, adaptive_T_path)

#     log_confusion_matrix_from_trial_summary(logger)
    
#     if arduino: 
#         arduino.write(b'0')
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
#     if arduino: arduino.write(b'0')

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
#                     arduino.write(b'1')
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
#                 if arduino: arduino.write(b'0')
#                 display_multiple_messages_with_udp(["Incorrect", "Hand Stationary"], [config.red, config.white], [-100, 100], config.TIME_STATIONARY, None, udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], logger, eeg_state)

#         else: # REST Trial
#             msg_txt = "Correct" if prediction == 100 else "Incorrect"
#             col = config.green if prediction == 100 else config.red
#             # Ensure glove is open
#             if arduino: arduino.write(b'0')
#             display_multiple_messages_with_udp([msg_txt, "Hand Stationary"], [col, config.white], [-100, 100], config.TIME_STATIONARY, None, udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], logger, eeg_state)

#         # -----------------------------------------------------------
#         # PHASE 3: RELAXATION (End of Trial)
#         # -----------------------------------------------------------
#         # Open glove for the next trial
#         if arduino: arduino.write(b'0')

#         display_fixation_period(duration=3, eeg_state=eeg_state)
#         current_trial += 1

#     # Cleanup
#     if current_trial == len(trial_sequence) and config.SAVE_ADAPTIVE_T:
#         save_transform(Prev_T, counter, adaptive_T_path)

#     log_confusion_matrix_from_trial_summary(logger)
    
#     if arduino: 
#         arduino.write(b'0')
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
import os
import random
import time
import serial
import sys  # ✅ faltaba (lo usas en sys.exit)
from pylsl import StreamInlet, resolve_stream

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
logger = LoggerManager.auto_detect_from_subject(
    subject=config.TRAINING_SUBJECT,
    base_path=Path(config.DATA_DIR),
    mode="online"
)
# Log config snapshot
loggable_fields = [
    "UDP_MARKER", "UDP_ROBOT", "UDP_FES", "ARM_SIDE", "TOTAL_TRIALS",
    "TIME_MI", "FES_toggle", "TRAINING_SUBJECT"
]
config_log_subset = {k: getattr(config, k) for k in loggable_fields if hasattr(config, k)}
logger.save_config_snapshot(config_log_subset)

eeg_dir = logger.log_base / "eeg"
adaptive_T_path = eeg_dir / "adaptive_T.pkl"

Prev_T, counter = load_transform(adaptive_T_path)
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
    screen = pygame.display.set_mode((1920, 1080),pygame.NOFRAME)
    # Si tú quieres forzar 1920x1080 aquí, lo puedes hacer,
    # pero para que el indicador se vea proporcional, lo dejamos dinámico:
    screen_width = monitor_w
    screen_height = monitor_h
else:
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
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


# ============================================================
# ARDUINO SETUP
# ============================================================
ARDUINO_PORT = os.environ.get("ARDUINO_PORT", "")
ARDUINO_BAUD = int(os.environ.get("ARDUINO_BAUD", 9600))
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

# Load Model
subject_model_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
subject_model_path = os.path.join(subject_model_dir, f"sub-{config.TRAINING_SUBJECT}_model.pkl")

try:
    with open(subject_model_path, 'rb') as f:
        model = pickle.load(f)
    logger.log_event(f"✅ Model loaded: {subject_model_path}")
except FileNotFoundError:
    logger.log_event(f"❌ Model not found: {subject_model_path}", level="error")
    sys.exit(1)

predictions_list = []
ground_truth_list = []
# ============================================================
# WIRE RUNTIME OBJECTS
# ============================================================
_RC.config = config
_RC.logger = logger
_RC.model = model
_RC.screen = screen
_RC.screen_width = screen_width
_RC.screen_height = screen_height
_RC.udp_socket_marker = udp_socket_marker
_RC.udp_socket_robot  = udp_socket_robot
_RC.udp_socket_fes    = udp_socket_fes
_RC.FES_toggle = FES_toggle
_RC.Prev_T = Prev_T
_RC.counter = counter

# NOTE: We do not pass '_RC.arduino' because runtime_common
# will not handle the glove. The glove is handled by this main script.


# ============================================================
# ✅ PRE-TRIAL INDICATOR (MATCH OFFLINE LOOK)
# ============================================================
NEXT_INDICATOR_POS = (0.50, 0.28)
NEXT_INDICATOR_SCALE = 1.00

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

def draw_pretrial_screen_online(mode, time_ball_state=1):
    """
    Replica el look de OFFLINE en preparación:
      - MI: cuadro rojo + flecha derecha
      - REST: círculo azul + flecha izquierda
      - time_balls en mode='single' en el indicador NEXT
    """
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)

    pos_x = int(screen_width * NEXT_INDICATOR_POS[0])
    pos_y = int(screen_height * NEXT_INDICATOR_POS[1])
    base_size = int(min(screen_width, screen_height) * 0.08 * NEXT_INDICATOR_SCALE)

    is_mi = (mode == 0)
    next_color = (255, 50, 50) if is_mi else (0, 120, 255)

    # 1) Shape background
    if is_mi:
        bg_rect = pygame.Rect(pos_x - base_size // 2, pos_y - base_size // 2, base_size, base_size)
        pygame.draw.rect(screen, next_color, bg_rect)
    else:
        pygame.draw.circle(screen, next_color, (pos_x, pos_y), base_size // 2)

    # 2) Single time-ball indicator (igual al offline)
    draw_time_balls(
        time_ball_state,
        screen_width,
        screen_height,
        mode="single",
        indicator_color=next_color,
        single_pos=NEXT_INDICATOR_POS,
        ball_radius=int(base_size * 0.4),
    )

    # 3) Texto de preparación
    font_prep = pygame.font.SysFont(None, 96)
    if is_mi:
        prep_msg = f"Prepare to close {config.ARM_SIDE.upper()} hand"
    else:
        prep_msg = "Rest"

    txt_surface = font_prep.render(prep_msg, True, config.white)
    screen.blit(
        txt_surface,
        (screen_width // 2 - txt_surface.get_width() // 2, screen_height // 2 + 300),
    )

    # 4) Flecha direccional
    arrow_dir = "right" if is_mi else "left"
    draw_arrow_directional(screen, pos_x, pos_y, base_size // 2.5, (255, 255, 255), direction=arrow_dir)

    pygame.display.flip()


def main():
    logger.log_event("Resolving EEG data stream via LSL...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    eeg_state = EEGStreamState(inlet=inlet, config=config, logger=logger)

    trial_sequence = generate_trial_sequence(total_trials=config.TOTAL_TRIALS, max_repeats=config.MAX_REPEATS)
    current_trial = 0
    running = True
    clock = pygame.time.Clock()

    display_fixation_period(duration=3, eeg_state=eeg_state)

    # Ensure glove is open at start
    if arduino:
        arduino.write(b'0')

    while running and current_trial < len(trial_sequence):
        logger.log_event(f"--- Trial {current_trial+1}/{len(trial_sequence)} START ---")

        # 1) Decide modo del trial
        mode = trial_sequence[current_trial]

        # 2) ✅ Pantalla de preparación con el mismo look que OFFLINE
        draw_pretrial_screen_online(mode=mode, time_ball_state=1)

        # 3) Waiting / Countdown
        waiting_for_press = True
        countdown_start = None
        countdown_duration = 1500  # ms

        while waiting_for_press:
            eeg_state.update()

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
                elapsed = pygame.time.get_ticks() - countdown_start

                # ✅ Re-dibujar la pantalla de preparación para mantener el indicador visible
                #    (puedes animar time_ball_state si quieres; aquí lo mantenemos en 1)
                draw_pretrial_screen_online(mode=mode, time_ball_state=1)

                if elapsed >= countdown_duration:
                    waiting_for_press = False
            else:
                # Si no hay timing, igual mantenemos la pantalla
                draw_pretrial_screen_online(mode=mode, time_ball_state=1)

            clock.tick(60)

        if not running:
            break

        mode = trial_sequence[current_trial]

        # 4) Baseline
        try:
            eeg_state.compute_baseline(duration_sec=config.BASELINE_DURATION)
        except ValueError:
            continue

        # -----------------------------------------------------------
        # PHASE 1: EFFORT (Sensory FES Only)
        # -----------------------------------------------------------
        prediction, confidence, leaky_integrator, trial_probs, earlystop_flag = show_feedback(
            duration=config.TIME_MI,
            mode=mode,
            eeg_state=eeg_state
        )

        append_trial_probabilities_to_csv(
            trial_probabilities=trial_probs, mode=mode, trial_number=current_trial + 1,
            predicted_label=prediction, early_cutout=earlystop_flag,
            mi_threshold=config.THRESHOLD_MI, rest_threshold=config.THRESHOLD_REST,
            logger=logger, phase="MI" if mode == 0 else "REST"
        )

        # -----------------------------------------------------------
        # PHASE 2: REWARD (Motor FES + Glove + Robot)
        # -----------------------------------------------------------

        predictions_list.append(prediction)
        ground_truth_list.append(200 if mode == 0 else 100)

        if mode == 0:  # MI Trial
            if prediction == 200:  # SUCCESS (Threshold reached)

                # 1) CLOSE GLOVE (Reward Trigger)
                if arduino:
                    arduino.write(b'1')
                    logger.log_event("✅ Prediction Success -> Closing Glove (Reward)")

                # 2) MOTOR FES
                if FES_toggle:
                    send_udp_message(
                        udp_socket_fes,
                        config.UDP_FES["IP"],
                        config.UDP_FES["PORT"],
                        "FES_MOTOR_GO",
                        logger=logger
                    )

                # 3) ROBOT
                messages = ["Correct", "Hand close"]
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
                    eeg_state, leaky_integrator
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
                    arduino.write(b'0')
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
                arduino.write(b'0')
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
            arduino.write(b'0')

        display_fixation_period(duration=3, eeg_state=eeg_state)
        current_trial += 1

    # Cleanup / Save adaptive
    if current_trial == len(trial_sequence) and config.SAVE_ADAPTIVE_T:
        save_transform(Prev_T, counter, adaptive_T_path)

    log_confusion_matrix_from_trial_summary(logger)

    if arduino:
        arduino.write(b'0')
        arduino.close()

    pygame.quit()


if __name__ == "__main__":
    main()
