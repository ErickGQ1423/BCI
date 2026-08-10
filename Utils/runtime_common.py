
# # runtime_common.py
# # Centralizes functions used by both ExperimentDriver_Online and ExperimentDriver_Bimanual
# # with minimal edits to math/logic. Runtime objects are "wired" in by the drivers at startup.

# import time
# import numpy as np
# import pygame
# from sklearn.covariance import LedoitWolf
# from pyriemann.estimation import Shrinkage
# from pyriemann.utils.geodesic import geodesic_riemann
# from pyriemann.utils.base import invsqrtm
# import pandas as pd
# from sklearn.metrics import confusion_matrix
# # Visualization utils (UI draws)
# from Utils.visualization import (
#     draw_arrow_fill,
#     draw_ball_fill,
#     draw_fixation_cross,
#     draw_time_balls,
# )
# # Experiment utilities
# from Utils.experiment_utils import (
#     generate_trial_sequence,
#     LeakyIntegrator,
#     RollingScaler,
#     save_transform,
#     load_transform
# )

# # UDP helper
# from Utils.networking import send_udp_message,display_multiple_messages_with_udp

# # --------- Runtime "globals" (wired by each driver right after init) ---------
# # These names intentionally match what the original functions referenced.
# config = None
# logger = None
# model = None

# # Surfaces/screen geometry used by draw helpers
# screen = None
# screen_width = None
# screen_height = None

# # UDP sockets
# udp_socket_marker = None
# udp_socket_robot  = None
# udp_socket_fes    = None

# # Flags
# FES_toggle = None

# # Adaptive recentering state
# Prev_T = None
# counter = 0


# # ----------------- Common helpers -----------------

# def log_confusion_matrix_from_trial_summary(logger):
#     df = pd.read_csv(logger.trial_summary_path)

#     # Separate into valid and ambiguous trials
#     ambiguous_trials = df[df["Predicted Label"].isna()]
#     valid_trials = df.dropna(subset=["Predicted Label"])

#     valid_trials.loc[:, "Predicted Label"] = valid_trials["Predicted Label"].astype(int)
#     valid_trials.loc[:, "True Label"] = valid_trials["True Label"].astype(int)

#     # Count correct predictions
#     correct = (valid_trials["Predicted Label"] == valid_trials["True Label"]).sum()
#     incorrect = len(valid_trials) - correct
#     ambiguous = len(ambiguous_trials)
#     total = correct + incorrect + ambiguous

#     # Generate confusion matrix
#     if not valid_trials.empty:
#         cm = confusion_matrix(
#             valid_trials["True Label"], valid_trials["Predicted Label"],
#             labels=[200, 100]
#         )
#         logger.log_event("Confusion Matrix (Correct/Incorrect Only):")
#         logger.log_event(f"  Actual 200 (MI)    | Predicted 200 (MI): {cm[0][0]} | Predicted 100 (REST): {cm[0][1]}")
#         logger.log_event(f"  Actual 100 (REST)  | Predicted 200 (MI): {cm[1][0]} | Predicted 100 (REST): {cm[1][1]}")
#     else:
#         logger.log_event("No non-ambiguous trials to compute confusion matrix.")

#     # Log summary stats
#     if total:
#         percent_correct_incl_ambiguous = (correct / total) * 100
#         percent_correct_excl_ambiguous = (correct / (correct + incorrect)) * 100 if (correct + incorrect) > 0 else 0
#         logger.log_event(f"✅ % Total Accuracy (Including ambiguous): {percent_correct_incl_ambiguous:.2f}%")
#         logger.log_event(f"✅ % Decision Accuracy (Excluding ambiguous): {percent_correct_excl_ambiguous:.2f}%")
#         logger.log_event(f"⚠️ Ambiguous trials (not counted in exclusive metric): {ambiguous}")
#     else:
#         logger.log_event("No trials available to compute statistics.")



# def append_trial_probabilities_to_csv(trial_probabilities, mode, trial_number,
#                                       predicted_label, early_cutout,
#                                       mi_threshold, rest_threshold, logger,
#                                       phase):
#     correct_class = 200 if mode == 0 else 100
#     trial_probabilities = np.array(trial_probabilities)

#     if trial_probabilities.shape[1] != 3:
#         logger.log_event(f"❌ Error: Unexpected shape {trial_probabilities.shape}. Expected (N,3). Skipping save.")
#         return

#     for row in trial_probabilities:
#         timestamp, prob_rest, prob_mi = row
#         logger.log_decoder_output(
#             trial=trial_number,
#             timestamp=timestamp,
#             prob_mi=prob_mi,
#             prob_rest=prob_rest,
#             true_label=correct_class,
#             predicted_label=predicted_label,
#             early_cutout=early_cutout,
#             mi_threshold=mi_threshold,
#             rest_threshold=rest_threshold,
#             phase=phase
#         )

#     logger.log_event(
#         f"✅ Logged {len(trial_probabilities)} rows for Trial {trial_number} | "
#         f"True: {correct_class}, Predicted: {predicted_label}, Early Cut: {early_cutout}, Phase: {phase}"
#     )


# def display_fixation_period(duration=3, eeg_state=None):
#     """
#     Displays a blank screen with a fixation cross for a given duration.
    
#     Parameters:
#     - duration (int): Time in seconds for which the fixation period lasts.
#     - eeg_state: Optional EEGState object to be updated during the fixation period.
#     """
#     start_time = time.time()
#     clock = pygame.time.Clock()

#     while time.time() - start_time < duration:
#         # Fill screen with background color
#         pygame.display.get_surface().fill(config.black)

#         # Draw UI elements
#         draw_fixation_cross(screen_width, screen_height)
#         draw_ball_fill(0, screen_width, screen_height)
#         draw_arrow_fill(0, screen_width, screen_height)
#         draw_time_balls(0, screen_width, screen_height)

#         pygame.display.flip()

#         # Update EEG buffer if provided
#         if eeg_state is not None:
#             eeg_state.update()

#         # Handle quit events
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 return

#         clock.tick(60)


# # Interpolation function to compute fill amount between SHAPE_MIN and SHAPE_MAX
# def interpolate_fill(value):
#     return max(0, min(1, (value - config.SHAPE_MIN) / (config.SHAPE_MAX - config.SHAPE_MIN)))

# def calculate_fill_levels(running_avg_confidence, mode):
#     """
#     Determines the fill levels for both MI (arrow) and Rest (ball) based on accumulated probability.

#     Parameters:
#         running_avg_confidence (float): The leaky-integrated probability estimate.
#         mode (int): 0 for MI trial (fill square), 1 for Rest trial (fill ball).

#     Returns:
#         tuple: (fill_arrow, fill_ball) - Values between 0 and 1 indicating fill levels.
#     """
#     # Ensure probability stays within configured bounds
#     prob = max(0, min(1, running_avg_confidence))
#     prob_inverse = 1 - prob  # Inverse probability for the other shape


#     # Determine fill levels
#     fill_mi = interpolate_fill(prob) if prob >= config.SHAPE_MIN else 0  # MI shape fills when prob > SHAPE_MIN
#     fill_rest = interpolate_fill(prob_inverse) if prob_inverse >= config.SHAPE_MIN else 0  # Rest shape fills when 1-prob > SHAPE_MIN

#     # Swap roles if in Rest mode
#     if mode == 1:
#         return fill_rest, fill_mi  # Flip values for Rest condition
#     return fill_mi, fill_rest  # Default for MI mode


# def handle_fes_activation(mode, running_avg_confidence, fes_active):
#     """
#     Manages the activation of sensory FES based on the running average probability.

#     Parameters:
#         mode (int): 0 for MI (Motor Imagery), 1 for Rest.
#         running_avg_confidence (float): Current probability estimate.
#         fes_active (bool): Current state of FES (True if active, False if inactive).
#         logger: LoggerManager instance used for structured logging.

#     Returns:
#         bool: Updated FES state after processing.
#     """
#     # Determine if FES should be active:
#     # - If mode is MI (0) and confidence > 0.5 → Turn on FES
#     # - If mode is Rest (1) and confidence < 0.5 → Turn on FES
#     fes_should_be_active = (mode == 0 and running_avg_confidence > 0.5) or \
#                            (mode == 1 and running_avg_confidence < 0.5)

#     # Activate FES if needed
#     if fes_should_be_active and not fes_active:
#         if FES_toggle == 1:
#             send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_SENS_GO", logger=logger)
#             logger.log_event("Sensory FES activated.")
#         else:
#             logger.log_event("FES toggle is off — activation skipped.")
#         return True

#     # Deactivate FES if needed
#     elif not fes_should_be_active and fes_active:
#         if FES_toggle == 1:
#             send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
#             logger.log_event("Sensory FES stopped.")
#         else:
#             logger.log_event("FES toggle is off — stop command skipped.")
#         return False

#     # No change in state
#     return fes_active

# def classify_real_time(eeg_state, window_size_samples, all_probabilities, predictions, mode, leaky_integrator, update_recentering=True):
#     global counter
#     global Prev_T

#     pygame.display.flip()
#     pygame.event.get()  # Heartbeat to OS

#     try:
#         window, _ = eeg_state.get_baseline_corrected_window(window_size_samples)
#     except ValueError:
#         return leaky_integrator.accumulated_probability, predictions, all_probabilities

#     # === Covariance Matrix ===
#     cov_matrix = (window @ window.T) / np.trace(window @ window.T)

#     if config.LEDOITWOLF:
#         cov_matrix = np.array([LedoitWolf().fit(cov_matrix).covariance_])
#     else:
#         cov_matrix = np.expand_dims(cov_matrix, axis=0)
#         shrinkage = Shrinkage(shrinkage=config.SHRINKAGE_PARAM)
#         cov_matrix = shrinkage.fit_transform(cov_matrix)

#     # === Adaptive Recentering ===
#     if config.RECENTERING:
#         cov_matrix = np.squeeze(cov_matrix, axis=0)

#         if counter == 0 or Prev_T is None:
#             Prev_T = cov_matrix

#         T_test = geodesic_riemann(Prev_T, cov_matrix, 1 / (counter + 1))
#         T_invsqrtm = invsqrtm(Prev_T)
#         cov_matrix = T_invsqrtm @ cov_matrix @ T_invsqrtm.T
#         cov_matrix = np.expand_dims(cov_matrix, axis=0)

#     # === Classification ===
#     probabilities = model.predict_proba(cov_matrix)[0]
#     predicted_label = model.classes_[np.argmax(probabilities)]

#     correct_label = 200 if mode == 0 else 100
#     correct_class_idx = np.where(model.classes_ == correct_label)[0][0]
#     current_confidence = probabilities[correct_class_idx]

#     # === Determine if recentering update should occur ===
#     should_update_T = False
#     if config.RECENTERING and update_recentering:
#         if config.USE_CONFIDENCE_GATE:
#             correct_label = 200 if mode == 0 else 100
#             correct_class_idx = np.where(model.classes_ == correct_label)[0][0]
#             current_confidence = probabilities[correct_class_idx]
#             predicted_correct = (predicted_label == correct_label)
#             confident_enough = (current_confidence >= config.RECENTERING_CONFIDENCE_THRESHOLD)
#             should_update_T = predicted_correct and confident_enough
#         else:
#             # Always update if gating is disabled
#             should_update_T = True

#     if should_update_T:
#         Prev_T = T_test
#         counter += 1


#     # === Update Logs ===
#     predictions.append(predicted_label)
#     all_probabilities.append([time.time(), probabilities[0], probabilities[1]])

#     return current_confidence, predictions, all_probabilities




# def hold_messages_and_classify(messages, colors, offsets, duration, mode, udp_socket, udp_ip, udp_port,
#                                eeg_state, leaky_integrator):
#     """
#     Holds visual messages on the screen while running real-time EEG classification in the background.
#     Classifies every STEP_SIZE seconds using the most recent WINDOW_SIZE seconds of EEG data.

#     Returns:
#     - int: Final classification result (200 or 100)
#     - list: All classification probabilities
#     - bool: Whether an early stop occurred
#     """
#     font = pygame.font.SysFont(None, 72)
#     start_time = time.time()
#     early_stop = False

#     step_size = config.STEP_SIZE  # e.g. 1/16s
#     window_size = config.CLASSIFY_WINDOW / 1000  # ms → seconds
#     window_size_samples = int(window_size * config.FS)

#     correct_class = 200 if mode == 0 else 100
#     incorrect_class = 100 if mode == 0 else 200

#     min_predictions_before_stop = config.MIN_PREDICTIONS
#     num_predictions = 0
#     accuracy_threshold = config.THRESHOLD_MI if mode == 0 else config.THRESHOLD_REST 

#     all_probabilities = []
#     predictions = []
#     running_avg_confidence = 0.5
#     current_confidence = 0.5

#     next_tick = time.time()  # Classify immediately
#     pygame.display.update()
#     clock = pygame.time.Clock()

#     while time.time() - start_time < duration:
#         now = time.time()

#         # === Update EEG Buffer ===
#         eeg_state.update()

#         # === Draw Messages ===
#         pygame.display.get_surface().fill((0, 0, 0))
#         for i, text in enumerate(messages):
#             message = font.render(text, True, colors[i])
#             pygame.display.get_surface().blit(
#                 message,
#                 (pygame.display.get_surface().get_width() // 2 - message.get_width() // 2,
#                  pygame.display.get_surface().get_height() // 2 + offsets[i])
#             )
#         pygame.display.flip()

#         # === Classify every step_size seconds ===
#         if now >= next_tick:
#             current_confidence, predictions, all_probabilities = classify_real_time(
#                 eeg_state, window_size_samples,
#                 all_probabilities, predictions,
#                 mode, leaky_integrator,
#                 update_recentering=config.UPDATE_DURING_MOVE
#             )
#             next_tick += step_size 
#             if all_probabilities and getattr(config, "SEND_PROBS", False):
#                 prob_mi, prob_rest = all_probabilities[-1][2], all_probabilities[-1][1]
#                 send_udp_message(
#                     udp_socket_marker,
#                     config.UDP_MARKER["IP"],
#                     config.UDP_MARKER["PORT"],
#                     f"{config.TRIGGERS['ROBOT_PROBS']},{prob_mi:.5f},{prob_rest:.5f}",
#                     quiet=True
#                 )

#             if current_confidence > 0:
#                 num_predictions += 1

#             running_avg_confidence = leaky_integrator.update(current_confidence)

#             if num_predictions >= min_predictions_before_stop and running_avg_confidence < config.RELAXATION_RATIO * accuracy_threshold:
#                 early_stop = True

#                 logger.log_event(f"Early stop triggered! Confidence: {running_avg_confidence:.2f} after {num_predictions} predictions")

#                 send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_EARLYSTOP"], logger=logger)
#                 send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_END"], logger=logger)

#                 if FES_toggle == 1:
#                     send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
#                     logger.log_event("FES_STOP signal sent due to early stop.")
#                 else:
#                     logger.log_event("FES is disabled — no FES_STOP sent.")

#                 display_multiple_messages_with_udp(
#                     ["Stopping Robot"], [(255, 0, 0)], [0], duration=3,
#                     udp_messages=[config.ROBOT_OPCODES["STOP"]], udp_socket=udp_socket, udp_ip=udp_ip, udp_port=udp_port, logger=logger
#                 )
#                 break

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 return None

#         clock.tick(60)

#     if not early_stop:
#         send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_END"], logger=logger)

#     final_class = correct_class if running_avg_confidence >= config.RELAXATION_RATIO * accuracy_threshold else incorrect_class
#     logger.log_event(f"Confidence at the end of motion: {running_avg_confidence:.2f} after {num_predictions} predictions")

#     return final_class, all_probabilities, early_stop




# def show_feedback(duration=5, mode=0, eeg_state = None):
#     """
#     Displays feedback animation, collects EEG data, and performs real-time classification
#     using a sliding window approach with early stopping based on posterior probabilities.
#     """
#     start_time = time.time()
#     step_size = config.STEP_SIZE  # Sliding window step size (seconds)
#     window_size = config.CLASSIFY_WINDOW / 1000  # Convert ms to seconds
#     window_size_samples = int(window_size * config.FS)
#     step_size_samples = int(step_size * config.FS)
#     FES_active = False
#     all_probabilities = []
#     predictions = []
#     leaky_integrator = LeakyIntegrator(alpha=config.INTEGRATOR_ALPHA)  # Confidence smoothing
#     min_predictions = config.MIN_PREDICTIONS
#     earlystop_flag = False

#     classification_results = []
#     # Define the correct class based on mode
#     # Define the correct class based on mode
#     correct_class = 200 if mode == 0 else 100  # 200 = Right Arm MI, 100 = Rest
#     incorrect_class = 100 if mode == 0 else 200  # The opposite class

#     # accuracy threshold based on mode
#     accuracy_threshold = config.THRESHOLD_MI if mode == 0 else config.THRESHOLD_REST 
#     opposed_threshold = config.THRESHOLD_REST if mode == 0 else config.THRESHOLD_MI
#     # Preprocess the baseline dataset before feedback starts
#     # Preprocess the baseline dataset before feedback starts
#     pygame.display.flip()

#     # Send UDP triggers
#     if mode == 0:  # Red Arrow Mode (Motor Imagery)
#         send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_BEGIN"], logger=logger)
#         if FES_toggle == 1:
#             send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_SENS_GO", logger=logger)
#             FES_active = True
#         else:
#             logger.log_event("FES is disabled.")
#             FES_active = False
#     else:  # Blue Ball Mode (Rest)
#         send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_BEGIN"], logger=logger)
#         FES_active = False

#     clock = pygame.time.Clock()
#     running_avg_confidence = 0.5  # Initial placeholder
#     current_confidence = 0.5 # Initial placeholder for initial window updates
#     next_tick = start_time + window_size  # Skip first second

#     while time.time() - start_time < duration:
#         eeg_state.update()

#         now = time.time()
#         if now >= next_tick:
#             current_confidence, predictions, all_probabilities = classify_real_time(
#                 eeg_state,
#                 window_size_samples,
#                 all_probabilities,
#                 predictions,
#                 mode,
#                 leaky_integrator
#             )
#             next_tick += step_size 
#             if all_probabilities and getattr(config, "SEND_PROBS", False):
#                 prob_mi, prob_rest = all_probabilities[-1][2], all_probabilities[-1][1]
#                 send_udp_message(
#                     udp_socket_marker,
#                     config.UDP_MARKER["IP"],
#                     config.UDP_MARKER["PORT"],
#                     f"{config.TRIGGERS['MI_PROBS' if mode == 0 else 'REST_PROBS']},{prob_mi:.5f},{prob_rest:.5f}",
#                     quiet=True
#                 )


#         running_avg_confidence = leaky_integrator.update(current_confidence)
#         if FES_toggle == 1:
#             FES_active = handle_fes_activation(mode, running_avg_confidence, FES_active)

#         screen.fill(config.black)
#         MI_fill, Rest_fill = calculate_fill_levels(running_avg_confidence, mode)

#         if mode == 0:
#             draw_arrow_fill(MI_fill, screen_width, screen_height)
#             draw_fixation_cross(screen_width, screen_height)
#             draw_ball_fill(Rest_fill, screen_width, screen_height)
#             draw_time_balls(2, screen_width, screen_height)
#             message = pygame.font.SysFont(None, 96).render(f"Imagine closing {config.ARM_SIDE.upper()} hand", True, config.white)
#         else:
#             draw_ball_fill(Rest_fill, screen_width, screen_height)
#             draw_fixation_cross(screen_width, screen_height)
#             draw_arrow_fill(MI_fill, screen_width, screen_height)
#             draw_time_balls(3, screen_width, screen_height)
#             message = pygame.font.SysFont(None, 96).render("Rest", True, config.white)

#         screen.blit(message, (screen_width // 2 - message.get_width() // 2, screen_height // 2 + 300))
#         pygame.display.flip()
#         clock.tick(60)
#         # --- Early-stop logic (supports correct-only or either-threshold) ---
#         hit_correct   = (len(predictions) >= min_predictions) and (running_avg_confidence >= accuracy_threshold)
#         hit_incorrect = (len(predictions) >= min_predictions) and (running_avg_confidence <= (1 - opposed_threshold))

#         should_earlystop = hit_correct or (config.EARLYSTOP_MODE == "either" and hit_incorrect)
#         if should_earlystop:
#             earlystop_flag = True

#             # Figure out which class triggered the stop (for logging/triggers)
#             if hit_correct:
#                 stop_reason = "correct"
#                 trigger_key = "MI_EARLYSTOP" if mode == 0 else "REST_EARLYSTOP"
#             else:
#                 stop_reason = "incorrect"
#                 trigger_key = "REST_EARLYSTOP" if mode == 0 else "MI_EARLYSTOP"

#             logger.log_event(
#                 f"Early stopping triggered ({stop_reason}). "
#                 f"Confidence={running_avg_confidence:.2f}, "
#                 f"min_preds={min_predictions}, "
#                 f"mode={'MI' if mode==0 else 'REST'}"
#             )

#             # Stop FES if active
#             if FES_toggle == 1:
#                 send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
#             else:
#                 logger.log_event("FES is disabled.")

#             # Emit the appropriate EARLYSTOP trigger
#             send_udp_message(
#                 udp_socket_marker,
#                 config.UDP_MARKER["IP"],
#                 config.UDP_MARKER["PORT"],
#                 config.TRIGGERS[trigger_key],
#                 logger=logger
#             )
#             break

    
#     pygame.display.flip()
#     # Final Decision
#     if running_avg_confidence >= accuracy_threshold:
#         final_class = correct_class
#     elif running_avg_confidence <= (1 - opposed_threshold):
#         final_class = incorrect_class
#     else:
#         final_class = None  # Ambiguous zone
    
#     if final_class is not None:
#         logger.log_event(
#             f"Final decision: {final_class}, Confidence for correct({correct_class}) class: "
#             f"{running_avg_confidence:.2f}, at sample size {len(predictions)}"
#         )
#     else:
#         logger.log_event(
#             f"Ambiguous final decision — no threshold met. Confidence: {running_avg_confidence:.2f}, "
#             f"MI threshold: {config.THRESHOLD_MI}, REST threshold: {config.THRESHOLD_REST}, "
#             f"Samples: {len(predictions)}"
#         )
#     if FES_toggle == 1 and FES_active:
#         send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
#     else:
#         logger.log_event("FES disable not needed.")


#     send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_END" if mode==0 else "REST_END"], logger=logger)
#     pygame.time.delay(300)  # ~300 ms delay to allow the visual feedback to complete rendering
#     return final_class, running_avg_confidence, leaky_integrator, all_probabilities, earlystop_flag


# runtime_common.py
# Centralizes functions used by both ExperimentDriver_Online and ExperimentDriver_Bimanual
# with minimal edits to math/logic. Runtime objects are "wired" in by the drivers at startup.

import time
import numpy as np
import pygame
from sklearn.covariance import LedoitWolf
from pyriemann.estimation import Shrinkage
from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.utils.base import invsqrtm
import pandas as pd
from sklearn.metrics import confusion_matrix

# Visualization utils (UI draws)
from Utils.visualization import (
    draw_arrow_fill,
    draw_ball_fill,
    draw_fixation_cross,
    draw_time_balls,
)

# Experiment utilities
from Utils.experiment_utils import (
    generate_trial_sequence,
    LeakyIntegrator,
    RollingScaler,
    save_transform,
    load_transform
)

# UDP helper
from Utils.networking import send_udp_message, display_multiple_messages_with_udp

# --------- Runtime "globals" (wired by each driver right after init) ---------
# These names intentionally match what the original functions referenced.
config = None
logger = None
model = None
template = None

# M2 cross-subject model state
model_pkg       = None   # full model package (M2_LDA_shrink_MDM format)
observer_model_pkg = None  # optional warmup package, logs only; never controls feedback
prep_epoch      = None   # latest full prep epoch/window used by M2 online logic
m2_ch_idx       = None   # channel indices for model picks in live stream
_m2_last_step   = -1     # last M2 step logged (avoid duplicate logs per step)
_m2_lda_probs   = []     # LDA P(MI) per step — accumulated for end-of-trial summary
_m2_lda_compact_probs = []  # LDA compacto de 3 canales P(MI) per step
_m2_mdm_probs   = []     # MDM confidence per step — accumulated for end-of-trial summary
_m2_lr_probs    = []     # LR observer P(MI) per step
_m2_svm_probs   = []     # SVM observer P(MI) per step
_m2_shadow_records = []  # exact per-step P(MI) values; diagnostics only
_m2_warmup_mdm_probs = []  # warmup observer MDM P(MI) per step
_m2_warmup_lda_probs = []  # warmup observer LDA P(MI) per step
_m2_quality_reject_count = 0
_m2_quality_bad_trial = False
_m2_quality_last_logged_step = -1

# M2 Riemannian recentering state (one Prev_T per time step)
m2_prev_T       = None   # list[ndarray] of size n_timepoints; None until first trial
m2_rec_counter  = 0      # accepted recentering updates
m2_rec_seen_trials = 0   # usable trials observed while recentering is enabled

# Surfaces/screen geometry used by draw helpers
screen = None
screen_width = None
screen_height = None

# UDP sockets
udp_socket_marker = None
udp_socket_robot  = None
udp_socket_fes    = None

# Flags
FES_toggle = None

# Adaptive recentering state
Prev_T = None
counter = 0


def _prep_decoder_mode():
    cfg = config
    if cfg is None:
        try:
            import config as cfg
        except Exception:
            cfg = None
    mode = getattr(cfg, "PREP_DECODER_MODE", None)
    if mode:
        return str(mode).upper()
    # Backward compatibility for temporary test configs.
    return "M2_CUMULATIVE" if getattr(cfg, "M2_LIVE_WINDOW", False) else "FROZEN_EPOCH_DEBUG"


def reset_m2_quality_state():
    global _m2_quality_reject_count, _m2_quality_bad_trial, _m2_quality_last_logged_step
    _m2_quality_reject_count = 0
    _m2_quality_bad_trial = False
    _m2_quality_last_logged_step = -1


def _is_spd_finite(matrix, min_eig=1e-12):
    """Small guard for Riemannian operations in online mode."""
    try:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return False
        if not np.isfinite(matrix).all():
            return False
        eigvals = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        return bool(np.all(eigvals > min_eig))
    except Exception:
        return False


def _check_eeg_quality(window_ch):
    """Return (ok, reason, metrics) for the selected model channels."""
    cfg = config
    if cfg is None:
        try:
            import config as cfg
        except Exception:
            cfg = None

    if not getattr(cfg, "EEG_QUALITY_GATE", True):
        return True, "disabled", {}

    if window_ch is None:
        return False, "empty_window", {}

    x = np.asarray(window_ch, dtype=float)
    if x.ndim != 2 or x.size == 0:
        return False, "bad_shape", {"shape": getattr(x, "shape", None)}

    if not np.isfinite(x).all():
        return False, "nan_or_inf", {}

    ptp = np.ptp(x, axis=1)
    rms = np.sqrt(np.mean(x * x, axis=1))
    max_abs = np.max(np.abs(x), axis=1)

    min_ptp = float(getattr(cfg, "EEG_QUALITY_MIN_PTP_UV", 0.2))
    max_ptp = float(getattr(cfg, "EEG_QUALITY_MAX_PTP_UV", 250.0))
    max_rms = float(getattr(cfg, "EEG_QUALITY_MAX_RMS_UV", 75.0))
    max_abs_lim = float(getattr(cfg, "EEG_QUALITY_MAX_ABS_UV", 250.0))
    bad_channels_to_reject = int(getattr(cfg, "EEG_QUALITY_BAD_CHANNELS_TO_REJECT", 1))
    hard_max_abs = float(getattr(cfg, "EEG_QUALITY_HARD_MAX_ABS_UV", max_abs_lim * 2))

    flat_count = int(np.sum(ptp < min_ptp))
    high_ptp_count = int(np.sum(ptp > max_ptp))
    high_rms_count = int(np.sum(rms > max_rms))
    high_abs_count = int(np.sum(max_abs > max_abs_lim))

    metrics = {
        "ptp_max": float(np.max(ptp)),
        "ptp_min": float(np.min(ptp)),
        "rms_max": float(np.max(rms)),
        "abs_max": float(np.max(max_abs)),
        "flat_ch": flat_count,
        "high_ptp_ch": high_ptp_count,
        "high_rms_ch": high_rms_count,
        "high_abs_ch": high_abs_count,
    }

    if metrics["abs_max"] > hard_max_abs:
        return False, "hard_abs_too_high", metrics
    if flat_count >= bad_channels_to_reject:
        return False, "flat_signal", metrics
    if high_ptp_count >= bad_channels_to_reject:
        return False, "ptp_too_high", metrics
    if high_rms_count >= bad_channels_to_reject:
        return False, "rms_too_high", metrics
    if high_abs_count >= bad_channels_to_reject:
        return False, "abs_too_high", metrics

    return True, "ok", metrics


def _predict_m2_pkg_pmi(pkg, step, epoch_ch, raw_step, all_t_idx, decoder_mode,
                        recenter=False):
    """Return (MDM P(MI), LDA P(MI)) for an M2 package without side effects."""
    p_mdm_mi = 0.5
    p_lda_mi = 0.5
    if pkg is None or pkg.get("model_type") != "M2_LDA_shrink_MDM":
        return p_mdm_mi, p_lda_mi

    try:
        mi_id = pkg["MI_ID"]
        mdm_model = pkg["mdm_models"][step]
        mdm_template = pkg["mdm_templates"][step]
        t_end = all_t_idx[step] + 1
        raw_for_mdm = (
            raw_step
            if decoder_mode == "M2_CUMULATIVE"
            else epoch_ch[:, :t_end]
        )
        tmpl_cut = mdm_template
        raw_for_mdm = _match_mdm_template_samples(
            raw_for_mdm,
            tmpl_cut,
        )
        extended = np.concatenate([raw_for_mdm, tmpl_cut], axis=0)
        raw_cov = extended @ extended.T
        tr = np.trace(raw_cov)
        if tr > 1e-12 and np.isfinite(tr):
            cov_norm = raw_cov / tr
            pkg_requires_recenter = (
                pkg.get("mdm_recenter_mode") == "train_riemann_mean"
            )
            if ((recenter or pkg_requires_recenter) and m2_prev_T is not None
                    and step < len(m2_prev_T)
                    and m2_prev_T[step] is not None):
                try:
                    T_inv = invsqrtm(m2_prev_T[step])
                    cov_recentered = T_inv @ cov_norm @ T_inv.T
                    cov_recentered = 0.5 * (cov_recentered + cov_recentered.T)
                    if _is_spd_finite(cov_recentered):
                        cov_norm = cov_recentered
                except Exception:
                    pass
            cov = cov_norm + pkg["cov_reg"] * np.eye(cov_norm.shape[0])
            cov = 0.5 * (cov + cov.T)
            proba = mdm_model.predict_proba(np.expand_dims(cov, 0))[0]
            mi_col = list(mdm_model.classes_).index(mi_id)
            p_mdm_mi = float(proba[mi_col])
    except Exception:
        p_mdm_mi = 0.5

    try:
        mi_id = pkg["MI_ID"]
        lda_model = pkg["skl_models"][step]
        t_idx = all_t_idx[:step + 1]
        if decoder_mode == "M2_CUMULATIVE":
            lda_t_idx = np.linspace(0, raw_step.shape[1] - 1, step + 1).astype(int)
            features = raw_step[:, lda_t_idx].flatten().reshape(1, -1)
        else:
            features = epoch_ch[:, t_idx].flatten().reshape(1, -1)
        p_lda = lda_model.predict_proba(features)[0]
        mi_col = list(lda_model.classes_).index(mi_id)
        p_lda_mi = float(p_lda[mi_col])
    except Exception:
        p_lda_mi = 0.5

    return p_mdm_mi, p_lda_mi


def _predict_skl_pmi(model, mi_id, features):
    """Return P(MI) from a sklearn-style model; 0.5 if unavailable."""
    try:
        proba = model.predict_proba(features)[0]
        mi_idx = list(model.classes_).index(mi_id)
        return float(proba[mi_idx])
    except Exception:
        return 0.5


def _match_mdm_template_samples(signal, template):
    """Muestrea señal y plantilla MDM con el mismo número de puntos."""
    target_samples = int(template.shape[1])
    if signal.shape[1] == target_samples:
        return signal
    if signal.shape[1] < 1 or target_samples < 1:
        raise ValueError("MDM signal/template cannot be empty")
    sample_indices = np.linspace(
        0,
        signal.shape[1] - 1,
        target_samples,
    ).round().astype(int)
    return signal[:, sample_indices]


def predict_m2_full_window(pkg, epoch_ch, recenter=False):
    """Evaluate every available observer once on the complete M2 window."""
    if pkg is None or pkg.get("model_type") != "M2_LDA_shrink_MDM":
        return {}
    if epoch_ch is None or epoch_ch.ndim != 2 or epoch_ch.shape[1] < 2:
        return {}

    n_steps = int(pkg["n_timepoints"])
    step = n_steps - 1
    all_t_idx = np.linspace(0, epoch_ch.shape[1] - 1, n_steps).astype(int)
    raw_step = epoch_ch
    p_mdm, p_lda = _predict_m2_pkg_pmi(
        pkg,
        step,
        epoch_ch,
        raw_step,
        all_t_idx,
        "M2_CUMULATIVE",
        recenter=recenter,
    )

    features = epoch_ch[:, all_t_idx].flatten().reshape(1, -1)
    mi_id = pkg["MI_ID"]
    probabilities = {
        "MDM": float(p_mdm),
        "LDA_shrink": float(p_lda),
    }

    compact_models = pkg.get("compact_lda_models", [])
    compact_picks = pkg.get("compact_lda_picks", [])
    if len(compact_models) > step and compact_picks:
        try:
            compact_indices = [
                pkg["picks"].index(channel)
                for channel in compact_picks
            ]
            compact_features = epoch_ch[
                compact_indices,
                :,
            ][:, all_t_idx].flatten().reshape(1, -1)
            probabilities["LDA_shrink_3ch"] = _predict_skl_pmi(
                compact_models[step],
                mi_id,
                compact_features,
            )
        except (KeyError, ValueError):
            pass

    observer_models = pkg.get("observer_skl_models", {})
    for observer_name in ("LR", "SVM"):
        models = observer_models.get(observer_name, [])
        if len(models) > step:
            probabilities[observer_name] = _predict_skl_pmi(
                models[step],
                mi_id,
                features,
            )

    return probabilities


# ============================================================
# OFFLINE-MATCHED VISUAL IDENTITY (Execution Indicator)
# ============================================================
NEXT_INDICATOR_POS = (0.50, 0.28)
NEXT_INDICATOR_SCALE = 1.00

def draw_arrow_directional(screen_surf, pos_x, pos_y, size, color, direction="right"):
    """
    Same arrow geometry as OFFLINE (line + triangle tip + offset correction).
    """
    line_len = size * 0.8
    tri_size = size // 2
    offset = 5  # pixels

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

    pygame.draw.line(screen_surf, color, line_start, line_end, 12)
    pygame.draw.polygon(screen_surf, color, points)


def draw_execution_identity_indicator(mode):
    """
    Draws the OFFLINE-style identity DURING EXECUTION:
      - MI Trial (mode==0): red square + right arrow
      - REST Trial (mode==1): blue circle + left arrow
    Uses same sizes/ratios as OFFLINE.
    """
    pos_x = int(screen_width * NEXT_INDICATOR_POS[0])
    pos_y = int(screen_height * NEXT_INDICATOR_POS[1])
    base_size = int(min(screen_width, screen_height) * 0.08 * NEXT_INDICATOR_SCALE)

    if mode == 0:
        # Square (red) + inner square (same as offline show_feedback)
        bg_rect = pygame.Rect(pos_x - base_size // 2, pos_y - base_size // 2, base_size, base_size)
        pygame.draw.rect(screen, (255, 50, 50), bg_rect)

        inner_rect = pygame.Rect(
            pos_x - int(base_size * 0.35),
            pos_y - int(base_size * 0.35),
            int(base_size * 0.7),
            int(base_size * 0.7),
        )
        pygame.draw.rect(screen, (255, 50, 50), inner_rect)

        draw_arrow_directional(screen, pos_x, pos_y, base_size // 2.5, (255, 255, 255), direction="right")
    else:
        # Circle (blue) + inner circle
        pygame.draw.circle(screen, (0, 120, 255), (pos_x, pos_y), base_size // 2)
        pygame.draw.circle(screen, (0, 120, 255), (pos_x, pos_y), int(base_size * 0.35))

        draw_arrow_directional(screen, pos_x, pos_y, base_size // 2.5, (255, 255, 255), direction="left")


# ----------------- Common helpers -----------------

def log_confusion_matrix_from_trial_summary(logger):
    df = pd.read_csv(logger.trial_summary_path)

    # Separate into valid and ambiguous trials
    ambiguous_trials = df[df["Predicted Label"].isna()]
    valid_trials = df.dropna(subset=["Predicted Label"])

    valid_trials.loc[:, "Predicted Label"] = valid_trials["Predicted Label"].astype(int)
    valid_trials.loc[:, "True Label"] = valid_trials["True Label"].astype(int)

    # Count correct predictions
    correct = (valid_trials["Predicted Label"] == valid_trials["True Label"]).sum()
    incorrect = len(valid_trials) - correct
    ambiguous = len(ambiguous_trials)
    total = correct + incorrect + ambiguous
    decided = correct + incorrect

    # Generate confusion matrix
    if not valid_trials.empty:
        cm = confusion_matrix(
            valid_trials["True Label"], valid_trials["Predicted Label"],
            labels=[200, 100]
        )
        logger.log_event("Confusion Matrix (Correct/Incorrect Only):")
        logger.log_event(f"  Actual 200 (MI)    | Predicted 200 (MI): {cm[0][0]} | Predicted 100 (REST): {cm[0][1]}")
        logger.log_event(f"  Actual 100 (REST)  | Predicted 200 (MI): {cm[1][0]} | Predicted 100 (REST): {cm[1][1]}")
    else:
        logger.log_event("No non-ambiguous trials to compute confusion matrix.")

    # Log summary stats
    if total:
        percent_correct_incl_ambiguous = (correct / total) * 100
        percent_correct_excl_ambiguous = (correct / decided) * 100 if decided > 0 else 0
        coverage = (decided / total) * 100
        ambiguous_mi = (ambiguous_trials["True Label"] == 200).sum()
        ambiguous_rest = (ambiguous_trials["True Label"] == 100).sum()
        total_mi = (df["True Label"] == 200).sum()
        total_rest = (df["True Label"] == 100).sum()
        correct_mi = ((valid_trials["True Label"] == 200) & (valid_trials["Predicted Label"] == 200)).sum()
        correct_rest = ((valid_trials["True Label"] == 100) & (valid_trials["Predicted Label"] == 100)).sum()
        mi_recall_incl_ambiguous = (correct_mi / total_mi) * 100 if total_mi > 0 else 0
        rest_recall_incl_ambiguous = (correct_rest / total_rest) * 100 if total_rest > 0 else 0
        logger.log_event(
            f"📊 Trial counts: total={total} | correct={correct} | incorrect={incorrect} "
            f"| ambiguous={ambiguous} | decided={decided} ({coverage:.2f}%)"
        )
        logger.log_event(
            f"📊 Ambiguous by class: MI={ambiguous_mi}/{total_mi} | REST={ambiguous_rest}/{total_rest}"
        )
        logger.log_event(
            f"📊 Class recall incl. ambiguous as misses: MI={mi_recall_incl_ambiguous:.2f}% "
            f"| REST={rest_recall_incl_ambiguous:.2f}%"
        )
        logger.log_event(f"✅ % Total Accuracy (Including ambiguous): {percent_correct_incl_ambiguous:.2f}%")
        logger.log_event(f"✅ % Decision Accuracy (Excluding ambiguous): {percent_correct_excl_ambiguous:.2f}%")
        logger.log_event(f"⚠️ Ambiguous trials (not counted in exclusive metric): {ambiguous}")
    else:
        logger.log_event("No trials available to compute statistics.")



def append_trial_probabilities_to_csv(trial_probabilities, mode, trial_number,
                                      predicted_label, early_cutout,
                                      mi_threshold, rest_threshold, logger,
                                      phase):
        # ── SIMULATION MODE — lista vacía, nada que guardar ──────
    if len(trial_probabilities) == 0:
        logger.log_event(f"ℹ️ No probabilities to log (simulation mode) — Trial {trial_number}")
        return

    correct_class = 200 if mode == 0 else 100
    trial_probabilities = np.array(trial_probabilities)

    if trial_probabilities.shape[1] != 3:
        logger.log_event(f"❌ Error: Unexpected shape {trial_probabilities.shape}. Expected (N,3). Skipping save.")
        return

    for row in trial_probabilities:
        timestamp, prob_rest, prob_mi = row
        logger.log_decoder_output(
            trial=trial_number,
            timestamp=timestamp,
            prob_mi=prob_mi,
            prob_rest=prob_rest,
            true_label=correct_class,
            predicted_label=predicted_label,
            early_cutout=early_cutout,
            mi_threshold=mi_threshold,
            rest_threshold=rest_threshold,
            phase=phase
        )

    logger.log_event(
        f"✅ Logged {len(trial_probabilities)} rows for Trial {trial_number} | "
        f"True: {correct_class}, Predicted: {predicted_label}, Early Cut: {early_cutout}, Phase: {phase}"
    )


def display_fixation_period(duration=3, eeg_state=None):
    """
    Displays a blank screen with a fixation cross for a given duration.

    Parameters:
    - duration (int): Time in seconds for which the fixation period lasts.
    - eeg_state: Optional EEGState object to be updated during the fixation period.
    """
    start_time = time.time()
    clock = pygame.time.Clock()

    while time.time() - start_time < duration:
        # Fill screen with background color
        pygame.display.get_surface().fill(config.black)

        # Draw UI elements
        draw_fixation_cross(screen_width, screen_height)
        draw_ball_fill(0, screen_width, screen_height)
        draw_arrow_fill(0, screen_width, screen_height)
        draw_time_balls(0, screen_width, screen_height)

        pygame.display.flip()

        # Update EEG buffer if provided
        if eeg_state is not None:
            eeg_state.update()

        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        clock.tick(60)


# Interpolation function to compute fill amount between SHAPE_MIN and SHAPE_MAX
def interpolate_fill(value):
    return max(0, min(1, (value - config.SHAPE_MIN) / (config.SHAPE_MAX - config.SHAPE_MIN)))

def calculate_fill_levels(running_avg_confidence, mode):
    """
    Determines the fill levels for both MI (arrow) and Rest (ball) based on accumulated probability.

    Parameters:
        running_avg_confidence (float): The leaky-integrated probability estimate.
        mode (int): 0 for MI trial (fill square), 1 for Rest trial (fill ball).

    Returns:
        tuple: (fill_arrow, fill_ball) - Values between 0 and 1 indicating fill levels.
    """
    # Ensure probability stays within configured bounds
    prob = max(0, min(1, running_avg_confidence))
    prob_inverse = 1 - prob  # Inverse probability for the other shape

    # Determine fill levels
    fill_mi = interpolate_fill(prob) if prob >= config.SHAPE_MIN else 0
    fill_rest = interpolate_fill(prob_inverse) if prob_inverse >= config.SHAPE_MIN else 0

    # Swap roles if in Rest mode
    if mode == 1:
        return fill_rest, fill_mi
    return fill_mi, fill_rest


def handle_fes_activation(mode, running_avg_confidence, fes_active):
    """
    Manages the activation of sensory FES based on the running average probability.
    """
    fes_should_be_active = (mode == 0 and running_avg_confidence > 0.5) or \
                           (mode == 1 and running_avg_confidence < 0.5)

    if fes_should_be_active and not fes_active:
        if FES_toggle == 1:
            send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_SENS_GO", logger=logger)
            logger.log_event("Sensory FES activated.")
        else:
            logger.log_event("FES toggle is off — activation skipped.")
        return True

    elif not fes_should_be_active and fes_active:
        if FES_toggle == 1:
            send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
            logger.log_event("Sensory FES stopped.")
        else:
            logger.log_event("FES toggle is off — stop command skipped.")
        return False

    return fes_active


def classify_real_time(eeg_state, window_size_samples, all_probabilities, predictions,
                       mode, leaky_integrator, update_recentering=True, elapsed_ms=None):
    global counter, Prev_T, prep_epoch
    global m2_ch_idx, _m2_last_step, _m2_lda_probs, _m2_lda_compact_probs
    global _m2_mdm_probs
    global _m2_lr_probs, _m2_svm_probs
    global _m2_shadow_records
    global _m2_warmup_mdm_probs, _m2_warmup_lda_probs
    global _m2_quality_reject_count, _m2_quality_bad_trial, _m2_quality_last_logged_step

    pygame.display.flip()
    pygame.event.get()  # Heartbeat to OS

    # ── M2 cross-subject mode ─────────────────────────────────
    if model_pkg is not None and model_pkg.get('model_type') == 'M2_LDA_shrink_MDM':
        decoder_mode = _prep_decoder_mode()
        epoch_source = prep_epoch
        window_mode = decoder_mode

        if decoder_mode == "M2_CUMULATIVE":
            try:
                epoch_source, _ = eeg_state.get_baseline_corrected_window(window_size_samples)
                prep_epoch = epoch_source
            except ValueError:
                epoch_source = prep_epoch
                window_mode = "M2_CUMULATIVE_FALLBACK"

        if epoch_source is None:
            return leaky_integrator.accumulated_probability, predictions, all_probabilities

        # Lazy init: find channel indices in live stream
        if m2_ch_idx is None:
            if eeg_state.channel_names:
                picks = model_pkg['picks']
                m2_ch_idx = [list(eeg_state.channel_names).index(ch)
                              for ch in picks
                              if ch in eeg_state.channel_names]

        if not m2_ch_idx:
            return leaky_integrator.accumulated_probability, predictions, all_probabilities

        n_steps = model_pkg['n_timepoints']
        requested_step = min(
            int((elapsed_ms or 0) / 250),
            n_steps - 1,
        )
        control_endpoint = float(
            getattr(
                config,
                "PREP_CONTROL_ENDPOINT",
                model_pkg["t_points"][-1],
            )
        )
        max_control_step = int(
            np.argmin(
                np.abs(
                    np.asarray(model_pkg["t_points"], dtype=float)
                    - control_endpoint
                )
            )
        )
        step = min(requested_step, max_control_step)
        REST_ID   = model_pkg['REST_ID']
        MI_ID     = model_pkg['MI_ID']

        # Sample indices within the 2.5s epoch window
        n_samp    = epoch_source.shape[1]
        all_t_idx = np.linspace(0, n_samp - 1, n_steps).astype(int)

        # Extract channels from the selected epoch source
        epoch_ch  = epoch_source[m2_ch_idx, :]   # (3, n_samp)

        quality_ok, quality_reason, quality_metrics = _check_eeg_quality(epoch_ch)
        if not quality_ok:
            _m2_quality_reject_count += 1
            _m2_quality_bad_trial = True
            if logger and step != _m2_quality_last_logged_step:
                _m2_quality_last_logged_step = step
                logger.log_event(
                    f"[BAD_EEG] step={step+1:02d}/{n_steps} "
                    f"reason={quality_reason} "
                    f"ptp_max={quality_metrics.get('ptp_max', float('nan')):.1f} "
                    f"rms_max={quality_metrics.get('rms_max', float('nan')):.1f} "
                    f"abs_max={quality_metrics.get('abs_max', float('nan')):.1f} "
                    f"bad_ch(ptp/rms/abs/flat)="
                    f"{quality_metrics.get('high_ptp_ch', 0)}/"
                    f"{quality_metrics.get('high_rms_ch', 0)}/"
                    f"{quality_metrics.get('high_abs_ch', 0)}/"
                    f"{quality_metrics.get('flat_ch', 0)} "
                    f"-> classification skipped"
                )
            return leaky_integrator.accumulated_probability, predictions, all_probabilities

        # MDM decision. Keep raw P(MI)/P(REST) separate from the
        # trial-oriented confidence used for early-stop.
        p_mi = 0.5
        p_rest = 0.5
        mdm_confidence = 0.5
        mdm_valid = False
        lda_valid = False
        lda_compact_valid = False
        lr_valid = False
        svm_valid = False
        model_errors = {}
        t_end = all_t_idx[step] + 1
        if decoder_mode == "M2_CUMULATIVE":
            raw_step = epoch_ch[:, -t_end:]
        else:
            raw_step = epoch_ch[:, :t_end]
        try:
            mdm_model    = model_pkg['mdm_models'][step]
            mdm_template = model_pkg['mdm_templates'][step]
            tmpl_cut     = mdm_template
            raw_step_mdm = _match_mdm_template_samples(
                raw_step,
                tmpl_cut,
            )
            extended     = np.concatenate(
                [raw_step_mdm, tmpl_cut],
                axis=0,
            )
            raw_cov      = extended @ extended.T
            tr           = np.trace(raw_cov)
            if tr <= 1e-12 or not np.isfinite(tr):
                raise ValueError("invalid covariance trace")
            cov_norm = raw_cov / tr
            # Riemannian recentering: whiten relative to running mean
            model_requires_recenter = (
                model_pkg.get("mdm_recenter_mode") == "train_riemann_mean"
            )
            if ((getattr(config, "RECENTERING", 0) or model_requires_recenter)
                    and m2_prev_T is not None
                    and step < len(m2_prev_T)
                    and m2_prev_T[step] is not None):
                try:
                    T_inv = invsqrtm(m2_prev_T[step])
                    cov_recentered = T_inv @ cov_norm @ T_inv.T
                    cov_recentered = 0.5 * (cov_recentered + cov_recentered.T)
                    if _is_spd_finite(cov_recentered):
                        cov_norm = cov_recentered
                except Exception as exc:
                    model_errors["MDM_RECENTER"] = str(exc)
            cov    = cov_norm + model_pkg['cov_reg'] * np.eye(cov_norm.shape[0])
            cov    = 0.5 * (cov + cov.T)
            cov    = np.expand_dims(cov, 0)
            mi_col = list(mdm_model.classes_).index(MI_ID)
            proba  = mdm_model.predict_proba(cov)[0]
            p_mi = float(proba[mi_col])
            if not np.isfinite(p_mi) or not 0.0 <= p_mi <= 1.0:
                raise ValueError(f"invalid probability {p_mi}")
            p_rest = float(1.0 - p_mi)
            mdm_confidence = p_mi if mode == 0 else p_rest
            mdm_valid = True
        except Exception as exc:
            model_errors["MDM"] = str(exc)

        p_lda_mi = 0.5
        p_lda_compact_mi = None
        p_lr_mi = 0.5
        p_svm_mi = 0.5
        try:
            t_idx = all_t_idx[:step + 1]
            if decoder_mode == "M2_CUMULATIVE":
                lda_t_idx = np.linspace(0, raw_step.shape[1] - 1, step + 1).astype(int)
                features = raw_step[:, lda_t_idx].flatten().reshape(1, -1)
            else:
                features = epoch_ch[:, t_idx].flatten().reshape(1, -1)
        except Exception as exc:
            features = None
            lda_t_idx = None
            model_errors["FEATURES"] = str(exc)

        try:
            if features is None:
                raise ValueError("features unavailable")
            lda_model = model_pkg['skl_models'][step]
            lda_proba = lda_model.predict_proba(features)[0]
            lda_mi_col = list(lda_model.classes_).index(MI_ID)
            p_lda_mi = float(lda_proba[lda_mi_col])
            if not np.isfinite(p_lda_mi) or not 0.0 <= p_lda_mi <= 1.0:
                raise ValueError(f"invalid probability {p_lda_mi}")
            lda_valid = True
        except Exception as exc:
            model_errors["LDA"] = str(exc)

        try:
            compact_models = model_pkg.get("compact_lda_models", [])
            compact_picks = model_pkg.get("compact_lda_picks", [])
            if len(compact_models) > step and compact_picks:
                if features is None or lda_t_idx is None:
                    raise ValueError("features unavailable")
                compact_indices = [
                    model_pkg["picks"].index(channel)
                    for channel in compact_picks
                ]
                compact_features = raw_step[
                    compact_indices,
                    :,
                ][:, lda_t_idx].flatten().reshape(1, -1)
                compact_model = compact_models[step]
                compact_proba = compact_model.predict_proba(compact_features)[0]
                compact_mi_col = list(compact_model.classes_).index(MI_ID)
                p_lda_compact_mi = float(compact_proba[compact_mi_col])
                if (
                    not np.isfinite(p_lda_compact_mi)
                    or not 0.0 <= p_lda_compact_mi <= 1.0
                ):
                    raise ValueError(
                        f"invalid probability {p_lda_compact_mi}"
                    )
                lda_compact_valid = True
        except Exception as exc:
            model_errors["LDA3"] = str(exc)

        observer_skl = model_pkg.get("observer_skl_models", {})
        try:
            if features is None or "LR" not in observer_skl:
                raise ValueError("model or features unavailable")
            lr_model = observer_skl["LR"][step]
            lr_proba = lr_model.predict_proba(features)[0]
            lr_mi_col = list(lr_model.classes_).index(MI_ID)
            p_lr_mi = float(lr_proba[lr_mi_col])
            if not np.isfinite(p_lr_mi) or not 0.0 <= p_lr_mi <= 1.0:
                raise ValueError(f"invalid probability {p_lr_mi}")
            lr_valid = True
        except Exception as exc:
            model_errors["LR"] = str(exc)

        try:
            if features is None or "SVM" not in observer_skl:
                raise ValueError("model or features unavailable")
            svm_model = observer_skl["SVM"][step]
            svm_proba = svm_model.predict_proba(features)[0]
            svm_mi_col = list(svm_model.classes_).index(MI_ID)
            p_svm_mi = float(svm_proba[svm_mi_col])
            if not np.isfinite(p_svm_mi) or not 0.0 <= p_svm_mi <= 1.0:
                raise ValueError(f"invalid probability {p_svm_mi}")
            svm_valid = True
        except Exception as exc:
            model_errors["SVM"] = str(exc)

        control_model = str(getattr(config, "PREP_CONTROL_MODEL", "MDM")).upper()
        if control_model in {"LDA", "LDA_SHRINK", "LDA_SHRINKAGE"}:
            p_control_mi = p_lda_mi
            control_name = "LDA_shrink"
            control_valid = lda_valid
        elif control_model in {"LDA3", "LDA_3CH", "LDA_SHRINK_3CH", "COMPACT_LDA"}:
            p_control_mi = p_lda_compact_mi if p_lda_compact_mi is not None else 0.5
            control_name = "LDA3"
            control_valid = lda_compact_valid
        elif control_model == "LR":
            p_control_mi = p_lr_mi
            control_name = "LR"
            control_valid = lr_valid
        elif control_model == "SVM":
            p_control_mi = p_svm_mi
            control_name = "SVM"
            control_valid = svm_valid
        else:
            p_control_mi = p_mi
            control_name = "MDM"
            control_valid = mdm_valid

        p_control_rest = 1.0 - p_control_mi
        control_confidence = p_control_mi if mode == 0 else p_control_rest
        predicted_label = 200 if p_control_mi >= 0.5 else 100

        # ── Acumular y loguear una vez por paso ───────────────
        # The online loop calls this function faster than M2 advances its
        # temporal step. Keep prep decisions based on unique M2 steps, not
        # repeated display/classification ticks within the same step.
        if step != _m2_last_step:
            _m2_last_step = step
            warmup_mdm_mi, warmup_lda_mi = _predict_m2_pkg_pmi(
                observer_model_pkg,
                step,
                epoch_ch,
                raw_step,
                all_t_idx,
                decoder_mode,
                recenter=False,
            )
            _m2_lda_probs.append(
                round(p_lda_mi, 3) if lda_valid else np.nan
            )
            if lda_compact_valid:
                _m2_lda_compact_probs.append(
                    round(p_lda_compact_mi, 3)
                )
            _m2_mdm_probs.append(round(p_mi, 3) if mdm_valid else np.nan)
            _m2_lr_probs.append(round(p_lr_mi, 3) if lr_valid else np.nan)
            _m2_svm_probs.append(round(p_svm_mi, 3) if svm_valid else np.nan)
            _m2_shadow_records.append({
                "step_index": int(step),
                "step": int(step + 1),
                "time": float(model_pkg["t_points"][step]),
                "probabilities": {
                    "MDM": float(p_mi) if mdm_valid else None,
                    "LDA": float(p_lda_mi) if lda_valid else None,
                    "LDA3": (
                        float(p_lda_compact_mi)
                        if lda_compact_valid
                        else None
                    ),
                    "LR": float(p_lr_mi) if lr_valid else None,
                    "SVM": float(p_svm_mi) if svm_valid else None,
                },
            })
            if observer_model_pkg is not None:
                _m2_warmup_mdm_probs.append(round(warmup_mdm_mi, 3))
                _m2_warmup_lda_probs.append(round(warmup_lda_mi, 3))
            if logger:
                compact_log = (
                    f"{p_lda_compact_mi:.3f}"
                    if lda_compact_valid
                    else "NA"
                )
                mdm_log = f"{p_mi:.3f}" if mdm_valid else "NA"
                lda_log = f"{p_lda_mi:.3f}" if lda_valid else "NA"
                lr_log = f"{p_lr_mi:.3f}" if lr_valid else "NA"
                svm_log = f"{p_svm_mi:.3f}" if svm_valid else "NA"
                logger.log_event(
                    f"[M2_step] paso={step+1:02d}/{n_steps} "
                    f"t={model_pkg['t_points'][step]:+.2f}s  "
                    f"window={window_mode}  "
                    f"MDM_PMI={mdm_log}  "
                    f"LDA={lda_log}  "
                    f"LDA3={compact_log}  "
                    f"LR={lr_log}  "
                    f"SVM={svm_log}  "
                    f"control={control_name}  "
                    f"control_valid={control_valid}  "
                    f"conf_{'MI' if mode == 0 else 'REST'}="
                    f"{control_confidence:.3f}"
                )
                for failed_model, error_text in model_errors.items():
                    logger.log_event(
                        f"[MODEL_ERROR] step={step+1:02d}/{n_steps} "
                        f"model={failed_model} error={error_text}"
                    )
                if observer_model_pkg is not None:
                    logger.log_event(
                        f"[M2_COMPARE] paso={step+1:02d}/{n_steps} "
                        f"master_mdm_pmi={p_mi:.3f} "
                        f"master_lda_pmi={p_lda_mi:.3f} "
                        f"warmup_mdm_pmi={warmup_mdm_mi:.3f} "
                        f"warmup_lda_pmi={warmup_lda_mi:.3f} "
                        f"control={'MI' if p_control_mi >= 0.5 else 'REST'}"
                    )

            if control_valid:
                predictions.append(predicted_label)
                all_probabilities.append(
                    [time.time(), p_control_rest, p_control_mi]
                )
        return control_confidence, predictions, all_probabilities

    # ── Legacy mode ───────────────────────────────────────────
    try:
        window, _ = eeg_state.get_baseline_corrected_window(window_size_samples)
    except ValueError:
        return leaky_integrator.accumulated_probability, predictions, all_probabilities


    if not np.isfinite(window).all():
        return leaky_integrator.accumulated_probability, predictions, all_probabilities

    T_test = None

    if hasattr(model, 'covmeans_'):
        # === MDM — Covariance Matrix (template matching Racz 2023 — 10×10) ===
        t_len    = window.shape[1]
        extended  = np.concatenate([window, template[:, :t_len]], axis=0)
        raw_cov   = extended @ extended.T
        trace_val = np.trace(raw_cov)

        if not np.isfinite(trace_val) or trace_val < 1e-12:
            return leaky_integrator.accumulated_probability, predictions, all_probabilities

        cov_matrix  = raw_cov / trace_val
        cov_matrix += 1e-4 * np.eye(10)
        cov_matrix  = np.expand_dims(cov_matrix, axis=0)

        if config.RECENTERING:
            cov_matrix = np.squeeze(cov_matrix, axis=0)
            if counter == 0 or Prev_T is None:
                Prev_T = cov_matrix
            T_test     = geodesic_riemann(Prev_T, cov_matrix, 1 / (counter + 1))
            T_invsqrtm = invsqrtm(Prev_T)
            cov_matrix = T_invsqrtm @ cov_matrix @ T_invsqrtm.T
            cov_matrix = np.expand_dims(cov_matrix, axis=0)

        probabilities = model.predict_proba(cov_matrix)[0]
        mdm_predicted = model.classes_[np.argmax(probabilities)]
        predicted_label   = 200 if mdm_predicted == 2 else 100
        mdm_correct_label = 2 if mode == 0 else 1
        correct_class_idx = np.where(model.classes_ == mdm_correct_label)[0][0]
        current_confidence = probabilities[correct_class_idx]

    else:
        # === LDA/sklearn — features de amplitud en puntos temporales ===
        n_samples = window.shape[1]
        t_idx     = np.linspace(0, n_samples - 1, 11).astype(int)
        features  = window[:, t_idx].flatten().reshape(1, -1)
        if not np.isfinite(features).all():
            return leaky_integrator.accumulated_probability, predictions, all_probabilities
        probabilities  = model.predict_proba(features)[0]
        classes        = model.classes_
        mi_idx         = int(np.argmax(classes))   # clase mayor = MI (200 o 2)
        rest_idx       = 1 - mi_idx
        predicted_label    = 200 if np.argmax(probabilities) == mi_idx else 100
        correct_class_idx  = mi_idx if mode == 0 else rest_idx
        current_confidence = probabilities[correct_class_idx]

    # === Determine if recentering update should occur ===
    should_update_T = False
    if config.RECENTERING and update_recentering:
        if config.USE_CONFIDENCE_GATE:
            predicted_correct = (predicted_label == (200 if mode == 0 else 100))
            confident_enough  = (current_confidence >= config.RECENTERING_CONFIDENCE_THRESHOLD)
            should_update_T   = predicted_correct and confident_enough
        else:
            should_update_T = True

    if should_update_T:
        Prev_T = T_test
        counter += 1

    predictions.append(predicted_label)
    all_probabilities.append([time.time(), probabilities[0], probabilities[1]])

    return current_confidence, predictions, all_probabilities


def update_m2_recentering(
    prep_prediction=None,
    target_label=None,
    prep_confidence=None,
):
    """Actualiza el recentering Riemanniano M2 al final de cada trial.

    Calcula la covarianza extendida (señal + template) para cada uno de los
    pasos usando el prep_epoch capturado al inicio del trial, y actualiza
    m2_prev_T[step] mediante interpolación geodésica suave.
    """
    global m2_prev_T, m2_rec_counter, m2_rec_seen_trials

    if not getattr(config, "RECENTERING", 0):
        return

    if (
        getattr(config, "RECENTERING_REQUIRE_NON_AMBIGUOUS", True)
        and prep_prediction is None
    ):
        if logger:
            logger.log_event("[M2_recentering] skipped — ambiguous decision")
        return

    if (
        getattr(config, "RECENTERING_REQUIRE_CORRECT", True)
        and target_label is not None
        and prep_prediction != target_label
    ):
        if logger:
            logger.log_event("[M2_recentering] skipped — decision did not match target")
        return

    min_conf = float(getattr(config, "RECENTERING_MIN_CONFIDENCE", 0.0))
    if prep_confidence is not None and prep_confidence < min_conf:
        if logger:
            logger.log_event(
                f"[M2_recentering] skipped — confidence={prep_confidence:.3f} "
                f"< {min_conf:.3f}"
            )
        return

    if _m2_quality_bad_trial:
        if logger:
            logger.log_event("[M2_recentering] skipped — BAD_EEG detected during trial")
        return

    if model_pkg is None or prep_epoch is None or not m2_ch_idx:
        return

    m2_rec_seen_trials += 1
    min_trials = int(getattr(config, "RECENTERING_MIN_TRIALS", 0))
    if m2_rec_seen_trials <= min_trials:
        if logger:
            logger.log_event(
                f"[M2_recentering] warmup {m2_rec_seen_trials}/{min_trials} — no update"
            )
        return

    n_steps   = model_pkg['n_timepoints']
    n_samp    = prep_epoch.shape[1]
    all_t_idx = np.linspace(0, n_samp - 1, n_steps).astype(int)
    epoch_ch  = prep_epoch[m2_ch_idx, :]

    if m2_prev_T is None:
        m2_prev_T = [None] * n_steps

    alpha       = float(getattr(config, "RECENTERING_ALPHA", 0.05))
    alpha       = min(max(alpha, 0.0), 1.0)
    updated     = 0
    n_ext       = len(m2_ch_idx) * 2   # canales señal + canales template
    min_samples = n_ext + 2             # mínimo para covarianza de rango completo

    for step in range(n_steps):
        try:
            mdm_template = model_pkg['mdm_templates'][step]
            t_end  = all_t_idx[step] + 1
            if mdm_template.shape[1] < min_samples:
                continue  # covarianza rango-deficiente — saltar este paso
            raw_s  = epoch_ch[:, :t_end]
            tmpl_s = mdm_template
            raw_s = _match_mdm_template_samples(raw_s, tmpl_s)
            ext    = np.concatenate([raw_s, tmpl_s], axis=0)
            raw_cov = ext @ ext.T
            tr = np.trace(raw_cov)
            if tr < 1e-12 or not np.isfinite(tr):
                continue
            cov = raw_cov / tr
            cov = 0.5 * (cov + cov.T)
            cov += model_pkg['cov_reg'] * np.eye(cov.shape[0])
            if not _is_spd_finite(cov):
                continue

            if m2_prev_T[step] is None:
                m2_prev_T[step] = cov
            else:
                candidate = geodesic_riemann(m2_prev_T[step], cov, alpha)
                candidate = 0.5 * (candidate + candidate.T)
                if not _is_spd_finite(candidate):
                    continue
                m2_prev_T[step] = candidate
            updated += 1
        except Exception:
            pass

    if updated > 0:
        m2_rec_counter += 1
    if logger:
        logger.log_event(
            f"[M2_recentering] accepted_updates={m2_rec_counter} | "
            f"seen={m2_rec_seen_trials} | α={alpha:.3f} | "
            f"pasos actualizados={updated}/{n_steps}"
        )


def hold_messages_and_classify(messages, colors, offsets, duration, mode, udp_socket, udp_ip, udp_port,
                               eeg_state, leaky_integrator):
    """
    Holds visual messages on the screen while running real-time EEG classification in the background.
    """
    font = pygame.font.SysFont(None, 72)
    start_time = time.time()
    early_stop = False

    step_size = config.STEP_SIZE
    window_size = config.CLASSIFY_WINDOW / 1000
    window_size_samples = int(window_size * config.FS)

    correct_class = 200 if mode == 0 else 100
    incorrect_class = 100 if mode == 0 else 200

    min_predictions_before_stop = config.MIN_PREDICTIONS
    num_predictions = 0
    accuracy_threshold = config.THRESHOLD_MI if mode == 0 else config.THRESHOLD_REST

    all_probabilities = []
    predictions = []
    running_avg_confidence = 0.5
    current_confidence = 0.5

    next_tick = time.time()
    pygame.display.update()
    clock = pygame.time.Clock()

    while time.time() - start_time < duration:
        now = time.time()

        eeg_state.update()

        pygame.display.get_surface().fill((0, 0, 0))
        for i, text in enumerate(messages):
            message = font.render(text, True, colors[i])
            pygame.display.get_surface().blit(
                message,
                (pygame.display.get_surface().get_width() // 2 - message.get_width() // 2,
                 pygame.display.get_surface().get_height() // 2 + offsets[i])
            )
        pygame.display.flip()

        if now >= next_tick:
            current_confidence, predictions, all_probabilities = classify_real_time(
                eeg_state, window_size_samples,
                all_probabilities, predictions,
                mode, leaky_integrator,
                update_recentering=config.UPDATE_DURING_MOVE
            )
            next_tick += step_size
            if all_probabilities and getattr(config, "SEND_PROBS", False):
                prob_mi, prob_rest = all_probabilities[-1][2], all_probabilities[-1][1]
                send_udp_message(
                    udp_socket_marker,
                    config.UDP_MARKER["IP"],
                    config.UDP_MARKER["PORT"],
                    f"{config.TRIGGERS['ROBOT_PROBS']},{prob_mi:.5f},{prob_rest:.5f}",
                    quiet=True
                )

            if current_confidence > 0:
                num_predictions += 1

            running_avg_confidence = leaky_integrator.update(current_confidence)

            if num_predictions >= min_predictions_before_stop and running_avg_confidence < config.RELAXATION_RATIO * accuracy_threshold:
                early_stop = True

                logger.log_event(
                    f"Early stop triggered! Confidence: {running_avg_confidence:.2f} after {num_predictions} predictions"
                )

                send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"],
                                 config.TRIGGERS["ROBOT_EARLYSTOP"], logger=logger)
                send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"],
                                 config.TRIGGERS["ROBOT_END"], logger=logger)

                if FES_toggle == 1:
                    send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"],
                                     "FES_STOP", logger=logger)
                    logger.log_event("FES_STOP signal sent due to early stop.")
                else:
                    logger.log_event("FES is disabled — no FES_STOP sent.")

                display_multiple_messages_with_udp(
                    ["Stopping Robot"], [(255, 0, 0)], [0], duration=3,
                    udp_messages=[config.ROBOT_OPCODES["STOP"]],
                    udp_socket=udp_socket, udp_ip=udp_ip, udp_port=udp_port, logger=logger
                )
                break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        clock.tick(60)

    if not early_stop:
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"],
                         config.TRIGGERS["ROBOT_END"], logger=logger)

    final_class = correct_class if running_avg_confidence >= config.RELAXATION_RATIO * accuracy_threshold else incorrect_class
    logger.log_event(f"Confidence at the end of motion: {running_avg_confidence:.2f} after {num_predictions} predictions")

    return final_class, all_probabilities, early_stop


def show_feedback(duration=5, mode=0, eeg_state=None):
    """
    Displays feedback animation, collects EEG data, and performs real-time classification
    using a sliding window approach with early stopping based on posterior probabilities.
    """
    start_time = time.time()
    step_size = config.STEP_SIZE
    window_size = config.CLASSIFY_WINDOW / 1000
    window_size_samples = int(window_size * config.FS)

    FES_active = False
    all_probabilities = []
    predictions = []
    leaky_integrator = LeakyIntegrator(alpha=config.INTEGRATOR_ALPHA)
    min_predictions = config.MIN_PREDICTIONS
    earlystop_flag = False

    correct_class = 200 if mode == 0 else 100
    incorrect_class = 100 if mode == 0 else 200

    accuracy_threshold = config.THRESHOLD_MI if mode == 0 else config.THRESHOLD_REST
    opposed_threshold = config.THRESHOLD_REST if mode == 0 else config.THRESHOLD_MI

    screen = pygame.display.get_surface()
    screen.fill(config.black)
    pygame.display.flip()

    # Send UDP triggers

    if mode == 0:
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"],
                         config.TRIGGERS["MI_BEGIN"], logger=logger)

    else:
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"],
                         config.TRIGGERS["REST_BEGIN"], logger=logger)

    # ── SIMULATION MODE — skip classification ───────────────── #Esta parte yo la agregue para probar la Simulacion con Online
    if getattr(config, 'SIMULATION_MODE', False):
        pygame.time.wait(int(duration * 1000))
        return None, 0.5, leaky_integrator, [], False
    
    clock = pygame.time.Clock()
    running_avg_confidence = 0.0
    current_confidence = 0.5
    next_tick = start_time + window_size

    while time.time() - start_time < duration:
        eeg_state.update()

        now = time.time()
        if now >= next_tick:
            current_confidence, predictions, all_probabilities = classify_real_time(
                eeg_state,
                window_size_samples,
                all_probabilities,
                predictions,
                mode,
                leaky_integrator
            )
            next_tick += step_size

            if all_probabilities and getattr(config, "SEND_PROBS", False):
                prob_mi, prob_rest = all_probabilities[-1][2], all_probabilities[-1][1]
                send_udp_message(
                    udp_socket_marker,
                    config.UDP_MARKER["IP"],
                    config.UDP_MARKER["PORT"],
                    f"{config.TRIGGERS['MI_PROBS' if mode == 0 else 'REST_PROBS']},{prob_mi:.5f},{prob_rest:.5f}",
                    quiet=True
                )

            # FES pulses on each correct classification step — no state machine
            if FES_toggle == 1:
                step_correct = (mode == 0 and current_confidence > 0.5) or \
                               (mode == 1 and current_confidence < 0.5)
                if step_correct:
                    send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_SENS_GO", logger=logger)
                    FES_active = True
                else:
                    send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
                    FES_active = False

        running_avg_confidence = leaky_integrator.update(current_confidence)

        # --------- DRAW (same logic, plus OFFLINE identity indicator) ---------
        screen.fill(config.black)
        MI_fill, Rest_fill = calculate_fill_levels(running_avg_confidence, mode)

        online_fill_alpha = getattr(
            config,
            "ONLINE_PREP_FEEDBACK_FILL_ALPHA",
            getattr(config, "ONLINE_FEEDBACK_FILL_ALPHA", getattr(config, "FEEDBACK_FILL_ALPHA", 180))
        )

        if mode == 0:
            draw_arrow_fill(MI_fill, screen_width, screen_height, fill_alpha=online_fill_alpha)
            draw_ball_fill(Rest_fill, screen_width, screen_height, fill_alpha=online_fill_alpha)
            message = pygame.font.SysFont(None, 96).render(
                f"Imagine closing {config.ARM_SIDE.upper()} hand", True, config.white
            )
        else:
            draw_ball_fill(Rest_fill, screen_width, screen_height, fill_alpha=online_fill_alpha)
            draw_arrow_fill(MI_fill, screen_width, screen_height, fill_alpha=online_fill_alpha)
            message = pygame.font.SysFont(None, 96).render("Rest", True, config.white)

        draw_fixation_cross(screen_width, screen_height)

        # ✅ OFFLINE identity during EXECUTION
        draw_execution_identity_indicator(mode)

        # (Optional) keep if you want: draw_time_balls(2 if mode == 0 else 3, screen_width, screen_height)

        screen.blit(message, (screen_width // 2 - message.get_width() // 2, screen_height // 2 + 300))
        pygame.display.flip()

        clock.tick(60)

        # --- Early-stop logic ---
        hit_correct = (len(predictions) >= min_predictions) and (running_avg_confidence >= accuracy_threshold)
        hit_incorrect = (len(predictions) >= min_predictions) and (running_avg_confidence <= (1 - opposed_threshold))

        should_earlystop = hit_correct or (config.EARLYSTOP_MODE == "either" and hit_incorrect)
        if should_earlystop:
            earlystop_flag = True

            if hit_correct:
                stop_reason = "correct"
                trigger_key = "MI_EARLYSTOP" if mode == 0 else "REST_EARLYSTOP"
            else:
                stop_reason = "incorrect"
                trigger_key = "REST_EARLYSTOP" if mode == 0 else "MI_EARLYSTOP"

            logger.log_event(
                f"Early stopping triggered ({stop_reason}). "
                f"Confidence={running_avg_confidence:.2f}, "
                f"min_preds={min_predictions}, "
                f"mode={'MI' if mode==0 else 'REST'}"
            )

            if FES_toggle == 1:
                send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
            else:
                logger.log_event("FES is disabled.")

            send_udp_message(
                udp_socket_marker,
                config.UDP_MARKER["IP"],
                config.UDP_MARKER["PORT"],
                config.TRIGGERS[trigger_key],
                logger=logger
            )
            break

    pygame.display.flip()

    # === Probability summary for diagnostics ===
    if all_probabilities:
        probs_arr = np.array(all_probabilities)  # shape (N, 3): [time, P_rest, P_mi]
        p_rest = probs_arr[:, 1]
        p_mi   = probs_arr[:, 2]
        mode_label = "MI" if mode == 0 else "REST"
        logger.log_event(
            f"[PROBS] {mode_label} | n={len(p_mi)} | "
            f"P(MI):   mean={p_mi.mean():.3f}  min={p_mi.min():.3f}  max={p_mi.max():.3f} | "
            f"P(REST): mean={p_rest.mean():.3f}  min={p_rest.min():.3f}  max={p_rest.max():.3f} | "
            f"integrator_final={running_avg_confidence:.3f}"
        )

    # Final Decision
    if running_avg_confidence >= accuracy_threshold:
        final_class = correct_class
    elif running_avg_confidence <= (1 - opposed_threshold):
        final_class = incorrect_class
    else:
        final_class = None

    if final_class is not None:
        logger.log_event(
            f"Final decision: {final_class}, Confidence for correct({correct_class}) class: "
            f"{running_avg_confidence:.2f}, at sample size {len(predictions)}"
        )
    else:
        logger.log_event(
            f"Ambiguous final decision — no threshold met. Confidence: {running_avg_confidence:.2f}, "
            f"MI threshold: {config.THRESHOLD_MI}, REST threshold: {config.THRESHOLD_REST}, "
            f"Samples: {len(predictions)}"
        )

    if FES_toggle == 1 and FES_active:
        send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
    else:
        logger.log_event("FES disable not needed.")

    send_udp_message(
        udp_socket_marker,
        config.UDP_MARKER["IP"],
        config.UDP_MARKER["PORT"],
        config.TRIGGERS["MI_END" if mode == 0 else "REST_END"],
        logger=logger
    )

    pygame.time.delay(300)
    return final_class, running_avg_confidence, leaky_integrator, all_probabilities, earlystop_flag
