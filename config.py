# Configuration file for EEG experiments

# Relevant Directories
# WORKING_DIR = "/home/arman-admin/Projects/Harmony/"
# DATA_DIR = "/home/arman-admin/Documents/CurrentStudy"

# WORKING_DIR = r"C:\Users\USER\OneDrive - Universidad Autonoma de Guadalajara\Estancia Texas parte 2\Proyecto\BCI"
# DATA_DIR    = r"C:\Users\USER\OneDrive - Universidad Autonoma de Guadalajara\Estancia Texas parte 2\Proyecto\CurrentStudy"


WORKING_DIR = "/home/lab-admin/BCI_project/BCI"
DATA_DIR = "/home/lab-admin/Documents/CNVStudy"
SIMULATION_MODE = False
#TRAINING_SUBJECT = "CNV_PILOT_SUBJ_012"
TRAINING_SUBJECT = "CNV_PILOT_SUBJ_016"
# Recording/logging source. Keep this separate from TRAINING_SUBJECT so the
# online decoder can use a shared model while LabRecorder saves under the
# participant/session being recorded.
RECORDING_DATA_DIR = "/home/lab-admin/Documents/CNVStudy"
RECORDING_SUBJECT = "CNV_PILOT_SUBJ_029"
# EEG Settings
CAP_TYPE = 32  
LOWCUT = 0.1  # Hz
HIGHCUT = 2.0  # Hz; aligned with paper-like expert+calibration training.
ONLINE_FILTER_ORDER = 2 # Streaming Butterworth order; matches paper-like training order.
LOWCUT_ERRP = 1 #Hz
HIGHCUT_ERRP = 10 #Hz
FS = 512  # Sampling frequency (Hz)
EEG_STREAM_MAX_AGE_S = 1.0 # Abort online decisions if LSL has delivered no samples for this long.
#MOTOR_CHANNEL_NAMES = ['FC1','FC2','C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7','P3', 'Pz', 'P4', 'P8', 'POz']
#MOTOR_CHANNEL_NAMES = ['FC5', 'FC1', 'C3', 'Cz', 'CP5', 'CP1', 'Fz']
MOTOR_CHANNEL_NAMES = ['FC3', 'FC1', 'FCz', 'C3', 'C1', 'Cz', 'CP3', 'CP1', 'CPz']

ERRP_CHANNEL_NAMES = ['F3', 'Fz', 'F4', 'FC1', 'FC2', 'Cz']
EOG_CHANNEL_NAMES = ['AUX1'] # List of EOG channel names to use
EOG_TOGGLE = 0  # Toggle to enable or disable EOG processing (1 = enabled, 0 = disabled)


# Experiment Parameters
ARM_SIDE = "Right"
EXPERIMENT_TYPE = "BASE" # BIMANUAL or BASE
TOTAL_TRIALS = 20   # Total number of trials
TOTAL_TRIALS_ERRP = 45 # Total number of trials for ErrP experiment
MAX_REPEATS = 3  # Maximum consecutive repeats of the same condition
N_SPLITS = 5  # Number of splits for KFold cross-validation
TIME_MI = 5 # time for motor imagery and rest
TIME_ROB =  5# time allocated for robot to move
TIME_STATIONARY = 1 # time for stationary feedback after no movement/failed movement trial
INTERTRIAL_DURATION = 5.0 # relaxation/fixation between trials; longer helps reduce MI carry-over
TIME_MASTER_MOVE = 5 # allowed timing for participant to position robot with master arm. Bimanual experiment.
TIMING = True #obsolete
SHAPE_MAX = 0.7 #maximum fill
SHAPE_MIN = 0.3 #minimum fill
ROBOT_TRAJECTORY = ["a"] # Not using
BIG_BROTHER_MODE = True #this toggle exports the game to the second monitor automatically, while retaining the running log in the first windows linux terminal
SEND_PROBS = False


# Early-stop policy: "correct_only" (current behavior) or "either"
EARLYSTOP_MODE = "correct_only"



# Classification Parameters
# Stable pilot condition:
# - Control uses the best current classical model from offline LOGO.
# - MDM/LR/SVM observer summaries stay in the logs for comparison.
# - Keep these fixed across participant runs unless starting a new condition.
CLASSIFY_WINDOW = 2500  # Duration of EEG data window for classification (milliseconds)
FILTER_BUFFER_SIZE = 3072 #6s at 512 Hz
BASELINE_DURATION = 1 #seconds
ONLINE_CAR_REFERENCE = "all_valid_eeg" # Current online: CAR before streaming filters; use "selected" for legacy selected-channel CAR.
ONLINE_CAR_DROP_CHANNELS = ['T7', 'T8', 'Fp1', 'Fpz', 'Fp2'] # Match TimePoints CAR base; M1/M2/AUX are already excluded upstream.
ONLINE_BASELINE_DURATION = 2.0 # Match offline baseline(-5,-3): 2 s baseline before prep.
ONLINE_BASELINE_END_OFFSET = 0.5 # Baseline ends 0.5 s before prep onset to avoid prep/transient contamination.
ACCURACY_THRESHOLD = 0.8  # OBS Accuracy threshold to determine "Correct" (plan to obsolete)
THRESHOLD_MI = 0.6 #Threshold for MI "correct"
THRESHOLD_REST = 0.50 #Threshold for REST "Correct"
# Regla híbrida pseudo-online seleccionada:
# 1) MDM endpoint -0.50 s, 2) MDM ponderado hasta -0.50 s,
# 3) consenso temporal de observadores LDA/LDA3/LR/SVM.
ENDPOINT_MDM_MI_THRESHOLD = 0.70
ENDPOINT_MDM_REST_THRESHOLD = 0.30
MDM_WEIGHTED_RESCUE_ENABLED = True
MDM_WEIGHTED_MI_THRESHOLD = 0.70
MDM_WEIGHTED_REST_THRESHOLD = 0.30
VIEWER_TEMPORAL_RESCUE_ENABLED = True
VIEWER_TEMPORAL_RESCUE_MODELS = ["LDA", "LDA3", "LR", "SVM"]
VIEWER_TEMPORAL_REQUIRED_VOTES = 3
VIEWER_TEMPORAL_MIN_VOTE_FRACTION = 0.60
RELAXATION_RATIO = 0.0 # relaxation ratio for sustained MI during movement
MIN_PREDICTIONS = 99 # Block legacy early stop; endpoint + temporal rescue control this test.
MIN_FINAL_PREDICTIONS = 6 # Min valid predictions required to accept a final prep decision
FINAL_DECISION_MIN_VOTE_FRACTION = 0.60 # Majority strength required for final prep decision
EARLYSTOP_CONSECUTIVE_PREDICTIONS = 2 # Recent same-class predictions required for early stop
OBSERVER_MIN_STEPS = 0 # Min M2 steps required for observer model trial decisions
EARLY_RAW_DECISION_MAX_STEPS = 8 # Observer-only RAW summary using first N M2 steps; does not control feedback
STEP_SIZE = 1/16
CLASSIFICATION_OFFSET = 0 # Offset for "classification window" starting point
CLASSIFICATION_SCHEME_OPT = "TIMESERIES"
#CLASSIFICATION_SCHEME_OPT = "FREQUENCY"
# Preparation decoder timing:
# "M2_CUMULATIVE" = online M2: model k receives data accumulated from prep onset to step k.
# "FROZEN_EPOCH_DEBUG" = legacy/debug path using the epoch captured at prep onset.
PREP_DECODER_MODE = "M2_CUMULATIVE"
PREP_CONTROL_MODEL = "MDM" # Online feedback control: MDM, LDA_shrink/LDA, LDA3, LR, or SVM.
PREP_EARLY_STOP_ENABLED = False # Activa feedback al superar umbral con evidencia sostenida.
SHADOW_MODEL_ANALYSIS_ENABLED = True # Diagnóstico target-independent; nunca controla el sistema.
ENDPOINT_VALIDATION_ENABLED = False # Disabled for the first combined-model online test; LDA/LR remain diagnostic observers.
EARLYSTOP_VALIDATION_ENABLED = False # LDA/LR también validan early stop MDM; si ambos discrepan => AMBIGUOUS.
# Congela el control secuencial en el mejor endpoint LOGO de S012.
# Los observadores de ventana completa se calculan igualmente al final.
PREP_CONTROL_ENDPOINT = -0.50
WARMUP_OBSERVER_ENABLED = False
WARMUP_MODEL_PATH = "/home/lab-admin/Documents/CurrentStudy/sub-CNV_PILOT_SUBJ_012/models/sub-CNV_PILOT_SUBJ_012_model_warmup_S005_OFFLINE.pkl"
ONLINE_MODEL_PATH = "/home/lab-admin/Documents/CNVStudy/sub-CNV_PILOT_SUBJ_029/models/sub-CNV_PILOT_SUBJ_029_model_expert-SUBJ021_plus-4runs_S002_OFFLINE.pkl"
OFFLINE_FEEDBACK_FILL_ALPHA = 255 # 0 transparent, 255 opaque for offline MI/REST fill visuals
OFFLINE_REST_NEUTRAL_VISUAL = True # Offline only: keep REST triggers/timing but hide REST text/blue visual cue.
ONLINE_REST_NEUTRAL_PREP_VISUAL = True # Online only: keep REST prep predictions/triggers but show neutral intertrial visual.
ONLINE_PREP_FEEDBACK_FILL_ALPHA = 20 # 0 transparent, 255 opaque for online preparation fill visuals
ONLINE_EXEC_FEEDBACK_FILL_ALPHA = 255 # 0 transparent, 255 opaque for online execution/reward fill visuals
ONLINE_PREP_VISUAL_RAMP_MS = 250 # Suavizado visual; no modifica decisiones, umbrales ni FES
ONLINE_FEEDBACK_FILL_ALPHA = ONLINE_PREP_FEEDBACK_FILL_ALPHA # Backward-compatible online default
FEEDBACK_FILL_ALPHA = ONLINE_PREP_FEEDBACK_FILL_ALPHA # Backward-compatible default for shared/legacy scripts
SURFACE_LAPLACIAN_TOGGLE = 0 #apply the surface laplacian spatial filter during online
SELECT_MOTOR_CHANNELS = 1 # toggle to select motor channels or not (can be used to select other channels too)
SELECT_ERRP_CHANNELS = 0 #toggle to select ERRP channels
INTEGRATOR_ALPHA = 0.90 # defines how fast the accumulated probability may change as new data comes in
SHRINKAGE_PARAM = 0.02 # hyperparameter for shrinkage regularization
LEDOITWOLF = 0 #Set to true to use ledoit wolf shrinkage regularization - otherwise pyreimannian will be used w/ shrinkage param shown above

# EEG signal quality gate. These checks run on the model channels before
# online classification/recentering. Units follow the live eegoSports stream
# display (uV in this setup).
EEG_QUALITY_GATE = True
EEG_QUALITY_MIN_PTP_UV = 0.2
EEG_QUALITY_MAX_PTP_UV = 150.0
EEG_QUALITY_MAX_RMS_UV = 50.0
EEG_QUALITY_MAX_ABS_UV = 150.0
EEG_QUALITY_BAD_CHANNELS_TO_REJECT = 2 # More flexible: reject only if this many model channels are bad
EEG_QUALITY_HARD_MAX_ABS_UV = 500.0 # Safety cutoff: one channel above this rejects immediately
PREP_COUNTDOWN_BAR_HEIGHT = 44

# adaptive Recentering parameters for config
RECENTERING = 1 # Enable online adaptive recentering for M2/MDM.
RECENTERING_MIN_TRIALS = 2 # Observe this many usable trials before the first update.
RECENTERING_ALPHA = 0.05 # Small geodesic update; avoids one trial dominating Prev_T.
RECENTERING_REQUIRE_NON_AMBIGUOUS = True
RECENTERING_REQUIRE_CORRECT = True
RECENTERING_MIN_CONFIDENCE = 0.55
M2_INIT_RECENTER_FROM_TRAINING = True # Initialize M2 whitening from the training Riemannian mean when available.
M2_USE_SAVED_ADAPTIVE_RECENTER = True # Continue from adaptive_T.pkl when available; otherwise start from training whitening.
ADAPTIVE_CONTINUITY_MODE = "fresh" # "current_session", "previous_same_condition", or "fresh".
ADAPTIVE_RECENTER_LOAD_PATH = None # Optional manual override. Keep None for automatic subject/session/condition handling.
USE_CONFIDENCE_GATE = 0 #update Previous transform ONLY in the event of lean condition
UPDATE_DURING_MOVE = 0 #this toggle defines whether or not the reimannian adaptive recentering scheme updates when the robot is moving. 0 = no, 1 = yes. The algo will update always during MI
SAVE_ADAPTIVE_T = True #this toggle saves "Adaptive_T" to the EEG directory during an active session between runs - this way, we can continue w/ the current estimated whitening transform. Disabling this will start a fresh transform each time


# FES Parameters
FES_toggle = 1
FES_CHANNEL = "blue"
FES_TIMING_OFFSET = 7 
# above for motor FES, cut out X seconds before the full duration of movement. This should represent when the robot will naturally reach the end of motion (in successful case)

# Feedback mode durante preparación:
#   "NONE"  = no activar nada durante preparación; el guante/FES solo se usan como reward final.
#   "GLOVE" = pulso del guante durante preparación MI.
#   "FES"   = FES sensorial durante preparación MI si FES_toggle == 1.
PREP_FEEDBACK_MODE = "FES"

FORCE_MI_PREDICTION = False  # ← poner False cuando uses gorra real


# Screen Dimensions
#SCREEN_WIDTH = 3840
#SCREEN_HEIGHT = 2160

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

USE_PREVIOUS_ONLINE_STATS = False # for z-score normalization of data coming in - this defines the starting point, False = use the stats from the training session, true = use previous online stats


# Colors
black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 120, 255)
red = (255, 0, 0)
green = (0, 255, 0)
orange = (255, 165, 0)

# software triggers
TRIGGERS = {
    "MI_BEGIN": "200",
    "MI_PREPARE": "210",
    "MI_END": "220",
    "MI_EARLYSTOP": "240",
    "MI_PROBS": "2000",

    #"TRAJECTORY_STAGE": "290",
    #"ACK_TRAJECTORY_STAGE":"295",
    "ROBOT_BEGIN": "300",
    "ACK_ROBOT_BEGIN": "305",
    "ROBOT_END": "320",
    "ACK_ROBOT_END": "325",
    "ROBOT_EARLYSTOP": "340",
    "ACK_ROBOT_STOP": "345",
    "ROBOT_PROBS": "3000",
    #"ROBOT_RESTART": "350",
    "ROBOT_PAUSE": "360",
    "ACK_ROBOT_PAUSE": "365",
    "ROBOT_RESUME": "370",
    "ACK_ROBOT_RESUME": "375",
    "ROBOT_HOME": "380",
    "ACK_ROBOT_HOME": "385",


    "ERRP_BEGIN": "400",
    "ERRP_END": "420",
    
    
    "REST_BEGIN": "100",
    "REST_PREPARE": "110",
    "REST_END": "120",
    "REST_EARLYSTOP": "140",
    "REST_PROBS": "1000",

    "INTERTRIAL_BEGIN": "600",
    "INTERTRIAL_END": "620",

    "MASTER_UNLOCK": "500",
    "ACK_MASTER_UNLOCK": "505",
    "MASTER_LOCK": "520",
    "ACK_MASTER_LOCK": "525",


}

# Robot Opcodes (symbolic names → opcode character)
ROBOT_OPCODES = {
    "TRAJECTORY_A": "a",      # straight ahead motion
    "TRAJECTORY_X": "x",      # Slightly raised motion
    "TRAJECTORY_Y": "y",      # Accross side motion
    "TRAJECTORY_Z": "z",      # Reach up motion
    "GO": "g",                # Execute movement (after MI success)
    "HOME": "h",              # Home
    "STOP": "s",              # Stop. Will return home automatically after several seconds
    "PAUSE": "p",             # Pause (several window allowed for resume)
    "RESUME": "r",            # Resume (resume trajectory if paused)
    "MASTER_UNLOCK": "m",     # Unlock master arm
    "MASTER_LOCK": "c",       # Lock master arm
    "QUERY": "q",             # Query joint angles, torques, velocities, end effector positions.
    "EXIT": "e"               # Exit / emergency stop
}

# UDP Settings
UDP_MARKER = {
    "IP": "127.0.0.1",
    "PORT": 15000
}

UDP_ROBOT = {
    "IP": "192.168.2.1",
    "PORT": 8080
}

UDP_FES = {
    "IP": "127.0.0.1",
    "PORT": 5005
}

UDP_CONTROL_BIND = {
    "IP":   "0.0.0.0",  # Bind on any local interface; robot still receives UDP_ROBOT
    "PORT": 8080
}


# === Arduino actuator ===
USE_ARDUINO = False         # Enable or disable Arduino actuator
ARDUINO_PORT = "/dev/ttyACM0"  # Windows: COMn / macOS: /dev/cu.usbmodemXXX
ARDUINO_BAUD = 9600         # Arduino communication baud rate

# Command mapping based on classifier output.
# Current pilot condition: detect CNV for hand OPENING.
# Previous close-hand mapping was MI=b"1", REST=b"0"; this is intentionally
# inverted so MI/reward opens the glove and rest/intertrial returns to baseline.
ARDUINO_CMD_MI   = b"0"     # Movement detected (label 200): open hand
ARDUINO_CMD_REST = b"1"     # Rest or ambiguous state detected: baseline/close
ARDUINO_MI_LABEL = "Open"
ARDUINO_INIT_SETTLE_SECONDS = 3.0 # Wait after initial close/baseline before trials start.
ARDUINO_REST_LABEL = "Rest"
