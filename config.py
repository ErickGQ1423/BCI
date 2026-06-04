# Configuration file for EEG experiments

# Relevant Directories
# WORKING_DIR = "/home/arman-admin/Projects/Harmony/"
# DATA_DIR = "/home/arman-admin/Documents/CurrentStudy"

# WORKING_DIR = r"C:\Users\USER\OneDrive - Universidad Autonoma de Guadalajara\Estancia Texas parte 2\Proyecto\BCI"
# DATA_DIR    = r"C:\Users\USER\OneDrive - Universidad Autonoma de Guadalajara\Estancia Texas parte 2\Proyecto\CurrentStudy"


WORKING_DIR = "/home/lab-admin/BCI_project/BCI"
DATA_DIR = "/home/lab-admin/Documents/CurrentStudy"
SIMULATION_MODE = False
#TRAINING_SUBJECT = "CNV_PILOT_SUBJ_012"
TRAINING_SUBJECT = "CNV_MAESTRO"

# Recording/logging source. Keep this separate from TRAINING_SUBJECT so the
# online decoder can use a shared model while LabRecorder saves under the
# participant/session being recorded.
RECORDING_DATA_DIR = "/home/lab-admin/Documents/CNVStudy"
RECORDING_SUBJECT = "CNV_PILOT_SUBJ_012"
# EEG Settings
CAP_TYPE = 32  
LOWCUT = 0.1  # Hz
HIGHCUT = 1.0  # Hz
LOWCUT_ERRP = 1 #Hz
HIGHCUT_ERRP = 10 #Hz
FS = 512  # Sampling frequency (Hz)
#MOTOR_CHANNEL_NAMES = ['FC1','FC2','C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7','P3', 'Pz', 'P4', 'P8', 'POz']
#MOTOR_CHANNEL_NAMES = ['FC5', 'FC1', 'C3', 'Cz', 'CP5', 'CP1', 'Fz']
MOTOR_CHANNEL_NAMES = ['FC5', 'FC1', 'C3', 'Cz', 'CP1', 'Fz']

ERRP_CHANNEL_NAMES = ['F3', 'Fz', 'F4', 'FC1', 'FC2', 'Cz']
EOG_CHANNEL_NAMES = ['AUX1'] # List of EOG channel names to use
EOG_TOGGLE = 0  # Toggle to enable or disable EOG processing (1 = enabled, 0 = disabled)


# Experiment Parameters
ARM_SIDE = "Right"
EXPERIMENT_TYPE = "BASE" # BIMANUAL or BASE
TOTAL_TRIALS = 10   # Total number of trials
TOTAL_TRIALS_ERRP = 45 # Total number of trials for ErrP experiment
MAX_REPEATS = 3  # Maximum consecutive repeats of the same condition
N_SPLITS = 5  # Number of splits for KFold cross-validation
TIME_MI = 5 # time for motor imagery and rest
TIME_ROB =  5# time allocated for robot to move
TIME_STATIONARY = 1 # time for stationary feedback after no movement/failed movement trial
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
CLASSIFY_WINDOW = 2500  # Duration of EEG data window for classification (milliseconds)
FILTER_BUFFER_SIZE = 3072 #6s at 512 Hz
BASELINE_DURATION = 1 #seconds
ACCURACY_THRESHOLD = 0.8  # OBS Accuracy threshold to determine "Correct" (plan to obsolete)
THRESHOLD_MI = 0.50 #Threshold for MI "correct"
THRESHOLD_REST = 0.7 #Threshold for REST "Correct"
RELAXATION_RATIO = 0.0 # relaxation ratio for sustained MI during movement
MIN_PREDICTIONS = 8 # Min number of predictions during Online experiment before the decoder can end early
MIN_FINAL_PREDICTIONS = 8 # Min valid predictions required to accept a final prep decision
FINAL_DECISION_MIN_VOTE_FRACTION = 0.60 # Majority strength required for final prep decision
EARLYSTOP_CONSECUTIVE_PREDICTIONS = 3 # Recent same-class predictions required for early stop
STEP_SIZE = 1/16
CLASSIFICATION_OFFSET = 0 # Offset for "classification window" starting point
CLASSIFICATION_SCHEME_OPT = "TIMESERIES"
#CLASSIFICATION_SCHEME_OPT = "FREQUENCY"
# Preparation decoder timing:
# "M2_CUMULATIVE" = online M2: model k receives data accumulated from prep onset to step k.
# "FROZEN_EPOCH_DEBUG" = legacy/debug path using the epoch captured at prep onset.
PREP_DECODER_MODE = "M2_CUMULATIVE"
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
RECENTERING = 1 # adaptive recentering toggle
USE_CONFIDENCE_GATE = 0 #update Previous transform ONLY in the event of lean condition
UPDATE_DURING_MOVE = 0 #this toggle defines whether or not the reimannian adaptive recentering scheme updates when the robot is moving. 0 = no, 1 = yes. The algo will update always during MI
SAVE_ADAPTIVE_T = False #this toggle saves "Adaptive_T" to the EEG directory during an active session between runs - this way, we can continue w/ the current estimated whitening transform. Disabling this will start a fresh transform each time


# FES Parameters
FES_toggle = 1
FES_CHANNEL = "red"
FES_TIMING_OFFSET = 7 
# above for motor FES, cut out X seconds before the full duration of movement. This should represent when the robot will naturally reach the end of motion (in successful case)

# Feedback mode durante preparación: "FES" o "GLOVE"
PREP_FEEDBACK_MODE = "FES"  # cambiar a "FES" o "GLOVE" para usar estimulación

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
    "IP":   "192.168.2.2",  # your control Control Modul's IP
    "PORT": 8080
}


# === Arduino actuator ===
USE_ARDUINO = True          # Enable or disable Arduino actuator
ARDUINO_PORT = "/dev/ttyACM0"  # Windows: COMn / macOS: /dev/cu.usbmodemXXX
ARDUINO_BAUD = 9600         # Arduino communication baud rate

# Command mapping based on classifier output
ARDUINO_CMD_MI   = b"1"     # Movement detected (label 200)
ARDUINO_CMD_REST = b"0"     # Rest or ambiguous state detected
