import yaml
import re
import os
from pathlib import Path

# ====================================================
# ================== LOAD CONFIG =====================
# ====================================================

def load_config(config_path=None):
    """
    Load YAML configuration file.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


# ====================================================
# ================== LOAD VARIABLES ==================
# ====================================================

_config = load_config()

# --- Habitat types ---
HABITAT_TARGETS = _config["target_habitat"]
NUMBER_OF_HABITATS = _config["number_of_habitats"]

# --- Metrics selection ---
METRICS_TO_COMPUTE = _config["metrics_to_compute"]
SUITABLE_HAB_SHIFTS = _config["suitable_hab_shifts"]

# --- Raster parameters ---
CRS = _config["crs"]

# --- Thresholds ---
DEPTH_THRESHOLD = _config["depth_threshold"]

# --- Project structure ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FOLDER = PROJECT_ROOT / "data" / "input"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "output"

# --- Zone selection ---
AREA_FOLDER = INPUT_FOLDER / "Study_zone"
SHP_INPUT_FILE = os.path.join(AREA_FOLDER, _config["study_zone_shp"])


# --- Time selection - output folders creation---
OUTPUT_FOLDER_TIME = os.path.join(OUTPUT_FOLDER, "Typical_week_analysis")
os.makedirs(OUTPUT_FOLDER_TIME, exist_ok=True)
TIME_STEP_MIN = _config["time_step_min"]
TYPICAL_FLOW_FILENAME = _config["typical_filename"]


# --- Time parameters to calculate daily metrics---
TIMESTEPS_PER_DAY = int((24 * 60) / TIME_STEP_MIN)

# --- Hydro data ---
DATA_DIR_HYDRO = os.path.join(INPUT_FOLDER, "Flow_data")

# --- Shapefile columns ---
SHP_DEPTH_PREFIX = _config["shp_depth_prefix"]
SHP_VEL_PREFIX = _config["shp_vel_prefix"]
SHP_X_COLNAME = _config["shp_x_colname"]
SHP_Y_COLNAME = _config["shp_y_colname"]
SHP_ID_COLNAME = _config["shp_id_colname"]
SHP_SURF_COLNAME = _config["shp_surf_colname"]

# --- Threshold dictionaries ---
HABITAT_VELOCITY_THRESHOLDS = _config["habitat_velocity_thresholds"]
DRIFT_THRESHOLDS = _config["drift_thresholds_with_ramp"]
DRIFT_THRESHOLDS_NO_RAMP = _config["drift_thresholds"]
DESICCATION_THRESHOLDS = _config["desiccation_thresholds"]


# ====================================================
# ================== RUN FUNCTION ====================
# ====================================================

def run_analysis():
    print("Running analysis with:")
    print("Habitat targets:", HABITAT_TARGETS)
    print("CRS:", CRS)
    print("Zone shapefile:", SHP_INPUT_FILE)
