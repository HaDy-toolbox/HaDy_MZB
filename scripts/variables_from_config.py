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

    with open(config_path, encoding="utf-8") as file:
        # with open(config_path, "r") as file:
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

FINAL_CSV_PATH = _config["final_csv_path"]
FINAL_SHP_PATH = _config["final_shp_path"]
STATIC_HABITAT_CSV_PATH = _config["static_habitat_csv_path"]

# --- Zone selection ---
AREA_FOLDER = INPUT_FOLDER / "Shapefile"
SHP_INPUT_FILE = os.path.join(AREA_FOLDER, _config["input_shp_filename"])


# --- Time selection - output folders creation---
OUTPUT_FOLDER_TIME = os.path.join(OUTPUT_FOLDER, "Toolbox_run_output")
os.makedirs(OUTPUT_FOLDER_TIME, exist_ok=True)
TIME_STEP_MIN = _config["time_step_min"]
TYPICAL_FLOW_FILENAME = _config["input_flow_timeseries_filename"]
                                

# --- Time parameters to calculate daily metrics---
TIMESTEPS_PER_DAY = int((24 * 60) / TIME_STEP_MIN)

# --- Hydro data ---
DATA_DIR_HYDRO = os.path.join(INPUT_FOLDER, "Flow_data")

# --- Zone polygons ---
ZONE_PONTE_SHP = os.path.join(INPUT_FOLDER, "Focus_zone", _config["zone_interest_filename"])
ZONE_PONTE_FIELD = _config["zone_ponte_field"]
FOCUS_ON_ZONE = _config["focus_on_zone"]
ID_POLYGON = _config["id_polygon_zone_interest"]

# --- Shapefile columns ---
SHP_DEPTH_PREFIX = _config["shp_depth_prefix"]
SHP_VEL_PREFIX = _config["shp_vel_prefix"]
SHP_X_COLNAME = _config["shp_x_colname"]
SHP_Y_COLNAME = _config["shp_y_colname"]
SHP_ID_COLNAME = _config["shp_id_colname"]
SHP_SURF_COLNAME = _config["shp_surf_colname"]

# --- Flags ---
START_AT_FIRST_OCCURENCE = _config["start_at_first_occurrence"]

# --- Threshold dictionaries ---
HABITAT_VELOCITY_THRESHOLDS = _config["habitat_velocity_thresholds"]
DRIFT_THRESHOLDS_WITH_RAMP = _config["drift_thresholds_with_ramp"]
DRIFT_THRESHOLDS_NO_RAMP = _config["drift_thresholds"]
DESICCATION_THRESHOLDS = _config["desiccation_thresholds"]
UP_RAMP = _config["up_ramp"]


# ====================================================
# ================== RUN FUNCTION ====================
# ====================================================

def run_analysis():
    print("Running analysis with:")
    print("Habitat targets:", HABITAT_TARGETS)
    print("CRS:", CRS)
    print("Zone shapefile:", SHP_INPUT_FILE)
