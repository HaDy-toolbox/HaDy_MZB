"""
Main Pipeline
--------------

Orchestrates the full habitat/metrics workflow end-to-end, from the raw
input shapefile to final metrics shapefile and preview maps.

Steps:
1. Load the input shapefile (all mesh cells, all simulated discharges).
2. Extract the list of simulated discharge values from the depth/velocity
   column names (get_discharge_values).
3. Match each timestep of the "typical flow" discharge time series to its
   closest simulated discharge (match_closest_discharge), and determine the
   distinct set of discharges actually needed for the study
   (get_study_discharges).
4. Crop the mesh shapefile to only the relevant discharge columns and to
   cells wetted at the maximum discharge (prepare_wetted_shapefile_for_
   relevant_discharges), then export it to CSV (prepare_csv).
5. Branch depending on FOCUS_ON_ZONE:
   - If True: flag cells inside the zone of interest
     (add_zone_flag_to_mesh), classify habitat only within that zone
     (attribute_habitat_types_zone_only), then compute metrics with the
     zone-specific logic (process_mesh_data_focus_on_zone).
   - If False: classify habitat by current-velocity thresholds across the
     whole mesh (attribute_habitat_current_based), then compute metrics
     with the standard logic (process_mesh_data).
6. Join the resulting metrics CSV back onto the mesh geometry to produce a
   final metrics shapefile (join_mesh_with_CSV_data).
7. Crop the shapefile to essential columns and generate preview maps
   (crop_and_map_preview).

This script is the single entry point tying together discharge matching,
habitat classification, metrics calculation, and mapping.
"""

import geopandas as gpd
import os

#variables
from variables_from_config import HABITAT_VELOCITY_THRESHOLDS, DRIFT_THRESHOLDS_NO_RAMP, DESICCATION_THRESHOLDS, HABITAT_TARGETS, ZONE_PONTE_SHP, ZONE_INTEREST_FIELD, CRS, DEPTH_THRESHOLD, START_AT_FIRST_OCCURENCE, FOCUS_ON_ZONE, NUMBER_OF_HABITATS
from variables_from_config import SHP_DEPTH_PREFIX, SHP_VEL_PREFIX, SHP_X_COLNAME, SHP_Y_COLNAME, SHP_ID_COLNAME, SHP_AREA_COLNAME
from variables_from_config import OUTPUT_FOLDER_TIME, SHP_INPUT_FILE, DATA_DIR_HYDRO, TYPICAL_FLOW_FILENAME
from crop_metric_and_map_preview import main as crop_and_map_preview

from metrics_calculation_focus_on_zone import process_mesh_data_focus_on_zone
from metrics_calculation import process_mesh_data
from support_functions import get_discharge_values, prepare_wetted_shapefile_for_relevant_discharges, join_mesh_with_CSV_data
from discharge_time_series import match_closest_discharge, get_study_discharges
from habitat_classification import prepare_csv, add_zone_flag_to_mesh, attribute_habitat_types_zone_only, attribute_habitat_current_based

# ==================================================================
# =================================== Main ========================
# ==================================================================
shp_file_data = gpd.read_file(SHP_INPUT_FILE) #initial shapefile having all the data for each mesh and each discharge. The area extent is based on the configuration

discharges_values = get_discharge_values(shp_file_data, SHP_DEPTH_PREFIX, SHP_VEL_PREFIX)

# match closest discharge
typical_flow_time_series_path = os.path.join(DATA_DIR_HYDRO, TYPICAL_FLOW_FILENAME) #changes automatically based on if it is the week,...
output_dir_discharge_with_match = os.path.join(OUTPUT_FOLDER_TIME, "Discharge_with_match")
os.makedirs(output_dir_discharge_with_match, exist_ok=True)
discharge_with_match = match_closest_discharge(discharges_values, typical_flow_time_series_path, output_dir_discharge_with_match)

# select relevant discharges 
relevant_discharges = get_study_discharges(discharge_with_match)
print("Discharges to study:", relevant_discharges)

# crop the shapefile: to the corresponding known discharges from the discharge time series; and keeping only the wetted mask (meshed wet for the max considered discharge)
output_dir_additional_shp = os.path.join(OUTPUT_FOLDER_TIME, "Cropped_shapefiles")
os.makedirs(output_dir_additional_shp, exist_ok=True)
output_prepared_shp = os.path.join(output_dir_additional_shp, "cropped_discharge_and_wetted_area.shp")

prepared_gdf = prepare_wetted_shapefile_for_relevant_discharges(
    gdf=shp_file_data,
    relevant_discharges=relevant_discharges,
    depth_prefix=SHP_DEPTH_PREFIX,
    velocity_prefix=SHP_VEL_PREFIX,
    depth_threshold=DEPTH_THRESHOLD,
    output_shp_path=output_prepared_shp,
    id_col_name=SHP_ID_COLNAME,
    area_col_name=SHP_AREA_COLNAME,
    x_col_name=SHP_X_COLNAME,
    y_col_name=SHP_Y_COLNAME
)

# ------------------------------------------------------------------------------------------------------------------
# Export the cropped shapefile to a csv 
output_csv = os.path.join(output_dir_additional_shp, "cropped_discharge_and_wetted_area.csv")
df_csv = prepare_csv(
    prepared_shp=prepared_gdf,
    output_csv=output_csv, 
    depth_prefix=SHP_DEPTH_PREFIX,
    vel_prefix=SHP_VEL_PREFIX
)
folder_suffix = os.path.basename(OUTPUT_FOLDER_TIME)  # e.g., "Short_small"

if FOCUS_ON_ZONE:
    output_dir_desiccation = os.path.join(OUTPUT_FOLDER_TIME, "Habitat_classification")
    os.makedirs(output_dir_desiccation, exist_ok=True)

    # Add focus zone flags to the meshes (1 if within the focus zone, 0 otherwise), saving the output as a df, not a csv since this in only an intermediary result
    df_with_zones = add_zone_flag_to_mesh(
        mesh_csv=output_csv,
        polygon_shp=ZONE_PONTE_SHP,
        zone_col=ZONE_INTEREST_FIELD,
        crs=CRS,
    )

    # Attribute habitat only for when is on the gravel bar (otherwise -1)
    output_csv_habitat_classification_focus_on_zone = os.path.join(output_dir_desiccation, "habitat_classification.csv")
    attribute_habitat_types_zone_only(
        mesh_df=df_with_zones,
        output_csv=output_csv_habitat_classification_focus_on_zone,
        zone_col=ZONE_INTEREST_FIELD,
        min_depth_threshold=DEPTH_THRESHOLD,
        max_depth_threshold = 0.30, #in m
        velocity_range=(0.05, 0.75) #in m/s
    )

    # Metrics calculation
    output_metrics = os.path.join(OUTPUT_FOLDER_TIME, "Metric_files")
    os.makedirs(output_metrics, exist_ok=True)
    output_metrics_values = os.path.join(output_metrics, "metrics.csv")

    process_mesh_data_focus_on_zone(discharge_with_match, output_csv_habitat_classification_focus_on_zone, output_metrics_values, DRIFT_THRESHOLDS_NO_RAMP, DESICCATION_THRESHOLDS, HABITAT_TARGETS, START_AT_FIRST_OCCURENCE)

    # Save the metrics to a shapefile 
    output_mesh_metrics_values = os.path.join(output_metrics, "metrics.shp")
    join_mesh_with_CSV_data(
        mesh_file=output_prepared_shp,
        csv_file=output_metrics_values,
        output_shp_file=output_mesh_metrics_values,
        id_col=SHP_ID_COLNAME
    )

else: 
    output_dir_habitat_classification = os.path.join(OUTPUT_FOLDER_TIME, "Habitat_classification")
    os.makedirs(output_dir_habitat_classification, exist_ok=True)
    output_habitat_attribution = os.path.join(output_dir_habitat_classification, "habitat_classification.csv")

    # Habitat attribution
    attribute_habitat_current_based(
        mesh_csv=output_csv,
        output_csv=output_habitat_attribution,
        min_depth_threshold=DEPTH_THRESHOLD,
        HABITAT_VELOCITY_THRESHOLDS=HABITAT_VELOCITY_THRESHOLDS,
        number_of_habitats = NUMBER_OF_HABITATS)
    
    # Metrics calculation
    output_metrics = os.path.join(OUTPUT_FOLDER_TIME, "Metric_files")
    os.makedirs(output_metrics, exist_ok=True)
    output_metrics_values = os.path.join(output_metrics, "metrics.csv")

    process_mesh_data(discharge_with_match, output_habitat_attribution, output_metrics_values, DRIFT_THRESHOLDS_NO_RAMP, DESICCATION_THRESHOLDS, HABITAT_TARGETS, START_AT_FIRST_OCCURENCE)

    # Save the metrics to a shapefile 
    output_mesh_metrics_values = os.path.join(output_metrics, "metrics.shp")
    join_mesh_with_CSV_data(
        mesh_file=output_prepared_shp,
        csv_file=output_metrics_values,
        output_shp_file=output_mesh_metrics_values,
        id_col=SHP_ID_COLNAME
    )

# ---- new step: crop to essentials + generate preview maps ----
crop_and_map_preview(output_mesh_metrics_values)