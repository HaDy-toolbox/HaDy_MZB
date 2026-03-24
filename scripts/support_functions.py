import geopandas as gpd
import pandas as pd

# ==================================================================
# =========== Gets discharge data from shp input file ==============
# ==================================================================
def get_discharge_values(shp_file_data, depth_prefix, velocity_prefix):
    """
    Extract discharge values from a GeoDataFrame based on depth/velocity column prefixes.

    Parameters:
    - shp_file_data: GeoDataFrame containing columns like 'ho6', 'vit6', etc.
    - depth_prefix: prefix for depth columns (default 'ho')
    - velocity_prefix: prefix for velocity columns (default 'vit')

    Returns:
    - discharges: sorted list of float discharge values
    - discharge_columns: dict mapping discharge -> {'depth': col_name, 'velocity': col_name}
    """
    # Find all columns that start with the depth prefix
    depth_cols = [col for col in shp_file_data.columns if col.startswith(depth_prefix)]
    
    # Extract the discharge numbers from the column names
    discharges = sorted([float(col.replace(depth_prefix, '')) for col in depth_cols])
    
    # Pair depth and velocity columns
    discharge_columns = {}
    for d in discharges:
        # Handles integers and floats in column names
        depth_col = f"{depth_prefix}{d}".replace('.0','')  # in case column is 'ho6' not 'ho6.0'
        velocity_col = f"{velocity_prefix}{d}".replace('.0','')
        discharge_columns[d] = {'depth': depth_col, 'velocity': velocity_col}
    
    # return discharges, discharge_columns
    return discharges



# ==================================================================
# ===== Filters the corresponding known discharges from the flow time-series 
# and then keeps only the meshes withing the wetted mask =====
# ==================================================================
#
def prepare_wetted_shapefile_for_relevant_discharges(
    gdf,
    relevant_discharges,
    depth_prefix,
    velocity_prefix,
    depth_threshold,
    output_shp_path,
    id_col_name,
    surf_col_name,
    x_col_name,
    y_col_name
):
    """
    Prepare a shapefile for analysis by:
    1) Keeping only ID, x, y, surface, geometry, and hoX/vitX columns for relevant discharges
    2) Replacing missing values with 0
    3) Keeping only meshes wetted at the maximum discharge

    Parameters:
    - gdf: input GeoDataFrame
    - relevant_discharges: list of float discharges to study
    - depth_prefix: prefix for depth columns (e.g. 'ho')
    - velocity_prefix: prefix for velocity columns (e.g. 'vit')
    - depth_threshold: minimum depth defining wetted area (m)
    - output_shp_path: output shapefile path
    - id_col_name, surf_col_name, x and y column names

    Returns:
    - filtered GeoDataFrame
    """

    # --------------------------------------------------
    # 1. Check mandatory columns
    # --------------------------------------------------
    required_cols = [id_col_name, surf_col_name, x_col_name, y_col_name]
    for col in required_cols:
        if col not in gdf.columns:
            raise ValueError(f"Column '{col}' not found in shapefile")

    # --------------------------------------------------
    # 2. Build hoX / vitX column list
    # --------------------------------------------------
    discharge_cols = []

    for q in relevant_discharges:
        q_str = str(int(q)) if float(q).is_integer() else str(q)
        # q_str = f"{q:g}"

        depth_col = f"{depth_prefix}{q_str}"
        vel_col = f"{velocity_prefix}{q_str}"

        if depth_col in gdf.columns:
            discharge_cols.append(depth_col)
        if vel_col in gdf.columns:
            discharge_cols.append(vel_col)

    if not discharge_cols:
        raise ValueError("No hoX / vitX columns found for relevant discharges")

    # --------------------------------------------------
    # 3. Subset columns
    # --------------------------------------------------
    columns_to_keep = (
        [id_col_name, surf_col_name, x_col_name, y_col_name]
        + discharge_cols
        + ['geometry']
    )

    out_gdf = gdf[columns_to_keep].copy()

    # --------------------------------------------------
    # 4. Replace missing values
    # --------------------------------------------------
    out_gdf = out_gdf.fillna(0)

    # --------------------------------------------------
    # 5. Filter wetted meshes at max discharge
    # --------------------------------------------------
    max_q = max(relevant_discharges)
    max_q_str = str(int(max_q)) if float(max_q).is_integer() else str(max_q)
    max_depth_col = f"{depth_prefix}{max_q_str}"

    if max_depth_col not in out_gdf.columns:
        raise ValueError(f"Depth column '{max_depth_col}' not found")

    out_gdf = out_gdf[out_gdf[max_depth_col] > depth_threshold].copy()

    # --------------------------------------------------
    # 6. Save output
    # --------------------------------------------------
    out_gdf.to_file(output_shp_path)

    print(
        f"✅ Prepared shapefile saved\n"
        f"   - Wetted at Qmax = {max_q} (depth > {depth_threshold})\n"
        f"   - Path: {output_shp_path}"
    )

    return out_gdf

# ==================================================================
# Exports a shp as a .csv
# ==================================================================
def export_shapefile_to_csv(shp_to_convert, output_csv_path):
    df = shp_to_convert.drop(columns='geometry').copy()
    df.to_csv(output_csv_path, index=False)

    print(f"✅ CSV exported to:\n{output_csv_path}")
    return df

# ==================================================================
# Converts the .csv back to a polygon shapefile
# ==================================================================
def join_mesh_with_CSV_data(mesh_file, csv_file, output_shp_file, id_col):
    """
    Join a GeoDataFrame (mesh geometry) with a results DataFrame and save to a new shapefile.

    Parameters:
    - mesh_file: str, path to the mesh shapefile (polygons or points)
    - csv_file: str, path to the CSV file with results
    - output_shp_file: str, path to save the merged shapefile
    - id_col: str, column name to join on (default "id")
    """
    mesh_gdf = gpd.read_file(mesh_file) # Load base mesh geometry
    results_df = pd.read_csv(csv_file) # Load results
    mesh_with_results = mesh_gdf.merge(results_df, on=id_col, how="left") # Merge on the specified ID column

    COLUMN_RENAME_MAP = { # Rename columns to respect shapefile 10-char limit
        "hab3_shift_all_daily": "h3_sh_all",
        "hab3_shift_targ_daily": "h3_sh_suit", 
        "hab3_shift_dry_daily": "h3_sh_dry",
        "hab3_DryMax": "h3_drymax",
        "hab3_DesicRisk": "h3_desic",
        "hab3_DriftPerc": "h3_driftP",
        "hab3_DriftMax": "h3_driftM",
        "hab3_dur_drift_1": "h3_dd_1",
        "hab3_dur_drift_2": "h3_dd_2",
        "hab3_dur_drift_3": "h3_dd_3",
        "hab3_dur_drift_4": "h3_dd_4",
        "prob_hab_-1": "h3_prob_-1",
        "hab3_first_occurrence_time": "hab3_first",

        "hab2_shift_all_daily": "h2_sh_all",
        "hab2_shift_targ_daily": "h2_sh_suit",
        "hab2_shift_dry_daily": "h2_sh_dry",
        "hab2_DryMax": "h2_drymax",
        "hab2_DesicRisk": "h2_desic",
        "hab2_DriftPerc": "h2_driftP",
        "hab2_DriftMax": "h2_driftM",
        "hab2_dur_drift_1": "h2_dd_1",
        "hab2_dur_drift_2": "h2_dd_2",
        "hab2_dur_drift_3": "h2_dd_3",
        "hab2_dur_drift_4": "h2_dd_4",
        "prob_hab_-1": "h2_prob_-1",
        "hab2_first_occurrence_time": "hab2_first"
    }

    # Apply only existing columns
    rename_existing = {
        old: new
        for old, new in COLUMN_RENAME_MAP.items()
        if old in mesh_with_results.columns
    }

    mesh_with_results = mesh_with_results.rename(columns=rename_existing)
    mesh_with_results.to_file(output_shp_file) # Save the merged GeoDataFrame to a shapefile
    print(f"✅ Merged shapefile saved to {output_shp_file}")

