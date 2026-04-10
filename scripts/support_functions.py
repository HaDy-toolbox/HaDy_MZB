import geopandas as gpd
import pandas as pd

def discharge_to_col_str(q: float) -> str:
    """Convert a discharge value to its column-name string representation.
    4 → '4', 12.8 → '12_8'
    """
    q_str = str(int(q)) if float(q).is_integer() else str(q)
    return q_str.replace('.', '_')

def col_str_to_discharge(s: str) -> float:
    """Convert a column-name string back to a float discharge value.
    '4' → 4.0, '12_8' → 12.8
    """
    return float(s.replace('_', '.'))

# ==================================================================
# =========== Gets discharge data from shp input file ==============
# ==================================================================
# old version when discharge values in column names had decimals (e.g., 'ho12.8')

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
"""
def get_discharge_values(shp_file_data, depth_prefix, velocity_prefix):
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
"""

def get_discharge_values(shp_file_data, depth_prefix, velocity_prefix):
    depth_cols = [col for col in shp_file_data.columns if col.startswith(depth_prefix)]

    discharges = sorted([
        col_str_to_discharge(col[len(depth_prefix):])   # strip prefix, then convert
        for col in depth_cols
    ])

    discharge_columns = {}
    for q in discharges:
        q_str = discharge_to_col_str(q)
        discharge_columns[q] = {
            'depth':    f"{depth_prefix}{q_str}",
            'velocity': f"{velocity_prefix}{q_str}",
        }

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
        # q_str = str(int(q)) if float(q).is_integer() else str(q) # old version when discharge values in column names had decimals (e.g., 'ho12.8')
        # q_str = f"{q:g}"
        q_str = discharge_to_col_str(q)

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
    # max_q_str = str(int(max_q)) if float(max_q).is_integer() else str(max_q) # old version when discharge values in column names had decimals (e.g., 'ho12.8')
    max_q_str = discharge_to_col_str(max_q)
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

    COLUMN_RENAME_MAP = {}
    for h in range(10):  # habitat types 0 to 9
        COLUMN_RENAME_MAP.update({
            f"hab{h}_shift_all_daily":        f"h{h}_sh_all",
            f"hab{h}_shift_targ_daily":       f"h{h}_sh_suit",
            f"hab{h}_shift_dry_daily":        f"h{h}_sh_dry",
            f"hab{h}_DryMax":                 f"h{h}_dryMax",
            f"hab{h}_DesicRisk":              f"h{h}_desicR",
            f"hab{h}_DriftPerc":              f"h{h}_driftP",
            f"hab{h}_DriftMax":               f"h{h}_driftM",
            f"hab{h}_dur_drift_1":            f"h{h}_dd_1",
            f"hab{h}_dur_drift_2":            f"h{h}_dd_2",
            f"hab{h}_dur_drift_3":            f"h{h}_dd_3",
            f"hab{h}_dur_drift_4":            f"h{h}_dd_4",
            "prob_hab_-1":                    f"h{h}_prob_-1",
            f"hab{h}_first_occurrence_time":  f"h{h}_first",
            f"hab{h}_seq_count":              f"h{h}_nb_seq",
            f"hab{h}_max_cumul_dur":          f"h{h}_dur_max",
            f"hab{h}_median_dur":             f"h{h}_dur_med",
            f"hab{h}_dur_q1":                 f"h{h}_dur_q1",
            f"hab{h}_dur_q3":                 f"h{h}_dur_q3",
            f"hab{h}_dry_seq_count":          f"h{h}_dry_seq",
            f"hab{h}_dry_max_cumul_dur":      f"h{h}_dry_max",
            f"hab{h}_dry_median_dur":         f"h{h}_dry_med",
            f"hab{h}_dry_q1_dur":             f"h{h}_dry_q1",
            f"hab{h}_dry_q3_dur":             f"h{h}_dry_q3",
            f"hab{h}_desicRisk_median":       f"h{h}_dryMedR",
            f"hab{h}_desicRisk_q1":           f"h{h}_dry_q1R",
            f"hab{h}_desicRisk_q3":           f"h{h}_dry_q3R",
        })
    # Fields not habitat-specific
    COLUMN_RENAME_MAP.update({
        "desicRisk_median": "desicMedR",
        "desicRisk_q1":     "desic_q1R",
        "desicRisk_q3":     "desic_q3R",
    })

    # Apply only existing columns
    rename_existing = {
        old: new
        for old, new in COLUMN_RENAME_MAP.items()
        if old in mesh_with_results.columns
    }

    mesh_with_results = mesh_with_results.rename(columns=rename_existing)
    mesh_with_results.to_file(output_shp_file) # Save the merged GeoDataFrame to a shapefile
    print(f"✅ Merged shapefile saved to {output_shp_file}")

