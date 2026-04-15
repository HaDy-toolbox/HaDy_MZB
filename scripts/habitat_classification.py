import geopandas as gpd
import pandas as pd


def prepare_csv(prepared_shp, output_csv, depth_prefix, vel_prefix):
    """
    Convert prepared GeoDataFrame to CSV with renamed depth/velocity columns.
    No habitat attribution is performed here.

    Parameters:
    - prepared_shp: GeoDataFrame with columns like 'ho<q>' and 'vit<q>'
    - output_csv: path to save the CSV file
    - depth_prefix: prefix for depth columns (e.g., 'ho')
    - vel_prefix: prefix for velocity columns (e.g., 'vit')

    Returns:
    - pandas DataFrame with renamed Depth_ and Vel_ columns
    """

    df = prepared_shp.copy()

    # Drop geometry
    if "geometry" in df.columns:
        df = df.drop(columns="geometry")

    # Identify depth & velocity columns
    depth_cols = [c for c in df.columns if c.startswith(depth_prefix)]
    vel_cols = [c for c in df.columns if c.startswith(vel_prefix)]

    # Rename columns
    rename_dict = {c: "Depth_" + c[len(depth_prefix):] for c in depth_cols}
    rename_dict.update({c: "Vel_" + c[len(vel_prefix):] for c in vel_cols})
    df = df.rename(columns=rename_dict)

    df.to_csv(output_csv, index=False)
    print(f"✅ CSV without habitat attribution saved to {output_csv}")

    return df

def add_zone_flag_to_mesh(
    mesh_csv: str,
    polygon_shp: str,
    # output_csv: str,
    zone_col: str,
    crs: str,
    predicate: str = "within"
):
    """
    Adds a binary zone flag to a mesh CSV based on a polygon shapefile.
    1 = inside polygon, 0 = outside
    """

    # Load mesh
    df = pd.read_csv(mesh_csv)

    gdf_mesh = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["x_l93"], df["y_l93"]),
        crs=crs
    )

    # Load polygon
    gdf_zone = gpd.read_file(polygon_shp).to_crs(crs)

    # Spatial join
    joined = gpd.sjoin(
        gdf_mesh,
        gdf_zone,
        how="left",
        predicate=predicate
    )

    # Create flag
    joined[zone_col] = joined["index_right"].notna().astype(int)

    # Clean & export
    joined = joined.drop(columns=["geometry", "index_right"])
    
    # previously as a csv file (but since not used in the analysis, then we just pass it as a dataframe instead of a csv)
    # joined.to_csv(output_csv, index=False)
    # print(f"🏷️ Zone flag '{zone_col}' added → {output_csv}")
    # return output_csv

    return joined #intead of a csv, we return the dataframe with the zone flag that will be used for habitat attribution

def attribute_habitat_types_zone_only(
    mesh_df,
    output_csv: str,
    zone_col: str,
    min_depth_threshold: float,
    max_depth_threshold: float,  # in m
    velocity_range: tuple        # in m/s
):
    """
    Attributes habitat types ONLY for cells inside the zone (zone_col == 1).

    Habitat definition (only applied where zone_col == 1):
    -1: outside the zone (zone_col == 0)
    0: depth < min_depth_threshold
    1: depth >= min_depth_threshold but unsuitable
    2: min_depth_threshold < depth <= max_depth_threshold AND
       velocity in velocity_range
    """

    if isinstance(mesh_df, str):
        df = pd.read_csv(mesh_df)   # backwards compatible if called with a path
    else:
        df = mesh_df.copy()

    # df = pd.read_csv(mesh_df)

    # Identify discharges
    discharges = [c[len("Vel_"):] for c in df.columns if c.startswith("Vel_")]
    vmin, vmax = velocity_range

    for q in discharges:
        depth_col = f"Depth_{q}"
        vel_col = f"Vel_{q}"
        hab_col = f"Hab_{q}"

        # Initialize all cells as -1 for outside zone
        df[hab_col] = -1

        depth = df[depth_col]
        velocity = df[vel_col]

        # Mask for inside zone
        inside_zone = df[zone_col] == 1

        # Masks for habitat attribution (only inside zone)
        wet = (depth >= min_depth_threshold) & inside_zone
        depth_ok = (depth > min_depth_threshold) & (depth <= max_depth_threshold) & inside_zone
        velocity_ok = (velocity >= vmin) & (velocity <= vmax) & inside_zone

        # Habitat 2 → optimal (inside zone only)
        df.loc[depth_ok & velocity_ok, hab_col] = 2

        # Habitat 1 → wetted but not optimal (inside zone only)
        df.loc[wet & ~(depth_ok & velocity_ok), hab_col] = 1

        # Habitat 0 automatically remains 0 where depth < min_depth_threshold inside the zone
        df.loc[(depth < min_depth_threshold) & inside_zone, hab_col] = 0

    df.to_csv(output_csv, index=False)
    print(f"✅ Zone-restricted habitat attribution completed → {output_csv}")

    return df

# function that takes as an input the cropped shp and renames the vel and depth columns, defines habitat type for each threshold and export as a .csv
def attribute_habitat_current_based(
    mesh_csv: str,
    output_csv: str,
    min_depth_threshold: float,
    HABITAT_VELOCITY_THRESHOLDS: dict
):
    """
    Attributes habitat classes based on velocity thresholds (current-based logic).

    Habitat definition per discharge:
    0 : dry cell (depth < min_depth_threshold)
     1–6 : velocity-based classes for wetted cells

    Velocity classes are defined using HABITAT_VELOCITY_THRESHOLDS:
        class_1, class_2, class_3, class_4, class_5
    """

    df = pd.read_csv(mesh_csv)

    # Extract the raw suffix (e.g. '4', '12_8') — no float conversion needed
    vel_prefix = "Vel_"
    q_suffixes = [c[len(vel_prefix):] for c in df.columns if c.startswith(vel_prefix)]

    for q_str in q_suffixes:                  # q_str is '4' or '12_8', etc.
        depth_col = f"Depth_{q_str}"
        vel_col   = f"Vel_{q_str}"
        hab_col   = f"Hab_{q_str}"

        depth_vals = df[depth_col]
        vel_vals = df[vel_col]

        # Initialize as 0 (dry by default)
        df[hab_col] = 0

        # Wetted mask
        wetted = depth_vals >= min_depth_threshold

        # Assign habitat classes based on velocity
        df.loc[wetted & (vel_vals < HABITAT_VELOCITY_THRESHOLDS["class_1"]), hab_col] = 1

        df.loc[wetted &
               (vel_vals >= HABITAT_VELOCITY_THRESHOLDS["class_1"]) &
               (vel_vals < HABITAT_VELOCITY_THRESHOLDS["class_2"]), hab_col] = 2

        df.loc[wetted &
               (vel_vals >= HABITAT_VELOCITY_THRESHOLDS["class_2"]) &
               (vel_vals < HABITAT_VELOCITY_THRESHOLDS["class_3"]), hab_col] = 3

        df.loc[wetted &
               (vel_vals >= HABITAT_VELOCITY_THRESHOLDS["class_3"]) &
               (vel_vals < HABITAT_VELOCITY_THRESHOLDS["class_4"]), hab_col] = 4

        df.loc[wetted &
               (vel_vals >= HABITAT_VELOCITY_THRESHOLDS["class_4"]) &
               (vel_vals < HABITAT_VELOCITY_THRESHOLDS["class_5"]), hab_col] = 5

        df.loc[wetted &
               (vel_vals >= HABITAT_VELOCITY_THRESHOLDS["class_5"]), hab_col] = 6

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"✅ CSV with velocity-based habitat classification saved to {output_csv}")

    return df