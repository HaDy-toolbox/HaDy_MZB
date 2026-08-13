"""
Habitat Classification
------------------------

Converts prepared mesh geometry/hydraulic data into a habitat-typed CSV,
using one of two classification strategies depending on FOCUS_ON_ZONE.

Functions:
1. prepare_csv(prepared_shp, output_csv, depth_prefix, vel_prefix)
   - Drops geometry from the prepared GeoDataFrame and renames raw depth/
     velocity columns (e.g. 'ho3_1' -> 'Depth_3_1', 'vit3_1' -> 'Vel_3_1').
   - Saves the result as a plain CSV (no habitat attribution yet).

2. add_zone_flag_to_mesh(mesh_csv, polygon_shp, zone_col, crs, predicate) --> when there is a focus on a specific zone (FOCUS_ON_ZONE = True)
   - Turns the mesh CSV into points (using x/y columns) and spatially joins
     it against a polygon shapefile (e.g. a gravel-bar zone of interest).
   - Adds a binary column (zone_col): 1 if the mesh cell falls inside the
     polygon(s), 0 otherwise. Returns a DataFrame (not saved to disk, since
     it's only an intermediate step before habitat attribution).

3. attribute_habitat_types_zone_only(mesh_df, output_csv, zone_col,
   min_depth_threshold, max_depth_threshold, velocity_range) --> when there is a focus on a specific zone (FOCUS_ON_ZONE = True)
   - Used when FOCUS_ON_ZONE is True. For every discharge, assigns each
     cell a habitat class, but ONLY for cells inside the zone:
       -1 = outside the zone
        0 = inside zone but dry (depth < min_depth_threshold)
        1 = inside zone, wet, but not meeting depth/velocity criteria
        2 = inside zone, wet, and within the suitable depth/velocity window
     Saves the result to output_csv.

4. attribute_habitat_current_based(mesh_csv, output_csv, min_depth_threshold,
   HABITAT_VELOCITY_THRESHOLDS, number_of_habitats) --> when there is no focus on a specific zone (FOCUS_ON_ZONE = False)
   - Used when FOCUS_ON_ZONE is False. For every discharge, assigns each
     cell a habitat class purely from current velocity, based on a sorted
     set of user-defined thresholds (class_1, class_2, ...):
        0 = dry, 1..N = increasing velocity classes.
   - Infers/validates the total number of habitat classes from the
     threshold dictionary, then saves the classified CSV.

Both attribution functions add one 'Hab_<discharge>' column per simulated
discharge to the dataframe.
"""

import geopandas as gpd
import pandas as pd
import re 

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

# FUNCTION TO ADAPT TO CHANGE THE HABITAT CLASSIFICATION METHOD WHEN THE BOOLEAN FOCUS_ON_ZONE IS TRUE
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

# FUNCTION TO ADAPT TO CHANGE THE HABITAT CLASSIFICATION METHOD WHEN THE BOOLEAN FOCUS_ON_ZONE IS FALSE (CURRENT-BASED CLASSIFICATION)
def attribute_habitat_current_based(
    mesh_csv: str,
    output_csv: str,
    min_depth_threshold: float,
    HABITAT_VELOCITY_THRESHOLDS: dict,
    number_of_habitats: int
    ):
    """
    Attributes habitat classes based on velocity thresholds (current-based logic).

    Habitat definition per discharge:
    0 : dry cell (depth < min_depth_threshold)
    1..N-1 : velocity-based classes, bounded by sorted thresholds in
             HABITAT_VELOCITY_THRESHOLDS (class_1, class_2, ..., class_{N-1})
    N : wetted cell with velocity >= last threshold

    The number of habitat classes is inferred from the number of
    class_X keys in HABITAT_VELOCITY_THRESHOLDS (N = len(thresholds) + 1).
    If number_of_habitats is provided, it is checked against this inferred
    value for consistency.
    """

    df = pd.read_csv(mesh_csv)

    # Sort thresholds by their class index (class_1, class_2, ...) to guarantee order
    sorted_items = sorted(
        HABITAT_VELOCITY_THRESHOLDS.items(),
        key=lambda kv: int(re.search(r"\d+", kv[0]).group())
    )
    thresholds = [v for _, v in sorted_items]  # e.g. [0.05, 0.25, 0.75, 1.5, 2.5]

    inferred_n_habitats = len(thresholds) + 1  # +1 for the "above last threshold" class
    # (habitat 0 = dry is separate from this count of wetted classes)
    total_habitats = inferred_n_habitats + 1   # +1 again to include dry (0)

    if number_of_habitats is not None and number_of_habitats != total_habitats:
        raise ValueError(
            f"number_of_habitats ({number_of_habitats}) does not match thresholds "
            f"provided ({total_habitats} inferred from HABITAT_VELOCITY_THRESHOLDS)."
        )

    vel_prefix = "Vel_"
    q_suffixes = [c[len(vel_prefix):] for c in df.columns if c.startswith(vel_prefix)]

    for q_str in q_suffixes:
        depth_col = f"Depth_{q_str}"
        vel_col   = f"Vel_{q_str}"
        hab_col   = f"Hab_{q_str}"

        depth_vals = df[depth_col]
        vel_vals = df[vel_col]

        # Initialize as 0 (dry by default)
        df[hab_col] = 0

        wetted = depth_vals >= min_depth_threshold

        # Class 1: v < thresholds[0]
        df.loc[wetted & (vel_vals < thresholds[0]), hab_col] = 1

        # Middle classes: thresholds[i-1] <= v < thresholds[i]
        for i in range(1, len(thresholds)):
            lower = thresholds[i - 1]
            upper = thresholds[i]
            df.loc[
                wetted & (vel_vals >= lower) & (vel_vals < upper),
                hab_col
            ] = i + 1

        # Last class: v >= thresholds[-1]
        df.loc[wetted & (vel_vals >= thresholds[-1]), hab_col] = len(thresholds) + 1

    df.to_csv(output_csv, index=False)
    print(f"✅ CSV with velocity-based habitat classification saved to {output_csv}")

    return df

