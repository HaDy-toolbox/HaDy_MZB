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

    # Identify discharges automatically from Vel_ columns
    discharges = [c[len("Vel_"):] for c in df.columns if c.startswith("Vel_")]

    for q in discharges:
        depth_col = f"Depth_{q}"
        vel_col = f"Vel_{q}"
        hab_col = f"Hab_{q}"

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