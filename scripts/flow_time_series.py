import os
import pandas as pd

# ==================================================================
# =========== Flow time-series: match closest discharge ============
# ==================================================================
def match_closest_discharge(discharge_values, typical_flow_time_series_path, output_dir):
    """Match each timestep in a flow time series to the closest available discharge.
    
    Parameters:
    - discharge_values: list of float discharge values (from get_discharge_values)
    - typical_flow_time_series_path: path to CSV with a 'Discharge' column
    - output_dir: directory to save the output CSV
    
    Returns:
    - path to the output CSV
    """
    time_series_df = pd.read_csv(typical_flow_time_series_path)

    if 'Discharge' not in time_series_df.columns:
        raise ValueError("Input CSV must have a 'Discharge' column")

    def find_closest_discharge(target):
        return min(discharge_values, key=lambda x: abs(x - target))

    # Stored as float — no col-string encoding here
    time_series_df['Corresponding_known_discharge'] = (
        time_series_df['Discharge'].apply(find_closest_discharge)
    )

    output_csv = os.path.join(output_dir, "discharge_with_match.csv")
    time_series_df.to_csv(output_csv, index=False)
    print(f"✅ Matched discharge file saved to: {output_csv}")

    return output_csv


def get_study_discharges(discharge_csv_path):
    """
    Extract distinct discharge values from 'Corresponding_known_discharge' column.

    Parameters:
    - discharge_csv_path: path to the CSV produced by match_closest_discharge

    Returns:
    - sorted list of unique discharge floats
    """
    df = pd.read_csv(discharge_csv_path)

    if 'Corresponding_known_discharge' not in df.columns:
        raise ValueError("CSV must have a 'Corresponding_known_discharge' column")

    # Values are already floats in the CSV — no col_str_to_discharge needed
    relevant_discharges = sorted(df['Corresponding_known_discharge'].dropna().unique().tolist())

    return relevant_discharges