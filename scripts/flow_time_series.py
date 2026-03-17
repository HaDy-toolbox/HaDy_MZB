import os
import pandas as pd

# ==================================================================
# =========== Flow time-series: match closest discharge ============
# ==================================================================
def match_closest_discharge(discharge_values, typical_flow_time_series_path, output_dir): #typical_flow_time_series_path can be the day or the week when the function gets executed
    """Match each timestep in typical week to closest available discharge."""
    # time_series_df = pd.read_excel(typical_flow_time_series_path)
    time_series_df = pd.read_csv(typical_flow_time_series_path)
    
    def find_closest_discharge(target):
        return min(discharge_values, key=lambda x: abs(x - target))

    time_series_df['Corresponding_known_discharge'] = time_series_df['Discharge'].apply(find_closest_discharge)

    output_week_csv = os.path.join(output_dir, "discharge_with_match.csv")
    time_series_df.to_csv(output_week_csv, index=False)

    print(f"✅ Updated week discharge file saved to: {output_week_csv}")
    return output_week_csv

# ==================================================================
# ========== Flow time-series discharge values to study ============
# ==================================================================
def get_study_discharges(discharge_csv_path):
    """
    Extract distinct discharge values that appear in 'Corresponding_known_discharge' column.

    Parameters:
    - discharge_csv_path: path to the CSV file with 'Corresponding_known_discharge' column

    Returns:
    - sorted list of unique discharges (as floats)
    """
    df = pd.read_csv(discharge_csv_path)
    
    # Ensure the column exists
    if 'Corresponding_known_discharge' not in df.columns:
        raise ValueError("CSV must have a column named 'Corresponding_known_discharge'")
    
    # Get known float discharge values that are in the flow time-series and that will be relevant for the next steps
    relevant_know_discharges = sorted(float(q) for q in df['Corresponding_known_discharge'].unique())
    
    return relevant_know_discharges