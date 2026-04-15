import numpy as np
import pandas as pd
import math
from itertools import groupby

from variables_from_config import HABITAT_TARGETS
from variables_from_config import TIME_STEP_MIN
from variables_from_config import TIMESTEPS_PER_DAY
from variables_from_config import ZONE_PONTE_FIELD
from variables_from_config import SHP_DEPTH_PREFIX, SHP_VEL_PREFIX, SHP_X_COLNAME, SHP_Y_COLNAME, SHP_ID_COLNAME, SHP_SURF_COLNAME 
from variables_from_config import METRICS_TO_COMPUTE
from variables_from_config import UP_RAMP, DRIFT_THRESHOLDS_WITH_RAMP
from variables_from_config import ID_POLYGON
# ===================== Metrics calculation ======================
# ================================================================

# helper functions
def format_col(prefix, discharge):
    """Formats column names consistently. 4 → 'Hab_4', 12.8 → 'Hab_12_8'"""
    if float(discharge).is_integer():
        q_str = str(int(discharge))
    else:
        q_str = str(discharge).replace('.', '_')   # ← just add this replace
    return f"{prefix}_{q_str}"

def max_run_length(seq, valid_vals):
    """Computes the longest continuous sequence of valid values."""
    return max((sum(1 for _ in group) for val, group in groupby(seq) if val in valid_vals), default=0)

# not daily 
def count_shifts(seq, target_vals):
    """Counts transitions into target habitat(s)."""
    flags = [val in target_vals for val in seq]
    return sum(1 for i in range(1, len(flags)) if flags[i] and not flags[i-1])

def count_shifts_daily(seq, target_vals, timesteps_per_day):
    """
    Counts transitions into target habitat(s) 
    and returns the average number of shifts per day.
    """

    if not seq or len(seq) < 2:
        return np.nan

    n_days = len(seq) / timesteps_per_day

    if n_days > 0:
        total_shifts = sum(
            1 for i in range(1, len(seq))
            if seq[i] in target_vals and seq[i-1] not in target_vals
        )
        return round(total_shifts / n_days, 2)
    else:
        return np.nan

def count_within_group_shifts(seq, target_vals, timesteps_per_day):
    """Counts shifts *within* the same target habitat group."""
    n_days = len(seq) / timesteps_per_day

    if n_days > 0:
        # return int(round(total_shifts / n_days))
        return round(sum(
        1 for i in range(1, len(seq))
        if seq[i] in target_vals and seq[i-1] in target_vals and seq[i] != seq[i-1])/ n_days, 2) #to try to have decimals 
    else:
        return np.nan

def shift_all(seq):
    """Counts all habitat changes."""
    return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])

def compute_daily_shifts(habitat_seq, timesteps_per_day):
    """
    Compute the daily shift rate (number of habitat shifts per day) as a float having 2 decimals.

    Parameters
    ----------
    habitat_seq : list or array
        Time series of habitat types (integers, e.g. 0–6)
    timesteps_per_day : int
        Number of timesteps in one day (e.g. 144 for 10-min data)

    Returns
    -------
    round
        Average number of shifts per day (float having 2 decimals)
    """
    if not habitat_seq or len(habitat_seq) < 2:
        return np.nan

    total_shifts = shift_all(habitat_seq)
    n_days = len(habitat_seq) / timesteps_per_day
    if n_days > 0:
        return round(total_shifts / n_days, 2) #to have decimals 
    else:
        return np.nan

def max_dry_duration(seq):
    """Returns the maximum consecutive dry (0) timesteps. Here k is the value of the group --> we focus on k=0 here; and g is just an iterator finding how many values are in the group"""
    return max((len(list(g)) for k, g in groupby(seq) if k == 0), default=0)

def get_desiccation_risk(max_dry_blocks, desiccation_thresholds):
    """Classifies desiccation risk based on dry duration (in TIME_STEP_MIN timesteps)."""
    hours = max_dry_blocks * TIME_STEP_MIN / 60  # Convert 10-min timesteps to hours
    if hours <= desiccation_thresholds["desicc_1"]:
        return 1
    elif hours <= desiccation_thresholds["desicc_2"]:
        return 2
    elif hours <= desiccation_thresholds["desicc_3"]:
        return 3
    else:
        return 4

def get_desiccation_risk_from_hours(hours, desiccation_thresholds):
    """Classifies desiccation risk based on dry duration already expressed in hours."""
    if np.isnan(hours):
        return np.nan
    if hours <= desiccation_thresholds["desicc_1"]:
        return 1
    elif hours <= desiccation_thresholds["desicc_2"]:
        return 2
    elif hours <= desiccation_thresholds["desicc_3"]:
        return 3
    else:
        return 4
    
def compute_habitat_seq_stats(seq, target_val, time_step_min):
    """
    Computes statistics over individual sequences of a target habitat type.

    A sequence is a contiguous run of `target_val` in the sequence.
    Durations are expressed in hours.

    Parameters
    ----------
    seq : list
        Habitat time series.
    target_val : int
        The habitat value to analyse (e.g. 2 for target, 0 for dry).
    time_step_min : int or float
        Duration of one timestep in minutes (e.g. 10).

    Returns
    -------
    dict with keys:
        nb_seq           : int   — number of distinct sequences
        dur_max          : float — longest single sequence duration (hours)
        dur_med       : float — median sequence duration (hours)
        dur_q1           : float — 1st quartile (hours)
        dur_q3           : float — 3rd quartile (hours)
    All float values are np.nan when no seq exists.
    """
    durations_timesteps = [
        sum(1 for _ in group)
        for val, group in groupby(seq)
        if val == target_val
    ]

    n = len(durations_timesteps)

    if n == 0:
        return {
            "nb_seq": 0,
            "dur_max":        np.nan,
            "dur_med":        np.nan,
            "dur_q1":         np.nan,
            "dur_q3":         np.nan,
        }

    durations_hours = [d * time_step_min / 60 for d in durations_timesteps]

    return {
        "nb_seq": n,
        "dur_max":     round(float(max(durations_hours)), 2),
        "dur_med":     round(float(np.median(durations_hours)), 2),
        "dur_q1":      round(float(np.percentile(durations_hours, 25)), 2),
        "dur_q3":      round(float(np.percentile(durations_hours, 75)), 2),
    }

def compute_desiccation_seq_stats(seq, time_step_min):
    """Convenience wrapper: dry sequence stats (habitat 0)."""
    stats = compute_habitat_seq_stats(seq, target_val=0, time_step_min=time_step_min)
    return {
        "dry_seq":            stats["nb_seq"],
        "dry_max":            stats["dur_max"],
        "dry_med":            stats["dur_med"],
        "dry_q1":             stats["dur_q1"],
        "dry_q3":             stats["dur_q3"],
    }

def compute_drift_risk(vel_seq, depth_seq, drift_thresholds):
    """
    Computes drift risk per timestep using velocity and up-ramping rate.
    """
    drift = []

    for j in range(len(vel_seq)):
        vel = vel_seq[j]
        if j > 0:
            delta_depth = depth_seq[j] - depth_seq[j - 1] #now - previously. we only have a look at upramping here so sign does not matter
            ramp_rate_cm_min = (delta_depth / TIME_STEP_MIN) * 100  # cm/min over 10 min
        else:
            ramp_rate_cm_min = 0

        if np.isnan(vel):
            drift.append(np.nan)
        elif vel <= drift_thresholds["drift_1"]:
            drift.append(1)
        elif drift_thresholds["drift_1"] < vel < drift_thresholds["drift_2"]:
            # drift.append(20)  # drift class 2 (type 0)
            drift.append(2)  # drift class 2 (type 0)
        elif drift_thresholds["drift_2"] <= vel <= drift_thresholds["drift_3"]:
            # drift.append(21 if ramp_rate_cm_min <= drift_thresholds["drift_ramp_threshold_2"] else 3)
            drift.append(3)  # drift class 2 (type 0)
        elif vel > drift_thresholds["drift_3"]:
            drift.append(4)
        else:
            drift.append(np.nan)
    return drift

def compute_drift_risk_with_up_ramping(vel_seq, depth_seq, drift_thresholds, ramp_rate_cm_min):
    """
    Computes drift risk per timestep using velocity and up-ramping rate.
    """
    drift = []

    for j in range(len(vel_seq)):
        vel = vel_seq[j]
        if j > 0:
            delta_depth = depth_seq[j] - depth_seq[j - 1]
            ramp_rate_cm_min = (delta_depth / TIME_STEP_MIN) * 100
        else:
            ramp_rate_cm_min = 0

        if np.isnan(vel):
            drift.append(np.nan)
        elif vel <= drift_thresholds["drift_1"]:
            drift.append(1)
        elif drift_thresholds["drift_1"] < vel < drift_thresholds["drift_2"]:
            drift.append(20)
        elif drift_thresholds["drift_2"] <= vel <= drift_thresholds["drift_3"]:
            drift.append(21 if ramp_rate_cm_min <= drift_thresholds["drift_ramp_threshold"] else 3)
        elif vel > drift_thresholds["drift_3"]:
            drift.append(4)
        else:
            drift.append(np.nan)
    return drift

def compute_drift_percentile_class(drift_series, percentile=90):
    """Returns drift risk percentile class (1–4)."""
    valid = [r for r in drift_series if r in [1, 2, 3, 4, 20, 21]] #I keep the 2 for wen no data on ramping rate: it is just 2 
    if not valid:
        return np.nan
    valid_mapped = [2 if r in [20, 21] else r for r in valid]
    perc_value = np.percentile(valid_mapped, percentile)
    return int(min(max(math.ceil(perc_value), 1), 4))

def compute_habitat_durations(hab_seq):
    """Computes durations for each habitat and returns both counts and total duration."""
    durations = {f"dur_h_{h}": hab_seq.count(h) for h in range(-1, 3)}
    total_duration = len(hab_seq)
    return durations, total_duration

def compute_habitat_probabilities(durations, total_duration):
    """Computes probabilities from durations."""
    if total_duration == 0:
        return {f"prob_h_{h}": np.nan for h in range(-1, 3)}
    return {f"prob_h_{h}": durations[f"dur_h_{h}"] / total_duration for h in range(-1, 3)}

def get_most_probable_habitat(row, prob_prefix='prob_h_'):
    """Returns the habitat with the highest probability."""
    habitat_probs = {int(col.replace(prob_prefix, '')): row[col]
                     for col in row.index if col.startswith(prob_prefix) and pd.notna(row[col])}
    if not habitat_probs:
        return np.nan
    return max(habitat_probs, key=habitat_probs.get)

def compute_habitat_metrics(
    hab_seq,
    vel_seq,
    dep_seq,
    drift_thresholds,
    desiccation_thresholds,
    metrics_to_compute
):

    results = {}
    # -------------------------
    # Target habitat stats
    stats_target = compute_habitat_seq_stats(hab_seq, target_val=HABITAT_TARGETS[0], time_step_min=TIME_STEP_MIN)
    results["nb_seq"] = stats_target["nb_seq"]
    results["dur_max"]  = stats_target["dur_max"]
    results["dur_med"]     = stats_target["dur_med"]
    results["dur_q1"]         = stats_target["dur_q1"]
    results["dur_q3"]         = stats_target["dur_q3"]

    # -------------------------
    # SHIFTS
    # -------------------------
    if metrics_to_compute.get("shift_all_daily", False):
        results["sh_all"] = compute_daily_shifts(
            hab_seq, TIMESTEPS_PER_DAY
        )

    if metrics_to_compute.get("shift_targ_daily", False):
        results["sh_suit"] = count_shifts_daily(
            hab_seq, [2], TIMESTEPS_PER_DAY
        )

    if metrics_to_compute.get("shift_dry_daily", False):
        results["sh_dry"] = count_shifts_daily(
            hab_seq, [0], TIMESTEPS_PER_DAY
        )

    # -------------------------
    # DESICCATION
    # -------------------------
    if metrics_to_compute.get("dry_max", False) or \
       metrics_to_compute.get("desiccation_risk", False):

        max_dry = max_dry_duration(hab_seq)

        if metrics_to_compute.get("dry_max", False):
            results["dryMax"] = max_dry

        if metrics_to_compute.get("desiccation_risk", False):
            results["desicR"] = get_desiccation_risk(
                max_dry,
                desiccation_thresholds
            )
        stats = compute_desiccation_seq_stats(hab_seq, TIME_STEP_MIN)
        results["dry_seq"]            = stats["dry_seq"]
        results["dry_max"]            = stats["dry_max"]
        results["dry_med"]            = stats["dry_med"]
        results["dry_q1"]             = stats["dry_q1"]
        results["dry_q3"]             = stats["dry_q3"]

        results["dryMedR"] = get_desiccation_risk_from_hours(
            stats["dry_med"], desiccation_thresholds
        )
        results["dry_q1R"] = get_desiccation_risk_from_hours(
            stats["dry_q1"], desiccation_thresholds
        )
        results["dry_q3R"] = get_desiccation_risk_from_hours(
            stats["dry_q3"], desiccation_thresholds
        )
    # -------------------------
    # DRIFT
    # -------------------------
    if (
        metrics_to_compute.get("drift_percentile", False)
        or metrics_to_compute.get("drift_max", False)
        or metrics_to_compute.get("drift_durations", False)
    ):
        if UP_RAMP:
            drift_series = compute_drift_risk_with_up_ramping(
                vel_seq,
                dep_seq,
                DRIFT_THRESHOLDS_WITH_RAMP,
                ramp_rate_cm_min=DRIFT_THRESHOLDS_WITH_RAMP["drift_ramp_threshold"]
            )
            if metrics_to_compute.get("drift_durations", False):
                mapped_drift_series = [2 if d in (20, 21) else d for d in drift_series]
                drift_durations = {
                    f"dd_{d}": mapped_drift_series.count(d)
                    for d in [1, 2, 3, 4]
                }
                results.update(drift_durations)

        else:
            drift_series = compute_drift_risk(
                vel_seq,
                dep_seq,
                drift_thresholds
            )
            if metrics_to_compute.get("drift_durations", False):
                drift_durations = {
                    f"dd_{d}": drift_series.count(d)
                    for d in [1, 2, 3, 4]
                }
                results.update(drift_durations)

        if metrics_to_compute.get("drift_percentile", False):
            results["driftP"] = compute_drift_percentile_class(drift_series)

        if metrics_to_compute.get("drift_max", False):
            mapped = [2 if d in (20, 21) else d for d in drift_series if not np.isnan(d)]
            results["driftM"] = max(mapped, default=np.nan)

    return results

def process_mesh_data_focus_on_zone(flow_csv, mesh_csv, output_csv, drift_thresholds, desiccation_thresholds, target_habitat, start_at_first_occurrence):
    flow_data = pd.read_csv(flow_csv)
    mesh_data = pd.read_csv(mesh_csv)
    discharge_values = flow_data["Corresponding_known_discharge"].values #this gives an array (line) with all the sequence of discharge values. Length = time of timesteps, order = temporal sequence
    print(f"🕐⌛ Metrics are being calculated")
    hab_cols = [format_col("Hab", d) for d in discharge_values] #building up something like hab_cols = ["Hab_3_1", "Hab_3_1", "Hab_3_2"]
    vel_cols = [format_col("Vel", d) for d in discharge_values] #building up something like vel_cols = ["Vel_3_1", "Vel_3_1", "Vel_3_2"]
    dep_cols = [format_col("Depth", d) for d in discharge_values] #building up something like dep_cols = ["Depth_3_1", "Depth_3_1", "Depth_3_2"]

    missing = [c for c in hab_cols if c not in mesh_data.columns]
    if missing:
        raise ValueError(
            f"❌ Missing habitat columns in mesh CSV (example): {missing[:5]}"
        )

    # computing the metrics for each mesh cell (a row)
    results_process_mesh_data = [
        compute_mesh_metrics_for_row_boolean(row, hab_cols, vel_cols, dep_cols, drift_thresholds, desiccation_thresholds, target_habitat, start_at_first_occurrence)
        for _, row in mesh_data.iterrows()
    ]

    df = pd.DataFrame(results_process_mesh_data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Metrics saved to {output_csv}")


# metrics calculation since the first occurence of one habitat type for each patch (with a boolean so this is optional)
def compute_mesh_metrics_for_row_boolean(
    row,
    hab_cols,
    vel_cols,
    dep_cols,
    drift_thresholds,
    desiccation_thresholds,
    target_habitat,
    start_at_first_occurrence
):
    """
    Compute metrics for one mesh element.

    Parameters
    ----------
    start_at_first_occurrence : bool
        If True, metrics are computed only from the first timestep
        where the target habitat appears.
    """

    results = {
        "id": row[SHP_ID_COLNAME],
        "x": row[SHP_X_COLNAME],
        "y": row[SHP_Y_COLNAME],
        "surface": row[SHP_SURF_COLNAME],
        "focus_zone": row.get(ZONE_PONTE_FIELD, 0),
        "id_focus": row.get(ID_POLYGON, np.nan) 
    }

    # Build full sequences
    habitat_seq = [row.get(c, np.nan) for c in hab_cols]
    velocity_seq = [row.get(c, np.nan) for c in vel_cols]
    depth_seq = [row.get(c, np.nan) for c in dep_cols]

    # ---------------------------------------
    # Global (full-sequence) metrics
    # ---------------------------------------
    durations, total_duration = compute_habitat_durations(habitat_seq)
    probs = compute_habitat_probabilities(durations, total_duration)

    results.update(durations)
    results.update(probs)
    results["mostProb"] = get_most_probable_habitat(pd.Series(probs))

    # ---------------------------------------
    # Target habitat metrics
    # ---------------------------------------
    for hab in target_habitat:

        prefix = f"h{hab}_"

        if hab in habitat_seq:

            # 1️⃣ Find first occurrence
            first_idx = next(i for i, v in enumerate(habitat_seq) if v == hab)

            # Save timestep index
            results[prefix + "first"] = first_idx #"first_occurrence_time"

            """
            # 2️⃣ Slice if requested
            if start_at_first_occurrence:
                habitat_seq_used = habitat_seq[first_idx:]
                velocity_seq_used = velocity_seq[first_idx:]
                depth_seq_used = depth_seq[first_idx:]
            else:
                habitat_seq_used = habitat_seq
                velocity_seq_used = velocity_seq
                depth_seq_used = depth_seq

            # 3️⃣ Compute metrics
            metrics = compute_habitat_metrics(
                habitat_seq_used,
                velocity_seq_used,
                depth_seq_used,
                drift_thresholds,
                desiccation_thresholds,
                METRICS_TO_COMPUTE
            )
            """
            # --------------------------------------------------
            # 1️⃣ Compute ALL metrics on full sequence
            # --------------------------------------------------
            metrics_full = compute_habitat_metrics(
                habitat_seq,
                velocity_seq,
                depth_seq,
                drift_thresholds,
                desiccation_thresholds,
                METRICS_TO_COMPUTE
            )

            metrics = metrics_full.copy()

            # --------------------------------------------------
            # 2️⃣ If requested → recompute ONLY desiccation
            #     from first occurrence onward
            # --------------------------------------------------
            if start_at_first_occurrence:

                habitat_seq_sliced = habitat_seq[first_idx:]

                # recompute dry max on sliced sequence
                max_dry = max_dry_duration(habitat_seq_sliced)

                if METRICS_TO_COMPUTE.get("dry_max", False):
                    metrics["dryMax"] = max_dry

                if METRICS_TO_COMPUTE.get("desiccation_risk", False):
                    metrics["desicR"] = get_desiccation_risk(
                        max_dry,
                        desiccation_thresholds
                    )
                    stats = compute_desiccation_seq_stats(
                        habitat_seq_sliced, TIME_STEP_MIN
                    )
                    metrics["dry_seq"]        = stats["dry_seq"]
                    metrics["dry_max"]        = stats["dry_max"]
                    metrics["dry_med"]        = stats["dry_med"]
                    metrics["dry_q1"]         = stats["dry_q1"]
                    metrics["dry_q3"]         = stats["dry_q3"]

                    metrics["dryMedR"] = get_desiccation_risk_from_hours(
                        stats["dry_med"], desiccation_thresholds
                    )
                    metrics["dry_q1R"] = get_desiccation_risk_from_hours(
                        stats["dry_q1"], desiccation_thresholds
                    )
                    metrics["dry_q3R"] = get_desiccation_risk_from_hours(
                        stats["dry_q3"], desiccation_thresholds
                    )

            results.update({prefix + k: v for k, v in metrics.items()})

        else:
            # Habitat never occurs
            results[prefix + "first"] = np.nan

            # Build empty metric structure using config
            empty_metrics = compute_habitat_metrics([], [], [], drift_thresholds, desiccation_thresholds, METRICS_TO_COMPUTE)
            results.update({prefix + k: np.nan for k in empty_metrics.keys()})

    return results
