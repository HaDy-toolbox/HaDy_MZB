"""
Creation of metrics_essentials.shp and preview maps
-------------------------

Step 1: Crops metrics.shp down to metrics_essentials.shp, keeping only:
  - id, x, y, area
  - all habitat probability columns (pattern: prob_h_{N})
  - mostProb
  - conditionally, depending on config flags, for HABITAT_TARGETS:
      * shift_col  = h{HABITAT_TARGETS}_sh_suit   (if shift_targ_daily: true)
      * drift_col  = h{HABITAT_TARGETS}_driftP    (if drift_percentile: true)
      * desic_col  = h{HABITAT_TARGETS}_desicR    (if desiccation_risk: true)

Step 2: Generates map previews (spatial plots) of those same metrics from
the just-created metrics_essentials data — habitat probability, shifts,
drift, and desiccation. Only meshes where the target habitat occurs
(probability > 0) are coloured; everything else is shown in grey
("No data / not applicable").

Both outputs are written next to FULL_METRICS_SHP in data > output > Metric_files:
  - metrics_essentials.shp                     (the cropped shapefile)
  - Preview_maps/map_probability_h{N}.png       (and shifts/drift/desiccation)
  - Preview_maps/map_shifts_h{N}.png
  - Preview_maps/map_drift_h{N}.png
  - Preview_maps/map_desiccation_h{N}.png

Requires: geopandas, numpy, pandas, matplotlib
    pip install geopandas numpy pandas matplotlib
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from variables_from_config import FINAL_SHP_PATH, HABITAT_TARGETS, METRICS_TO_COMPUTE
from variables_from_config import SHP_X_COLNAME, SHP_Y_COLNAME, SHP_ID_COLNAME, SHP_AREA_COLNAME

# ---------------------------------------------------------------------------
# Configuration — 
# ---------------------------------------------------------------------------
FULL_METRICS_SHP = FINAL_SHP_PATH       # input shapefile
OUTPUT_NAME = "metrics_essentials.shp"  # output filename (written next to FULL_METRICS_SHP)
PREVIEW_DIRNAME = "Preview_maps"        # subfolder for map previews, next to FULL_METRICS_SHP
TARGET_HABITAT = HABITAT_TARGETS[0]     # using first target habitat

# Column names 
ID_COL = SHP_ID_COLNAME
X_COL = SHP_X_COLNAME
Y_COL = SHP_Y_COLNAME
AREA_COL = SHP_AREA_COLNAME
MOST_PROB_COL = "mostProb"

# Pattern used to find all habitat probability columns, e.g. prob_h_1, prob_h_2, ...
HABITAT_PROB_PATTERN = re.compile(r"^prob_h_\d+$")

COLORS = {
    "drift": ['#c9eac2', '#7bc77c', '#2a924b', '#00441b'],
    "probability": ['#ffe1e1', '#ffb0ac', '#ff8080', '#ff4040'],
    "shifts": ['#d7e6f5', '#afd1e7', '#3e8ec4', '#1663aa'],
    "desiccation": ['#fdd2a5', '#fd9243', '#df5005', '#7f2704'],
}

NO_DATA_COLOR = "#e0e0e0"

# ---------------------------------------------------------------------------
# Zoom window configuration
# ---------------------------------------------------------------------------
# In addition to the full-extent preview maps, a second, zoomed-in version
# of each map is generated over a small area, so local detail is easier to
# read. Set ZOOM_CENTER explicitly (x, y in the same CRS/units as the
# shapefile) to target a specific spot; leave it as None to default to the
# centroid of the data's full extent. The zoom window's width/height can be
# set explicitly via ZOOM_WIDTH / ZOOM_HEIGHT (same units as the shapefile),
# otherwise they default to ZOOM_FRACTION of the full extent's width/height.
ZOOM_ENABLED = True
ZOOM_CENTER = None       # e.g. (612345.0, 6543210.0); None = auto (centroid of full extent)
ZOOM_WIDTH = None        # e.g. 500 (in map units); None = derived from ZOOM_FRACTION
ZOOM_HEIGHT = None       # e.g. 500 (in map units); None = derived from ZOOM_FRACTION
ZOOM_FRACTION = 0.15     # used only when ZOOM_WIDTH / ZOOM_HEIGHT are None
ZOOM_SUFFIX = "_zoom"         # appended to the filename (before the extension) for the zoomed-in maps
ZOOM_AREA_SUFFIX = "_zoom_area"  # appended for the full-extent map with the zoom rectangle drawn on it


# ---------------------------------------------------------------------------
# Step 1: crop
# ---------------------------------------------------------------------------

def build_keep_columns(gdf: gpd.GeoDataFrame):
    """Returns (keep_cols, warnings) where warnings lists any requested
    conditional column that doesn't actually exist in the data."""
    keep_cols = [ID_COL, X_COL, Y_COL, AREA_COL]
    warnings = []

    habitat_prob_cols = sorted(c for c in gdf.columns if HABITAT_PROB_PATTERN.match(c))
    if not habitat_prob_cols:
        warnings.append(
            f"No columns matched HABITAT_PROB_PATTERN ({HABITAT_PROB_PATTERN.pattern}). "
            f"Check the pattern against your actual column names: {list(gdf.columns)}"
        )
    keep_cols += habitat_prob_cols

    keep_cols.append(MOST_PROB_COL)

    if METRICS_TO_COMPUTE.get("shift_targ_daily", False):
        col = f"h{TARGET_HABITAT}_sh_suit"
        keep_cols.append(col)
        if col not in gdf.columns:
            warnings.append(f"shift_targ_daily is true but '{col}' was not found in the data")

    if METRICS_TO_COMPUTE.get("drift_percentile", False):
        col = f"h{TARGET_HABITAT}_driftP"
        keep_cols.append(col)
        if col not in gdf.columns:
            warnings.append(f"drift_percentile is true but '{col}' was not found in the data")

    if METRICS_TO_COMPUTE.get("desiccation_risk", False):
        col = f"h{TARGET_HABITAT}_desicR"
        keep_cols.append(col)
        if col not in gdf.columns:
            warnings.append(f"desiccation_risk is true but '{col}' was not found in the data")

    return keep_cols, warnings


def crop_essentials(input_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(input_path)
    print(f"Loaded {len(gdf)} features, {len(gdf.columns)} columns from {input_path.resolve()}")

    keep_cols, warnings = build_keep_columns(gdf)

    for w in warnings:
        print(f"WARNING: {w}")

    keep_cols_final = [c for c in keep_cols if c in gdf.columns]
    missing = [c for c in keep_cols if c not in gdf.columns]
    if missing:
        print(f"WARNING: these expected columns were not found and will be skipped: {missing}")

    if not keep_cols_final:
        sys.exit("ERROR: no columns to keep — check your column name settings above.")

    essentials = gdf[keep_cols_final + ["geometry"]]
    return essentials


# ---------------------------------------------------------------------------
# Step 2: preview maps — binning
# ---------------------------------------------------------------------------

def bin_probability(values):
    edges = [0, 0.25, 0.50, 0.75, 1.0]
    labels = ["0.00–0.25", "0.25–0.50", "0.50–0.75", "0.75–1.00"]
    binned = pd.cut(values, bins=edges, labels=labels, include_lowest=True)
    return binned, labels


def bin_risk_class(values):
    edges = [0.5, 1.5, 2.5, 3.5, 4.5]
    labels = ["1", "2", "3", "4"]
    binned = pd.cut(values, bins=edges, labels=labels)
    return binned, labels


def _format_edge(v):
    """Formats a float bin edge with just enough decimals to stay
    distinguishable, so labels for very small ranges (e.g. shift values
    all below 1) don't collapse to identical-looking strings like
    '0.00–0.00'."""
    if v == 0:
        return "0"
    magnitude = abs(v)
    if magnitude < 0.001:
        return f"{v:.5f}"
    elif magnitude < 0.01:
        return f"{v:.4f}"
    elif magnitude < 1:
        return f"{v:.3f}"
    else:
        return f"{v:.2f}"


def bin_shifts(values):
    """
    Bins 'shift' values into up to 4 categories.

    Two distinct kinds of shift columns show up in practice:
      - integer-valued "shift counts" (e.g. 0, 1, 2, 3 shifts): keep the
        original behaviour - one category per distinct integer when
        there are <=4 of them, otherwise group into integer ranges.
      - continuous float values (e.g. h2_sh_suit, typically small
        fractions between 0 and 1): the old code assumed these were
        integers too, so `int(values.max())` truncated anything below 1
        down to 0, producing duplicate bin edges like [0, 0, 0, 0, 1]
        and crashing pd.cut. These are now binned into up to 4
        equal-width intervals over the column's own min/max instead,
        the same way bin_probability bins a 0-1 range.
    """
    unique_vals = np.sort(values.dropna().unique())

    if len(unique_vals) == 0:
        return pd.Series(index=values.index, dtype=object), []

    is_integer_like = bool(np.all(np.isclose(unique_vals, np.round(unique_vals))))

    if is_integer_like:
        if len(unique_vals) <= 4:
            edges = np.concatenate(([unique_vals[0] - 0.5], unique_vals + 0.5))
            labels = [str(int(v)) for v in unique_vals]
        else:
            max_val = int(values.max())
            edges = np.linspace(0, max_val + 1, 5, dtype=int)
            labels = [f"{edges[i]}-{edges[i+1]-1}" for i in range(4)]
    else:
        min_val = float(values.min())
        max_val = float(values.max())
        if len(unique_vals) <= 4:
            # few distinct float values - treat each as its own category
            # rather than smearing them into a continuous range
            eps = max((max_val - min_val) * 1e-6, 1e-9)
            edges = np.concatenate(([unique_vals[0] - eps], unique_vals + eps))
            labels = [_format_edge(float(v)) for v in unique_vals]
        elif min_val == max_val:
            # every valid value is identical - a single bin/category
            edges = [min_val - 1e-9, max_val + 1e-9]
            labels = [_format_edge(min_val)]
        else:
            edges = np.linspace(min_val, max_val, 5)
            labels = [f"{_format_edge(edges[i])}–{_format_edge(edges[i+1])}" for i in range(4)]

    binned = pd.cut(values, bins=edges, labels=labels, include_lowest=True, duplicates="drop")
    return binned, labels


# ---------------------------------------------------------------------------
# Step 2: preview maps — zoom window
# ---------------------------------------------------------------------------

def compute_zoom_bounds(gdf: gpd.GeoDataFrame):
    """Returns (xmin, xmax, ymin, ymax) for the zoom window, based on the
    ZOOM_CENTER / ZOOM_WIDTH / ZOOM_HEIGHT / ZOOM_FRACTION settings above.
    Computed once from the full data extent and reused for every metric so
    all zoomed maps show the exact same area."""
    minx, miny, maxx, maxy = gdf.total_bounds
    full_width = maxx - minx
    full_height = maxy - miny

    if ZOOM_CENTER is not None:
        cx, cy = ZOOM_CENTER
    else:
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2

    width = ZOOM_WIDTH if ZOOM_WIDTH is not None else full_width * ZOOM_FRACTION
    height = ZOOM_HEIGHT if ZOOM_HEIGHT is not None else full_height * ZOOM_FRACTION

    return (cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2)


# ---------------------------------------------------------------------------
# Step 2: preview maps — plotting
# ---------------------------------------------------------------------------

def plot_map(gdf, binned, labels, palette, title, save_path, valid_mask,
             zoom_bounds=None, zoom_rect=None):
    """Draws one preview map.

    zoom_bounds: if given, (xmin, xmax, ymin, ymax) - the axis is cropped
        to this window, producing a zoomed-in view of the same data
        (no need to recompute binning/colors for this).
    zoom_rect: if given, (xmin, xmax, ymin, ymax) - draws a dashed
        rectangle outlining this area on the (typically full-extent) map,
        so it's easy to see where the corresponding zoomed map was taken
        from. Ignored when zoom_bounds is also set (no point drawing the
        indicator on the zoomed map itself).
    """
    if len(labels) == 0:
        print(f"No valid values for {title}. Skipping.")
        return

    color_map = {label: palette[i % len(palette)] for i, label in enumerate(labels)}

    colors = pd.Series(NO_DATA_COLOR, index=gdf.index, dtype=object)
    colors.loc[valid_mask] = (
        binned.loc[valid_mask].astype(object).map(color_map).fillna(NO_DATA_COLOR)
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color=colors.tolist(), edgecolor="none")

    legend = [mpatches.Patch(color=color_map[l], label=l) for l in labels]
    legend.append(mpatches.Patch(color=NO_DATA_COLOR, label="No data / not applicable"))

    ax.legend(
        handles=legend,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=12,
        title_fontsize=13,
    )

    if zoom_bounds is not None:
        xmin, xmax, ymin, ymax = zoom_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    elif zoom_rect is not None:
        xmin, xmax, ymin, ymax = zoom_rect
        ax.add_patch(mpatches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, edgecolor="black", linewidth=1.5, linestyle="--", zorder=5,
        ))

    ax.set_title(title, fontsize=18)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved {save_path}")


def zoomed_path(save_path: Path) -> Path:
    """Builds the output filename for the zoomed-in counterpart of a map,
    e.g. map_probability_h2.png -> map_probability_h2_zoom.png."""
    return save_path.with_name(f"{save_path.stem}{ZOOM_SUFFIX}{save_path.suffix}")


def zoom_area_path(save_path: Path) -> Path:
    """Builds the output filename for the full-extent map annotated with
    the zoom rectangle, e.g.
    map_probability_h2.png -> map_probability_h2_zoom_area.png."""
    return save_path.with_name(f"{save_path.stem}{ZOOM_AREA_SUFFIX}{save_path.suffix}")


def plot_map_with_zoom(gdf, binned, labels, palette, title, save_path, valid_mask, zoom_bounds):
    """Saves three versions of the map, reusing the same binned
    values/colors for all of them so nothing is recomputed:
      1. the plain full-extent map, unchanged, at the original filename
      2. the same full-extent map with a dashed rectangle marking the
         zoom area (only if zooming is enabled)
      3. a zoomed-in view cropped to zoom_bounds (only if zooming is
         enabled)
    """
    # 1. plain original, no annotation
    plot_map(gdf, binned, labels, palette, title, save_path, valid_mask)

    if zoom_bounds is not None:
        # 2. full extent + dashed rectangle showing the zoom area
        plot_map(gdf, binned, labels, palette, f"{title} (zoom area)",
                  zoom_area_path(save_path), valid_mask, zoom_rect=zoom_bounds)

        # 3. zoomed-in view
        plot_map(gdf, binned, labels, palette, f"{title} (zoomed)",
                  zoomed_path(save_path), valid_mask, zoom_bounds=zoom_bounds)


def generate_preview_maps(gdf: gpd.GeoDataFrame, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    prob_col = f"prob_h_{TARGET_HABITAT}"
    if prob_col not in gdf.columns:
        raise ValueError(f"Column '{prob_col}' not found.")

    valid_mask = gdf[prob_col].fillna(0) > 0
    print(f"Meshes with habitat: {valid_mask.sum()} / {len(valid_mask)}")

    zoom_bounds = compute_zoom_bounds(gdf) if ZOOM_ENABLED else None
    if zoom_bounds is not None:
        xmin, xmax, ymin, ymax = zoom_bounds
        print(f"Zoom window: x=[{xmin:.2f}, {xmax:.2f}], y=[{ymin:.2f}, {ymax:.2f}]")

    # Probability
    binned = pd.Series(index=gdf.index, dtype=object)
    binned_valid, labels = bin_probability(gdf.loc[valid_mask, prob_col])
    binned.loc[valid_mask] = binned_valid
    plot_map_with_zoom(
        gdf, binned, labels, COLORS["probability"],
        f"Preview habitat {TARGET_HABITAT} probability",
        save_dir / f"map_probability_h{TARGET_HABITAT}.png",
        valid_mask, zoom_bounds,
    )

    # Shifts
    if METRICS_TO_COMPUTE.get("shift_targ_daily", False):
        shift_col = f"h{TARGET_HABITAT}_sh_suit"
        if shift_col in gdf.columns:
            binned = pd.Series(index=gdf.index, dtype=object)
            binned_valid, labels = bin_shifts(gdf.loc[valid_mask, shift_col])
            binned.loc[valid_mask] = binned_valid
            plot_map_with_zoom(
                gdf, binned, labels, COLORS["shifts"],
                f"Preview habitat {TARGET_HABITAT} daily shifts",
                save_dir / f"map_shifts_h{TARGET_HABITAT}.png",
                valid_mask, zoom_bounds,
            )
        else:
            print(f"WARNING: shift_targ_daily is true but '{shift_col}' not found — skipping shifts map.")

    # Drift
    if METRICS_TO_COMPUTE.get("drift_percentile", False):
        drift_col = f"h{TARGET_HABITAT}_driftP"
        if drift_col in gdf.columns:
            binned = pd.Series(index=gdf.index, dtype=object)
            binned_valid, labels = bin_risk_class(gdf.loc[valid_mask, drift_col])
            binned.loc[valid_mask] = binned_valid
            plot_map_with_zoom(
                gdf, binned, labels, COLORS["drift"],
                f"Preview habitat {TARGET_HABITAT} drift risk",
                save_dir / f"map_drift_h{TARGET_HABITAT}.png",
                valid_mask, zoom_bounds,
            )
        else:
            print(f"WARNING: drift_percentile is true but '{drift_col}' not found — skipping drift map.")

    # Desiccation
    if METRICS_TO_COMPUTE.get("desiccation_risk", False):
        desic_col = f"h{TARGET_HABITAT}_desicR"
        if desic_col in gdf.columns:
            binned = pd.Series(index=gdf.index, dtype=object)
            binned_valid, labels = bin_risk_class(gdf.loc[valid_mask, desic_col])
            binned.loc[valid_mask] = binned_valid
            plot_map_with_zoom(
                gdf, binned, labels, COLORS["desiccation"],
                f"Preview habitat {TARGET_HABITAT} desiccation risk",
                save_dir / f"map_desiccation_h{TARGET_HABITAT}.png",
                valid_mask, zoom_bounds,
            )
        else:
            print(f"WARNING: desiccation_risk is true but '{desic_col}' not found — skipping desiccation map.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(input_shp_path=None):
    input_path = Path(input_shp_path) if input_shp_path else Path(FULL_METRICS_SHP)
    if not input_path.exists():
        sys.exit(f"ERROR: input shapefile not found: {input_path.resolve()}")

    # Step 1: crop
    essentials = crop_essentials(input_path)

    output_path = input_path.parent / OUTPUT_NAME
    essentials.to_file(output_path)
    print(f"Wrote {len(essentials)} features to {output_path.resolve()}")
    print(f"Columns kept: {list(essentials.columns)}")

    # Step 2: preview maps, using the essentials we just built (no re-read needed)
    preview_dir = input_path.parent / PREVIEW_DIRNAME
    generate_preview_maps(essentials, preview_dir)
    print(f"Preview maps written to {preview_dir.resolve()}")


if __name__ == "__main__":
    main()