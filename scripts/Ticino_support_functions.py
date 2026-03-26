import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
import glob
import os

# folder containing rasters
folder = r"C:\Users\lecrivau\Documents\00_Research_Assistant\Toolbox\Ticino_case_study"

depth_files = sorted(glob.glob(os.path.join(folder, "d_*.tif")))
vel_files   = sorted(glob.glob(os.path.join(folder, "v_*.tif")))
dsm_file = os.path.join(folder, "DSM_2056_pix_size_0_5_Computational_mesh.tif")

# read reference raster (first depth raster)
with rasterio.open(depth_files[0]) as src:
    depth = src.read(1)
    transform = src.transform
    crs = src.crs
    height = src.height
    width = src.width
    res_x, res_y = src.res

cell_area = abs(res_x * res_y)

polygons = []
x_coords = []
y_coords = []
ids = []

id_counter = 1

# build polygons
for row in range(height):
    for col in range(width):

        x_min, y_max = transform * (col, row)
        x_max, y_min = transform * (col + 1, row + 1)

        geom = box(x_min, y_min, x_max, y_max)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        polygons.append(geom)
        x_coords.append(x_center)
        y_coords.append(y_center)
        ids.append(id_counter)

        id_counter += 1

gdf = gpd.GeoDataFrame({
    "id": ids,
    "x": x_coords,
    "y": y_coords,
    "area": cell_area
}, geometry=polygons, crs=crs)

# add elevation values (z)
with rasterio.open(dsm_file) as src:
    z_data = src.read(1).flatten()

gdf["z"] = z_data

# add depth values
for file in depth_files:
    discharge = os.path.basename(file).replace("d_", "").replace(".tif", "")
    
    with rasterio.open(file) as src:
        data = src.read(1).flatten()
    
    gdf[f"d_{discharge}"] = data

# add velocity values
for file in vel_files:
    discharge = os.path.basename(file).replace("v_", "").replace(".tif", "")
    
    with rasterio.open(file) as src:
        data = src.read(1).flatten()
    
    gdf[f"v_{discharge}"] = data


# remove nodata cells if needed
gdf = gdf.dropna()

# round all numeric columns to 2 decimals
for col in gdf.columns:
    if col != "geometry":
        gdf[col] = (gdf[col] * 100).round().astype(np.int32) / 100

# save shapefile
output = os.path.join(folder, "hydraulic_mesh_with_elevation.shp")
gdf.to_file(output)

print("Shapefile saved:", output)