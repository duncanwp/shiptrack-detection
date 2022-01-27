"""
Extract mean brightness temperature within shiptrack geometries
Accelerated by GPU using NVIDIA RAPIDS python libs
sufficient requirement RAPIDS 21.12

Angus Laurenson
Plymouth Marine Laboratory
anla@pml.ac.uk

"""

# usual imports
import xarray as xr
import geopandas as gpd
from shapely.geometry.polygon import Polygon
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import os

from datetime import datetime
import re
import argparse

# GPU packages
import cupy
import cudf
import cuspatial

if __name__ == "__main__":
    # main loop
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--geopackages", nargs="+")

    args = parser.parse_args()

    geopackages = args.geopackages

    signatures = [gpkg.split("/")[-1][:14] for gpkg in geopackages]

    composites = [
        os.dirname(gpkg).replace(
                'results',
                'data'
            ) + f'/{signature}_day_microphysics_.nc' \
            for gpkg,signature in zip(geopackages, signatures)
        ]

    # process in series but using the GPU
    for gpkg, composite in tqdm(zip(geopackages, composites)):
        
        # massive try except to log failures
        try:
            # open the geopackage
            gdf = gpd.read_file(gpkg, engine='GPKG', layer='swath')
            
            # move to the GPU
            gdf = cuspatial.from_geopandas(gdf)

            # open the composite file
            da = xr.open_dataarray(composite)

            # load to GPU
            da.data = cupy.array(da.data)

            # select brightness temperature "B"
            # convert to degree celcius
            da = da.sel(bands='B') - 273.15

            # create point coords
            x,y = cupy.meshgrid(
                cupy.array(da['x'].values),
                cupy.array(da['y'].values)
            )

            cudf_masks = cuspatial.point_in_polygon(
                # test points
                test_points_x = x.reshape(-1),
                test_points_y = y.reshape(-1),
                # polygon names
                poly_offsets = gdf.geometry.polygons.polys,
                # start index of each polygon
                # drop the last index from rings as it is the last point
                # half the values to get correct indices for .x and .y
                # as they refer to the flat .xy which is [x1,y1,x2,y2,...]
                poly_ring_offsets = gdf.geometry.polygons.rings//2,
                
                # arrays of polygon coordinates
                poly_points_x = gdf.geometry.polygons.x,
                poly_points_y = gdf.geometry.polygons.y,
            )

            # reshape the masks
            cudf_masks = [cudf_masks[col].values.reshape((da.y.size,da.x.size)) for col in cudf_masks.columns]
            
            # drop any spurious masks that have no True values
            # @anla : I think cuspatial.point_in_polygon returns an extra empty mask
            # and I do not know why it does this. This work around will add a small overhead
            # but will be resilient to changes in the future that fixx this behaviour
            cudf_masks = filter(cupy.any, cudf_masks)
            
            # put inside xarray 
            cudf_masks = [xr.DataArray( dims=('y','x'), data=mask) for mask in cudf_masks] 
            
            # drop the last mask as it is empty and has no corresponding polygon
            # not sure why this is returned, feel like it is a wrong behaviour...
        
            # use to index the dataset 
            gdf['temperature'] = [float(da.where(mask).mean().data) for mask in cudf_masks]

            # get the date from filename
            dt = datetime.strptime(re.findall("\d{13}", gpkg)[0])
            gdf['date'] = dt

            # write new layer to geopackage
            gdf[['temperature','date']].to_file(
                gpkg,
                driver='GPKG',
                layer='meta'
            )
        except Exception as e:
            print(f"geopackage {gpkg} raised error {e}")