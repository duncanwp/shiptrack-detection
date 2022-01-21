#!/usr/bin/env python

"""
Compute contours from netcdf files with single data variable

Angus Laurenson
Plymouth Marine Laboratory
anla@pml.ac.uk

Parameters

    input : a netcdf file
    vars : a list of datavariables
    levels : a list of contour levels

Usage:

    # compute two different contours for the same variable
    contour_maker.py --input example.nc --vars shiptracks shiptracks --levels 0.9 0.8

ToDo
 -  Implement cupy coordinate lookup for speedup 
    on large contour list
"""

import xarray as xr
import geopandas as gpd
import pandas as pd
import logging

from skimage import measure
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

from datetime import datetime
import argparse
import os
import re

logger = logging.getLogger(__name__)

def xarray_contour_to_geodataframe(da:xr.DataArray, level:float)->tuple:
    """Find contour of geolocated dataarray object and return
    geodataframe with geometry columns for unlabelled (swath)
    and labelled geographic coordinates (lat, lon)"""
    
    # get the contours at the given level
    # filter out those contours that can't make valid polygon
    contours = measure.find_contours(da.data.T, level)
    contours = list(filter(lambda x:len(x) > 3,contours))
    
    # swath coordinates are unlabelled array indices
    swath_geometry = MultiPolygon([Polygon(c.astype(int)) for c in contours])
    
    # geographic coordinates must be pulled from the coordinate arrays
    # coordinates might have different names/spellings, so preserve those
    contours_geographic = []
    for contour in contours:
        contours_geographic.append([[float(da[coordinate][y,x]) for coordinate in da.coords] for (x,y) in contour.astype(int)])
        
    geographic_geometry = MultiPolygon([Polygon(c) for c in contours_geographic])
    
    # create geodataframes
    swath_gdf = gpd.GeoDataFrame({'geometry':swath_geometry})
    geo_gdf = gpd.GeoDataFrame({'geometry':geographic_geometry})

    # add metadata to geodataframes
    # meta_df = pd.DataFrame(da.attrs)

    # swath_gdf = swath_gdf.join(meta_df)
    # geo_gdf = geo_gdf.join(meta_df)

    return swath_gdf, geo_gdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input',type=str)
    parser.add_argument('--vars', type=str, nargs='+')
    parser.add_argument('--levels',type=float, nargs='+')
    parser.add_argument('--overwrite',type=bool, default=False)

    args = parser.parse_args()

    ds = xr.open_dataset(args.input)

    for var, level in zip(args.vars, args.levels):
        
        # construct the geopackage filename from the output
        filename = args.input.replace('.nc',f'_contour_{var}_level_{level}.gpkg')

        # skip if existing
        if os.path.isfile(filename) & (args.overwrite is False):
            logger.info(f'skipping existing {filename}')
            continue

        else:
            # compute the geodataframe
            swath_gdf, geo_gdf = xarray_contour_to_geodataframe(ds[var], level)

            # datetime from filename
            dt = datetime.strptime("%Y%j%H%M%S", re.findall("\d{13}", filename)[0])

            swath_gdf['date'] = dt
            geo_gdf['date'] = dt

            # write to geopackage in layers
            swath_gdf.to_file(
                    filename,
                    driver='GPKG',
                    layer='swath'
                )

            geo_gdf.to_file(
                filename,
                driver='GPKG',
                layer='geographic'
            )

            # attr_gdf.to_file(
            #     filename,
            #     driver='GPKG',
            #     layer='attrs'
            # )

