"""
Reshape, cut, stich and more


"""

from tqdm import tqdm
import xarray as xr
from trollimage.xrimage import XRImage
import tensorflow as tf
import numpy as np
import argparse
import pandas as pd
import geopandas as gpd
import os
import re
from shapely.geometry import Polygon, MultiPolygon
from datetime import datetime
from functools import reduce
import logging

logger = logging.getLogger(f"ship_track_inference_as_{__name__}")

def swath_contour_to_geo_polygon(contour, latitudes, longitudes):
    """convert a single contour from swath coordinates (x,y)
    to geographic coordinates (lon, lat), given the 
    latitudes and longitudes arrays"""
    
    return Polygon([(float(longitudes[x,y]),
                  float(latitudes[x,y])) \
                 for (x,y) in contour.astype(int)])

def xr_vectoriser(da, level=0.2)-> gpd.GeoDataFrame:
    """
    input: arr -> a square 2D binary mask array
    output: polys -> a list of vectorised polygons
    
    From https://gist.github.com/Lkruitwagen/26c6ba8cadbfd89ab42f36f6a3bbdd35
    """    
    from skimage import measure
        
    contours = measure.find_contours(da.data, level)
    
    # filter out those contours that can't make valid polygon
    contours = list(filter(lambda x:len(x) > 3,contours))
    
    # polygons in swath coodinates (x,y)
    xy_multipolygon = MultiPolygon([Polygon(c.astype(int)) for c in contours])
    
    # polygons in geographic coordinate (lon, lat)
    geo_multipolygon = MultiPolygon([swath_contour_to_geo_polygon(c, da['latitude'], da['longitude']) for c in contours])
        
    gdf = gpd.GeoDataFrame({"geometry": [xy_multipolygon, geo_multipolygon], "coords":["swath","geographic"]})

    return gdf


def split_array(arr, step_size=440):
    x_len, y_len = arr.shape[0:2]
    x_split_locs = range(step_size, x_len, step_size)
    y_split_locs = range(step_size, y_len, step_size)
    v_splits = np.array_split(arr, x_split_locs, axis=0)
    return sum((np.array_split(v_split, y_split_locs, axis=1) for v_split in v_splits), [])


def get_image(file):
    from PIL import ImageOps, Image

    im_data = Image.open(file)
    
    # Drop the alpha channel
    im_data = np.array(im_data)[:, :, 0:3]
    
    return im_data


def resize_arr(arr, target_size, bilinear=True):
    from PIL import Image
    
    # @anla : I don't like this hidden behaviour
    resampler = Image.BILINEAR if bilinear else Image.NEAREST
    
    arr_im = Image.fromarray(arr) 

    arr_im = arr_im.resize(target_size, resample=resampler)
    
    new_arr = np.array(arr_im)
    return new_arr


def stitch(masks):

    nested_masks = [np.split(m, 3) for m in np.split(masks[..., 0], 5, 0)]
    mask = np.squeeze(np.block(nested_masks))

    return mask


def stretch_dataarray(x:xr.DataArray, stretch='histogram') -> xr.DataArray:
    '''perform a trollimage stretch operation with given method
    stretch is applied in-place and the enhancement attribute is updated'''
     
    xrimage = XRImage(x)
    
    xrimage.stretch(stretch=stretch)

    return xrimage.data

def rgb_netcdf_to_tensor(ncfile) -> tf.Tensor:
    # load netcdf as xr.dataarray and convert to tf.Tensor
    # transpose RGB channels to last dimension and downcast
    # to float32. Tensorflow expects this I think
    
    ds = xr.load_dataarray(ncfile)
    ds = ds.transpose(...,'bands').astype('float32')

    return tf.convert_to_tensor(ds)

def ncfiles_to_dataset(ncfiles) -> tf.data.Dataset:
    # given list of netcdf files, return a tf Dataset

    # this syntax returns a callable that itself returns iterable
    # necessary for use with tensorflow's from_generator function
    gen = lambda: (rgb_netcdf_to_tensor(x) for x in ncfiles)

    return tf.data.Dataset.from_generator(gen, output_types='float')
