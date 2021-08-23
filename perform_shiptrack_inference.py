#!/usr/bin/env python
"""
Run inference over a dataset using a saved tf model

@ anla
    
    What operations can we perform on the GPU to give speedup?
    * image enhancement? (currently histogram stretching is done before) possible?
    * contour finding? No implementation yet found
    * coordinate lookup for converting contours from x,y to lon,lat? Only worth it for large number of points.

"""
import matplotlib
matplotlib.use('agg')

from tqdm import tqdm
import xarray as xr
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os
import re
from shapely.geometry import Polygon, MultiPolygon
from datetime import datetime

# from multiprocessing import Pool

# Use of globals like this I do not like...
# IMG_SIZE = 448
# INT_IMG_SIZE = (5*IMG_SIZE, 3*IMG_SIZE)

import logging
logger = logging.getLogger(f"ship_track_inference_as_{__name__}")

def vectoriser(arr, level=0.2, latlon=True):
    """
    input: arr -> a square 2D binary mask array
    output: polys -> a list of vectorised polygons
    
    From https://gist.github.com/Lkruitwagen/26c6ba8cadbfd89ab42f36f6a3bbdd35
    """    
    from skimage import measure
    
    polys = []

    contours = measure.find_contours(arr, level)
    
    for c in contours:
        if latlon:
            # FIXME: This will need a da piping through (maybe this fn should always expect an xr.da?)
            c = np.stack([da.longitude[(np.round(c[:,1])).astype('int')], da.latitude[(np.round(c[:,0])).astype('int')]], 1)
        else:
            c.T[[0, 1]] = c.T[[1, 0]] # swap lons<->lats
        poly = geometry.Polygon(c) # pass into geometry
        polys.append(poly)
    # TODO: Should this return a shapely MultiPolygon or a GeoDataFrame...?
#     return geometry.MultiPolygon(polys)
    return gpd.GeoDataFrame({"geometry": polys})


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


def combine_masks(masks):

    nested_masks = [np.split(m, 3) for m in np.split(masks[..., 0], 5, 0)]
    mask = np.squeeze(np.block(nested_masks))

    return mask

def get_ship_track_array(rgb_array, tf_predictor):

    # reshape and split array prior to inference
    original_size = rgb_array.shape[1::-1] # The size is the inverse of the shape (minus the color channel)
    reshaped_rgb_array = resize_arr(rgb_array, INT_IMG_SIZE[::-1])
    split_rgb_array = split_array(reshaped_rgb_array / 255., IMG_SIZE)

    # call model for inference
    prediction = tf_predictor(data=tf.constant(split_rgb_array, dtype='float32'))

    # stitch and resize output back to original size
    inferred_ship_tracks = combine_masks(prediction['sigmoid'])
    inferred_ship_tracks = resize_arr(inferred_ship_tracks, original_size)
    
    return inferred_ship_tracks

def load_data(filename):
    # utility function to handle different inputs
    # takes filename, returns rgb_array
    # metadata can be picked up later

    # NETCDF
    if filename.endswith(".nc"):
        # open netcdf with xarray
        ds = xr.open_dataset(filename)

        # transpose dimensions to PIL compatible
        # netcdf dataset open
    
        # reorder dims to put bands last
        ds = ds.transpose('y','x','bands')

        # convert to numpy array, move to PIL compatible array...
        rgb_array = (np.array(ds['day_microphysics'].data)*255).astype(np.uint8)

        # return plain rgb array for further processing
        return rgb_array

    
    # PNG
    elif filename.endswith(".png"):
        # open png with PIL
        img = Image.open(filename)
        # turn to numpy array, dropping alpha
        rgb_array = np.array(img)[:, :, 0:3] 
        
        return rgb_array

def pre_processing(rgb_array):
    # perform pre_processing steps
    # histogram stretching
    # image resizing etc
    # return array
    return rgb_array

def post_processing(output, extension):
    # perform post processing as desired
    pass

def write_output(output, outname, **kwargs):
    # write output to file
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model', help="The model directory to use")
    parser.add_argument('infiles', nargs='*')
    parser.add_argument('--infile', help="Input file", type=argparse.FileType('r'))
    parser.add_argument('--show', action='store_true')
    
    parser.add_argument('--outdir', required=True, help="path to results directory")
    
    parser.add_argument('--outsuffix', help="suffix of output file", default='_inferred_ship_tracks')
    parser.add_argument('--outextension', help="file extension", default='.nc')

    parser.add_argument("--imsize", help="block size magic number",default=448)
    
    parser.add_argument('-l','--level', help="level of inference contour for geometry output", default=0.2)

#     parser.add_argument('--log-level', dest="log_level", default="INFO", choices=("INFO","DEBUG","WARNING","ERROR","CRITICAL"))
    
    args = parser.parse_args()
    
    # create logger with 'spam_application'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('inference.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # globals 
    IMG_SIZE = args.imsize
    INT_IMG_SIZE = (5*IMG_SIZE, 3*IMG_SIZE)
    logger.info(f"set image tiling globals : IMG_SIZE={IMG_SIZE}, INT_IMG_SIZE={INT_IMG_SIZE}")

    # load (compile?) trained model
    logger.info(f"loading model {args.model}")
    MODEL = tf.saved_model.load(args.model)
    tf_predictor = MODEL.signatures["serving_default"]

    if args.infile is not None:
        logger.info("Reading filelist from {}".format(args.infile))
        infiles = [x.replace("\n","") for x in args.infile.readlines()]
    else:
        infiles = args.infiles

    # main loop through input files...
    for file in tqdm(infiles):
            
        logger.info("creating output filename \n \
        checking if output already exists")
        # construct output filename from input filename
        basename = os.path.basename(file[:-4])+args.outsuffix+args.outextension

        date = datetime.strptime(
            re.findall("\d+",basename)[0],
            "%Y%j%H%M%S"
        )
        
        date_str = date.strftime("%Y/%m/%d")
        
        outdir = os.path.join(args.outdir, date_str)
        
        out_name = os.path.join(outdir, basename)
        
        # if the file you want to make exists - skip
        if os.path.isfile(out_name) == True:
            logger.info("output filename exists,\n \
            skipping processing")
            continue
        
        logger.debug(f"reading from {file}")
        
        # read in the data
        rgb_array = load_data(file)

        # preprocess the data
        rgb_array = pre_processing(rgb_array)

        logger.debug(f"rgb_array = {rgb_array}")
        
        # perform inference on rgb
        inferred_ship_tracks = get_ship_track_array(rgb_array, tf_predictor)
                
        # create directory for output
        if os.path.isdir(outdir) == False:
            os.makedirs(outdir)
        
        # NETCDF input
        if file[-3:] in [".nc", "hdf"]:
            ds = xr.open_dataset(file)
            ds['ship_tracks'] = xr.Variable(
                dims=['y','x'],
                data=inferred_ship_tracks,
                # FIXME @anla : this gets overwritten when contour to geometry
                attrs={'description':'inferred ship track array'},
            )
            
            logger.info(args.outextension)
            if args.outextension == ".nc":
                logger.info("writing netcdf")
                ds.to_netcdf(out_name)              

            # SHAPE FILE OUTPUT      
            elif args.outextension == ".shp":
                # assume shapefile output
                # get geometry and add to GeoDataFrame
                logger.info("computing shapes")
                gdf = xr_vectoriser(ds['ship_tracks'], level=args.level)

                # assign attributes from data array to GeoDataFrame

                gdf = gdf.join(pd.DataFrame(columns = ds['day_microphysics'].attrs.keys()))

                for key, value in ds['day_microphysics'].attrs.items():
                    gdf[key] = str(value)   

                # assign custom values from this operation
                gdf['model'] = args.model
                gdf['level'] = args.level
                gdf['description'] = f"multipolygon describing contour at level {args.level} of infered ship track probability"
                
                # write to shapefile
                gdf.to_file(out_name)


        else:
                
            # Get the polygons in image coordinates
            #polys = vectoriser(mask, level=0.2, latlon=False)
            
            # Get the polygons in lat/lon coordinates (FIXME this will require an xarray da with the right coordinates)
            #polys = vectoriser(mask, level=0.2, latlon=True)
            
            # TODO: Save to PostGIS DB?
            
            # Save the mask?
            np.savez_compressed(f"{file[:-4]}_{args.outfile}.npz", mask=inferred_ship_tracks)
            
            if args.show:
                fig, axs = plt.subplots(figsize=(20, 40))
                axs.imshow(rgb_array, vmin=0., vmax=1.)
                im=axs.imshow(inferred_ship_tracks, alpha=0.5, vmin=0, vmax=1)
                plt.savefig(f"{file[:-4]}_{args.outfile}.png")
