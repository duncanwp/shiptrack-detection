#!/usr/bin/env python
"""
Simplified script that run inference for NEODAAS request 21_02
that only accepts netcdf input files and returns either netcdf of gpkg

@ anla
    
    What operations can we perform on the GPU to give speedup?
    * image enhancement? (currently histogram stretching is done before) possible?
    * contour finding? No implementation yet found
    * coordinate lookup for converting contours from x,y to lon,lat? Only worth it for large number of points.

ISSUES:

    The main issue with this script is the pattern, list comprehensions to tf.data.Dataset,
    which does not make best use of the tf.data.Dataset class and cannot prefetch the data
    for higher performance.

    It would be nice to use the .map and .prefetch and .batch methods in a meaningful way.

    Currently, PIL is causing a loss of colourspace resolution. We rely on it for reshaping
    the images to a common shape. However, PIL will not accept a 3 channel (RGB) array unless 
    it is of the data type uint8 and has values between 0 and 255. Conversely, the model
    expects Tensors with floating dtype with values between 0 and 1.

    We could use tf.image to do the resizing and maybe splitting. However, these are known to
    have unexpected behaviour that is somewhat different to PIL. Never-the-less, it would be 
    good to lean into the tensorflow 'ecosystem' more, as that should lead to better performance
    and more canomical workflows and syntax.

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


def combine_masks(masks):

    nested_masks = [np.split(m, 3) for m in np.split(masks[..., 0], 5, 0)]
    mask = np.squeeze(np.block(nested_masks))

    return mask

def get_ship_track_array(rgb_array, tf_predictor):

    # reshape and split array prior to inference
    original_size = rgb_array.shape[1::-1] # The size is the inverse of the shape (minus the color channel)
    reshaped_rgb_array = resize_arr(rgb_array, INT_IMG_SIZE[::-1])
    split_rgb_array = split_array(reshaped_rgb_array, IMG_SIZE)

    # call model for inference
    prediction = tf_predictor(data=tf.constant(split_rgb_array, dtype='float32'))

    # stitch and resize output back to original size
    inferred_ship_tracks = combine_masks(prediction['sigmoid'])
    inferred_ship_tracks = resize_arr(inferred_ship_tracks, original_size)
    
    return inferred_ship_tracks

def stretch_dataarray(x:xr.DataArray, stretch='histogram') -> xr.DataArray:
    '''perform a trollimage stretch operation with given method
    stretch is applied in-place and the enhancement attribute is updated'''
     
    xrimage = XRImage(x)
    
    xrimage.stretch(stretch=stretch)

    return xrimage.data

def make_outname(file, outdir, suffix):
    basename = os.path.basename(file[:-4])+suffix

    date = datetime.strptime(
        re.findall("\d+",basename)[0],
        "%Y%j%H%M%S"
    )
    
    date_str = date.strftime("%Y/%m/%d")
    
    outdir = os.path.join(outdir, date_str)
    
    out_name = os.path.join(outdir, basename)

    return out_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model',
        nargs='?',
        help="The path to the trained model to use",
        default='/Lustre/user_scratch/anla/neodaas_requests_21_02/model/2/'
    )
    
    parser.add_argument('--infile',
        help="text file containing a list of files on which to perform inference",
        type=argparse.FileType('r')
    )    
    
    parser.add_argument('--outdir', required=True, help="path to results directory")

    parser.add_argument('--outsuffix', help="suffix of output file", default='_inferred_ship_tracks')
    
    parser.add_argument('--format', nargs='*', choices=['netcdf','geopackage'], default=['geopackage','netcdf'],
        help='list of output formats desired. Available formats are "geopackage" and "netcdf"'
    )
    
    parser.add_argument("--imsize", help="block size magic number",default=448)
    
    parser.add_argument("--batch", type=int, default=5)

    parser.add_argument(
        "--stretch",
        choices=('linear','crude','histogram'),
        default='histogram',
        help='what type of contrast stretching to use, see trollimage.xrimage.XRImage.stretch()'
    )

    parser.add_argument('--contour_level', type=float, help="level of inference contour for geometry output", default=0.2)

#     parser.add_argument('--log-level', dest="log_level", default="INFO", choices=("INFO","DEBUG","WARNING","ERROR","CRITICAL"))
    
    args = parser.parse_args()

    # configure the logging

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

    # global values for splitting up a scene into tiles
    IMG_SIZE = args.imsize
    INT_IMG_SIZE = (5*IMG_SIZE, 3*IMG_SIZE)
    BATCH_SIZE = args.batch

    logger.info(f"set image tiling globals : IMG_SIZE={IMG_SIZE}, INT_IMG_SIZE={INT_IMG_SIZE}")

    # load trained model
    logger.info(f"loading model {args.model}")
    MODEL = tf.saved_model.load(args.model)
    tf_predictor = MODEL.signatures["serving_default"]

    if args.infile is not None:
        logger.info("Reading filelist from {}".format(args.infile))
        infiles = [x.replace("\n","") for x in args.infile.readlines()]
    else:
        infiles = args.infiles

    # main loop through input files...
    file_batches = [infiles[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for i in range(len(infiles)//BATCH_SIZE + (len(infiles) % BATCH_SIZE > 0))]

    # dumb batching with no prefetching
    for file_batch in file_batches:
        
        logger.info(f"processing {len(file_batch)} files")
        
        # out names, to be manipulated further depending on format of output.
        outnames = [make_outname(file, args.outdir, args.outsuffix) for file in file_batch]

        logger.info(f"creating output directories if required")
        for outname in outnames:
            if os.path.isdir(os.path.dirname(outname)) == False:
                logger.debug(f"creating {os.path.dirname(outname)}")
                os.makedirs(os.path.dirname(outname))
        
        # logic to skip if outname exists!... comment out
        # file_batch = [file for (file, name) in zip(file_batch, outnames) if os.path.isfile(name) == True]

        # open datasets, (lazy operation)
        datasets = [xr.open_dataset(file) for file in file_batch]

        # tranpose to put bands last
        datasets = [x.transpose(...,'bands') for x in datasets]

        logger.info("contrast stretching")
        dataarrays = [stretch_dataarray(x['day_microphysics'], stretch=args.stretch) for x in datasets]

        logger.info("load the unlablled data")
        rgb_arrays = [np.array(x.astype('float32').data) for x in dataarrays]

        logger.info("resize the rgb_arrays")
        # PIL will not accept floating point, only ints betwee 0 and 255
        rgb_arrays = [resize_arr((x*255).astype(np.uint8), INT_IMG_SIZE[::-1]) for x in rgb_arrays]

        logger.info("split into expected image shape")
        # recast to float32 between 0 and 1, loss of resolution
        split_rgb_arrays = [split_array((x/255.).astype(np.float32), IMG_SIZE) for x in rgb_arrays]

        # split arrays are lists of arrays, these lists should be added together into one big list...

        split_rgb_arrays = reduce(lambda a,b:a+b, split_rgb_arrays)

        # DONT STACK because that is done later with tf.data.Dataset.batch()
        # logger.info("stack each list of subimages along new first dimension")
        # split_rgb_arrays = [np.stack(x) for x in split_rgb_arrays]
        # logger.debug(f"split rgb array list has elements {split_rgb_arrays[0]}")

        logger.info("convert into list of tf.Tensors ??")
        split_rgb_tensors = [tf.convert_to_tensor(x) for x in split_rgb_arrays]
        logger.debug(f"split_rgb_tensors = {type(split_rgb_tensors)}")

        logger.info("convert into a tf.Dataset?")
        tf_dataset = tf.data.Dataset.from_tensors(split_rgb_tensors)
        logger.debug(tf_dataset)


        # use dataset methods?...
        # tf_dataset.batch().map()

        # granules are split into 15 sub-images, therefore batchsize is 15x BATCHSIZE
        logger.info("perform inference on a single batch")
        for (images,) in tf_dataset.batch(BATCH_SIZE*15):
            logger.debug(f"tf_predictor(images) where images is type {type(images)} with shape {images.shape} ")
            prediction = tf_predictor(images)
            prediction = prediction['sigmoid']    

        logger.info(f"prediction success: type = {type(prediction)} with shape {prediction.shape}")
        # logger.debug(f"prediction = {prediction}")

        # nested list comprehension takes 15 inference arrays and makes them back into one image
        # FIXME @anla : this pattern is ugly and obscure.
        inferred_ship_tracks = [combine_masks(p) for p in [prediction[j*15:(j+1)*15] for j in range(len(prediction)//15)]]

        # reshape back to the dataarray shape,
        # except the bands dimension has been removed hence .shape[:2]
        inferred_ship_tracks = [resize_arr(x, ds.shape[:2]) for (x,ds) in zip(inferred_ship_tracks, dataarrays)]

        logger.info("create new variable in all dataset")
        out_datasets = []
        for (x, ds) in zip(inferred_ship_tracks, datasets):
            ds['ship_tracks'] = xr.Variable(
                dims=['x','y'],
                data=x,
                # FIXME @anla : this gets overwritten when contour to geometry
                attrs={'description':'inferred ship track array'},
            )

            out_datasets.append(ds[['ship_tracks']])
        

        logger.info("creating output file base names")
        logger.info(f"outdir = {args.outdir}")
        logger.debug(f"out_names = {outnames}")
        logger.info(f"number of files to write = {len(out_datasets)}")

        # write to desired output format
        if 'netcdf' in args.format:
            logger.info(f'writing to netcdfs')
            for (ds, out_name) in zip(out_datasets, outnames):
                logger.debug(f"writitng {ds} to files {out_name+'.c'}")
                ds.to_netcdf(
                    out_name + '.nc',
                    encoding={'ship_tracks':{'zlib':True,'complevel':4}}
                )

        if 'geopackage' in args.format:
            logger.info("writing to geopackages")

            for (ds, out_name) in zip(out_datasets, outnames):
                gdf = xr_vectoriser(ds['ship_tracks'], level=args.contour_level)

                # assign attributes from data array to GeoDataFrame
                gdf = gdf.join(pd.DataFrame(columns = ds['day_microphysics'].attrs.keys()))

                for key, value in ds['day_microphysics'].attrs.items():
                    gdf[key] = str(value)   

                # assign custom values from this operation
                gdf['model'] = args.model
                gdf['level'] = args.contour_level
                gdf['description'] = f"multipolygon describing contour at level {args.contour_level} of infered ship track probability"
                            
                # write to geopackage
                logger.debug(f'writing to geopackage {out_name + ".gpkg"}')
                gdf.to_file(
                    out_name + '.gpkg',
                    driver='GPKG',
                    layer='inference'
                )        

    # for file in tqdm(infiles):
            
    #     logger.info("creating output filename \n \
    #     checking if output already exists")
    #     # construct output filename from input filename
    #     basename = os.path.basename(file[:-4])+args.outsuffix

    #     date = datetime.strptime(
    #         re.findall("\d+",basename)[0],
    #         "%Y%j%H%M%S"
    #     )
        
    #     date_str = date.strftime("%Y/%m/%d")
        
    #     outdir = os.path.join(args.outdir, date_str)
        
    #     out_name = os.path.join(outdir, basename)
        
    #     # if the file you want to make exists - skip
    #     if os.path.isfile(out_name) == True:
    #         logger.info("output filename exists,\n \
    #         skipping processing")
    #         continue
        
    #     logger.debug(f"reading from {file}")
        
    #     # read in the data
    #     # open netcdf with xarray
    #     ds = xr.open_dataset(file)
    
    #     # reorder dims to put bands last
    #     ds = ds.transpose('y','x','bands')

    #     # enhance the image with a stretch
    #     # Duncan's model expects histogram stretch
    #     ds['day_microphysics'] = stretch_dataarray(ds['day_microphysics'], stretch=args.stretch)
        
    #     # rgb_array = (np.array(ds['day_microphysics'].data)*255).astype(np.uint8)
    #     rgb_array = ds['day_microphysics'].astype('float32').data

    #     # perform inference on rgb array
    #     inferred_ship_tracks = get_ship_track_array(rgb_array, tf_predictor)

    #     # create directory for output
    #     if os.path.isdir(outdir) == False:
    #         os.makedirs(outdir)
        
    #     # place infered ship track array into xr.DataSet
    #     ds['ship_tracks'] = xr.Variable(
    #             dims=['y','x'],
    #             data=inferred_ship_tracks,
    #             # FIXME @anla : this gets overwritten when contour to geometry
    #             attrs={'description':'inferred ship track array'},
    #         )

    #     # write to desired output format
    #     if 'netcdf' in args.format:
    #         logger.info(f'writing to netcdf {out_name + ".nc"}')
    #         ds[
    #             ['ship_tracks']
    #         ].to_netcdf(
    #             out_name + '.nc',
    #             encoding={'ship_tracks':{'zlib':True,'complevel':4}}
    #         )

    #     if 'geopackage' in args.format:
    #         logger.info("computing shapes")
    #         gdf = xr_vectoriser(ds['ship_tracks'], level=args.contour_level)

    #         # assign attributes from data array to GeoDataFrame
    #         gdf = gdf.join(pd.DataFrame(columns = ds['day_microphysics'].attrs.keys()))

    #         for key, value in ds['day_microphysics'].attrs.items():
    #             gdf[key] = str(value)   

    #         # assign custom values from this operation
    #         gdf['model'] = args.model
    #         gdf['level'] = args.contour_level
    #         gdf['description'] = f"multipolygon describing contour at level {args.contour_level} of infered ship track probability"
                        
    #         # write to geopackage
    #         logger.info(f'writing to geopackage {out_name + ".gpkg"}')
    #         gdf.to_file(
    #             out_name + '.gpkg',
    #             driver='GPKG',
    #             layer='inference'
    #         )