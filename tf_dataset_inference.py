#!/usr/bin/env python

"""
BATCHED INFERENCE SCRIPT FOR SHIPTRACK DETECTION

This script performs batched segmentation of shiptracks.
It is designed to be submitted to MAGEO as a job via SLURM,
see the SBATCH script `sbatch_tf_data_inference.sh`

The weird was to identify this design from more simple designs
such as serial inference in a naive for loop.

Author

    Angus Laurenson
    Plymouth Marine Laboratory
    anla@pml.ac.uk

Description

    Script takes netcdf files of AQUA MODIS day microphysics
    composite granules and applied model prediction on them.
    The inference array is then written to a netcdf file in
    the form of the granule, with geolocation information.

    Tensorflow.data.Datasets is used to maximise throughput
    by enabling batched inference, at the expense of parallel
    post-processing. Tensorflow blocks multiprocessing, meaning
    inference is run in parallel but post-processing must be 
    in series if done in the same script. Therefore, post-processing 
    has been commented out and is done by a seperate script.

    The script is self-contained, with a sequence of functions
    defined in the first section that are for use with tf.data
    pipeline.

USAGE:
    
    python tf_dataset_inference.py \
        --infile <file_of_input_filepaths> \
        --model <path_to_model> \
        --batch_size <int> \
        --contour_level <float(s)>

"""

# import the packages required
import tensorflow as tf
import xarray as xr
import numpy as np
from datetime import datetime
from glob import glob
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from trollimage.xrimage import XRImage
import argparse
import os

def nan_mask(func):
    # decorator that fills nans, runs func, then reapplies nan mask
    
    def wrapper(x):
        y = func(tf.where(tf.math.is_nan(x), x=tf.zeros_like(x), y=x))        
        return tf.where(tf.math.is_nan(x), np.nan, y)
    
    return wrapper

def rgb_netcdf_to_tensor(ncfile) -> tf.Tensor:
    # load netcdf as xr.dataarray and convert to tf.Tensor
    # transpose RGB channels to last dimension and downcast
    # to float32. Tensorflow expects this I think
    
    ds = xr.load_dataarray(ncfile, engine='netcdf4')
#     ds = ds.transpose(...,'bands').astype('float32')
    
    # use trollimage histrogram stretch
    # consistent with original training data
    # FIXME @anla : each image transformed independently
    # causes discrepencies between granules
    xrimg = XRImage(ds)
    xrimg.stretch_hist_equalize(approximate=False)
    
    # transpose to put RGB dim last
    data = xrimg.data.transpose(...,'bands').astype('float32')
    
    return tf.convert_to_tensor(data, name='ncfile')

def ncfiles_to_dataset(ncfiles) -> tf.data.Dataset:
    # given list of netcdf files, return a tf Dataset

    # this syntax returns a callable that itself returns iterable
    # necessary for use with tensorflow's from_generator function
    gen = lambda: (rgb_netcdf_to_tensor(x) for x in ncfiles)

    return tf.data.Dataset.from_generator(gen, output_types='float')

def resize_image(X:tf.Tensor, newshape) -> tf.Tensor:
    return tf.image.resize(
        images = X,
        size = newshape
    )

def cut_squares(X):
    # nested split concat calls
    # to break image into 5 x 3 grid
    # of 448 * 448 sub-images
    # FIXME @anla : this doesn't work on batched?
    return tf.concat(
        tf.split(
            tf.concat(
                tf.split(X, 5, axis=1),
                axis=0
            ),
            3,
            axis=2)
        ,
        axis=0
    )

def stitch_up(X):
    # nested concat split calls to
    # stitch back together a image
    # from a 5 x 3 grid of sub-images
    return tf.concat(
            tf.split(
                tf.concat(
                    tf.split(
                        X,
                        3, axis=0
                    ),
                    axis=2
                ),
                5,
                axis=0
            ),
            axis=1
        )

def prediction(x):
    return tf_predictor(x)['sigmoid']

def resize_image_back(x):
    return tf.image.resize(x, (2040,1354))

def ncfile_shapes_to_dataset(ncfiles) -> tf.data.Dataset:
    # given list of netcdf files, return a tf Dataset
    # that contains just their shape (x,y)
    # used for zipping together original shapes with transformed data
    # in order to reshape back to original size for geolocation

    # this syntax returns a callable that itself returns iterable
    # necessary for use with tensorflow's from_generator function
    gen = lambda: (tf.convert_to_tensor(xr.open_dataarray(x).shape[1:]) for x in ncfiles)

    return tf.data.Dataset.from_generator(gen, output_types='int32', output_shapes=(2))

def crop_image_back(x, shape):
    # crop image back to original size
    # this is tricky. using with zipped dataset
    
    # FIXME @anla : this just applied the first shape,
    # maybe the tensor needs split up into a list of tensors
    # and then each one reshaped differently (or does it?)
    return tf.image.crop_to_bounding_box(
        x, # image,
        0, # offset_height,
        0, # offset_width,
        shape[0],
        shape[1]# target_height, target_width
    )

def inference_to_dataset(x,ncfile,attrs={})->xr.Dataset:
    """given inference array x and source netcdf file
    return the inference array as a geolocated xr.Dataset
    """
    
    ds = xr.open_dataset(ncfile)
    ds = ds.transpose(...,'bands')
    
    x = tf.image.crop_to_bounding_box(
        x,0,0,ds['y'].size, ds['x'].size
    )
    
    x = x.numpy().squeeze()
    #print(x.shape)
    ds['shiptracks'] = xr.Variable(
        dims=('y','x'),
        data=x[:,:,0],
        attrs={'description':'inferred shiptrack array'}
    )
    
    # drop the original data variable, keep only results (shiptracks)
    ds = ds[['shiptracks']]
    ds.attrs.update(attrs)
    
    return ds

def xarray_contour_to_geodataframe(da:xr.DataArray, level:float)->gpd.GeoDataFrame:
    """Find contour of geolocated dataarray object and return
    geodataframe with geometry columns for unlabelled (swath)
    and labelled geographic coordinates (lat, lon)"""
    
    # get the contours at the given level
    # filter out those contours that can't make valid polygon
    from skimage import measure
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
    
    return gpd.GeoDataFrame({'swath_geometry':swath_geometry,'geographic_geometry':geographic_geometry})


@nan_mask
def prediction(x):
    # prediction with nans masked out
    return tf_predictor(x)['sigmoid']

def gen_global_attributes():
    """create a time-stamped dict with metadata"""

    return {'author':'Angus Laurenson',
        'email':'anla@pml.ac.uk',
        'institution':'Plymouth Marine Laboratory',
        'group':'NEODAAS',
        'project':'NEODAAS request 21_02, Watson-Paris shiptrack detection',
        'dated':datetime.now().strftime("%Y-%m-%d"),
        'description':'Ship-track clouds infered by a RESNET like model, trained by Duncan Watson-Paris of Oxford University'
    }

def is_file_valid(filename)->bool:
    """check if file should go into pipeline"""
    
    try:
        ds = xr.open_dataarray(filename)
        return True
    except:
        return False

if __name__ == '__main__':

    # list the files that need inference on
    parser = argparse.ArgumentParser()

    parser.add_argument('--infile', help="Input file", type=argparse.FileType('r'))
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--contour_level', nargs='*', type=float, help="level of inference contour for geometry output", default=0.8)
    parser.add_argument('--model',type=str, default='/Lustre/user_scratch/anla/shiptrack_request_21_02/model/2')

    args = parser.parse_args()

    # load the model as Duncan's example
    MODEL = tf.saved_model.load(args.model)
    tf_predictor = MODEL.signatures["serving_default"]

    # get a list of filenames
    ncfiles = [x.replace("\n","") for x in args.infile.readlines()]
    
    # filter for valid files
    #print('filtering input files...')
    ncfiles = list(filter(is_file_valid, ncfiles))

    ncfiles.sort()

    print(f"{len(ncfiles)} valid input files")

    # prepare filenames and path of inferred files
    outfiles = [x.replace(
        'day_microphysics_','inferred_shiptracks'
        ).replace(
            '/data/','/results/'
        ) for x in ncfiles]

    # filter out existing filenames to avoid
    # repeating effort
    # FIXME @anla : all nan files going in make all nan coming out...
    infiles, outfiles = zip(*[(x,y) for (x,y) in zip(ncfiles,outfiles) if os.path.isfile(y) is False])

    print(f"{len(ncfiles)-len(infiles)} files already processed")

    # magic numbers to satisfy the model
    IMAGE_SHAPE = 448*5,448*3
    N_PARALLEL=4


    # # # # START OF TENSORFLOW PIPELINE # # # #

    # pipeline appears to speed up inference by factor of 2 compared to naive loop

    # use generator pattern
    # interleaving a loading function fails due to netcdf4 parallel not installed
    tfds = ncfiles_to_dataset(infiles)

    # take the orginal shapes into a dataset
    tfds_shapes = ncfile_shapes_to_dataset(infiles)

    # padded batch handles different shaped files (2030 or 2040 in first axis)
    tfds = tfds.padded_batch(args.batch_size, padded_shapes=(2040,1354,3),
                            padding_values=0.0, drop_remainder=True)

    # resize to size that can be evenly split
    tfds = tfds.map(lambda x:tf.image.resize(x, IMAGE_SHAPE), num_parallel_calls=tf.data.AUTOTUNE)#tf.data.AUTOTUNE)

    # cut into squares are stack
    tfds = tfds.map(cut_squares, num_parallel_calls=tf.data.AUTOTUNE).prefetch(1)#tf.data.AUTOTUNE)

    # loop through the netcdf files and dataset
    # store inference array
    for i, x in enumerate(tqdm(tfds)):

        # perform inference on the batch
        y_batch = prediction(x)

        # stitch images back together
        # must be done on a batch
        y_batch = stitch_up(y_batch)

        # # # # POST PROCESSING # # # #        
        # FIXME @anla : tensorflow breaks other process based parallelism 
        # for example, the multiprocessing package. This slows post-processing

        batch_slice = slice(i*len(y_batch),(i+1)*len(y_batch))
        for y, infile, outfile in zip(y_batch, infiles[batch_slice], outfiles[batch_slice]):
            
            y = tf.image.resize(y, (2040,1354))
            #print(f"resized up {y.shape}")
            
            # crop to original shape and form
            # into a dataset with geographic coords
            ds = inference_to_dataset(y, infile)

            ds.attrs.update(gen_global_attributes())

            ds.attrs['model_path'] = args.model
        
            # create output file if needed
            if os.path.isdir(os.path.dirname(outfile)) is False:
                os.makedirs(os.path.dirname(outfile))
            
            # store inference array in netcdf4
            ds.to_netcdf(
                outfile,
                encoding={'shiptracks':{'zlib':True,'complevel':4}}
            )

            # FIXME @anla : post-processing is done in series
            # tensorflow monopolises multiprocessing