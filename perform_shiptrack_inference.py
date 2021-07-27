#!/usr/bin/env python
"""
Run inference over a dataset using a saved tf model

@ anla
    
    What operations can we perform on the GPU to give speedup?
    * image enhancement? (currently histogram stretching is done before)
    * contour finding?
    * coordinate lookup for converting contours from x,y to lon,lat?


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

# Use of globals like this I do not like...
# IMG_SIZE = 448
# INT_IMG_SIZE = (5*IMG_SIZE, 3*IMG_SIZE)


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
    return geopandas.GeoDataFrame({"geometry": polys})


def xr_vectoriser(da, level=0.2, loop='ListComp'):
    """
    input: arr -> a square 2D binary mask array
    output: polys -> a list of vectorised polygons
    
    From https://gist.github.com/Lkruitwagen/26c6ba8cadbfd89ab42f36f6a3bbdd35
    """    
    from skimage import measure
        
    contours = measure.find_contours(da.data, level)
    
    # list comprehension
    # 34.9 s ± 107 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # 2792 contours processed
    if loop == 'ListComp':
        
        polygons = [
            Polygon([(float(ds['longitude'][x,y]),
                      float(ds['latitude'][x,y])) \
                     for (x,y) in c.astype(int)])\
            for c in filter(lambda x:len(x) > 3,contours)]

    # normal for loop
    # 34.7 s ± 185 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # 2792 contours processed
    elif loop == 'ForLoop':

        polygons = []

        for c in tqdm(contours):
            
            lon_lat_pairs = []
            
            for (x,y) in c.astype(int):
                lon = float(ds['longitude'][x,y])
                lat = float(ds['latitude'][x,y])
                
                lon_lat_pairs.append((lon,lat))
            
        # catch exception when lat_lon_pairs
        # has len < 3
        try:
            polygon = Polygon(lon_lat_pairs)
            polygons.append(polygon)

        except AttributeError:
            pass
        
    # 10.2 s ± 66.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each
    # 2791 contours processed
    elif loop == 'MapPool':
        
        with Pool(4) as p:
            polygons = p.map(
                make_polygon,
                contours
            )

    return gpd.GeoDataFrame({"geometry": polygons})



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

    parser.add_argument('--outsuffix', help="suffix of output file", default='_inferred_ship_tracks')
    parser.add_argument('--outextension', help="file extension", default='.nc')

    parser.add_argument("--imsize", help="block size magic number",default=448)
    
    parser.add_argument('-l','--level', help="level of inference contour for geometry output", default=0.2)

    args = parser.parse_args()
    

    # globals 
    IMG_SIZE = args.imsize
    INT_IMG_SIZE = (5*IMG_SIZE, 3*IMG_SIZE)

    # load (compile?) trained model
    MODEL = tf.saved_model.load(args.model)
    tf_predictor = MODEL.signatures["serving_default"]

    if args.infile is not None:
        print("Reading filelist from {}".format(args.infile))
        infiles = args.infile.readlines()
    else:
        infiles = args.infiles

    # main loop through input files...
    for file in tqdm(infiles):

        # read in the data
        rgb_array = load_data(file)

        # preprocess the data
        rgb_array = pre_processing(rgb_array)

        # perform inference on rgb
        inferred_ship_tracks = get_ship_track_array(rgb_array, tf_predictor)
      
        # construct output filename from input filename
        stem = os.path.stem(file)
        out_name = os.path.join(stem, args.outsuffix, args.outextension)

        # NETCDF input
        if file[-3:] in [".nc", "hdf"]:
            ds = xr.open_dataset(file)
            ds['ship_tracks'] = xr.Variable(
                dims=['y','x'],
                data=inferred_ship_tracks,
                attrs={'description':'inferred ship track array'},
            )

            if args.outextension == ".nc":
                ds.to_netcdf(out_name)              

            # SHAPE FILE OUTPUT      
            elif args.outextension == "shp":
                # assume shapefile output
                # get geometry and add to GeoDataFrame
                gdf = xr_vectoriser(ds['ship_tracks'], level=args.level, loop='MapPool')

                # add metadata to the GeoDataFrame to describe
                # @anla : this is a bit hacky
                df = pd.DataFrame(index=[0])
                for key, value in ds.attrs.items():
                    if type(value).__name__ == "datetime":
                        value = pd.to_datetime(value)
                        
                    if type(value) == tuple:
                        value = ", ".join(value)
                    try:
                        df[key]=value
                    except:
                        print(f"{key} invalid")
                
                # add some more detail
                df['model'] = args.model
                df['level'] = args.level
                df['description'] = f"multipolygon describing contour at level {args.level} of infered ship track probability"

                # drop invalid column
                df = df.drop(columns='area')

                # join dataframe with geodataframe
                gdf = gdf.join(df.astype(str))

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
