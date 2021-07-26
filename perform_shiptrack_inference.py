#!/usr/bin/env python
"""
Run inference over a dataset using a saved tf model
"""
import matplotlib
matplotlib.use('agg')

import xarray as xr
import rasterio
import shapely
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt


IMG_SIZE = 448
INT_IMG_SIZE = (5*IMG_SIZE, 3*IMG_SIZE)


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
    
    resampler = Image.BILINEAR if bilinear else Image.NEAREST
    
    arr_im = Image.fromarray(arr) 

    arr_im = arr_im.resize(target_size, resample=resampler)
    
    new_arr = np.array(arr_im)
    return new_arr


def combine_masks(masks):

    nested_masks = [np.split(m, 3) for m in np.split(masks[..., 0], 5, 0)]
    mask = np.squeeze(np.block(nested_masks))

    return mask

def get_ship_track_mask(f):

    # netcdf dataset open
    if f.endswith(".nc"):
        ds = xr.open_dataset(f)

        # reorder dims to put bands last
        ds = ds.transpose('y','x','bands')

        # convert to numpy array, move to PIL compatible array...
        data = (np.array(ds['day_microphysics'].data)*255).astype(np.uint8)

    else:
        data  = get_image(f)
    
    original_size = data.shape[1::-1] # The size is the inverse of the shape (minus the color channel)

    reshaped_data = resize_arr(data, INT_IMG_SIZE[::-1])
    
    split_data = split_array(reshaped_data / 255., IMG_SIZE)

    prediction = tf_predictor(data=tf.constant(split_data, dtype='float32'))

    mask = combine_masks(prediction['sigmoid'])
    
    new_mask = resize_arr(mask, original_size)
    
    # return dataset or plain arrays.
    # try except is faster?
    if f.endswith(".nc"):
        # add the mask back into the dataset
        ds['mask'] = xr.Variable(
            dims=['y','x'],
            data=new_mask
        )

        return ds
    
    else:
        return data, new_mask



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model', help="The model directory to use")
    parser.add_argument('infiles', nargs='*')
    parser.add_argument('--infile', help="Input file", type=argparse.FileType('r'))
    parser.add_argument('--show', action='store_true')
    parser.add_argument('-o', '--outfile', help="Output name", default='mask')

    args = parser.parse_args()

    model = tf.saved_model.load(args.model)
    tf_predictor = model.signatures["serving_default"]

    if args.infile is not None:
        print("Reading filelist from {}".format(args.infile))
        infiles = args.infile.readlines()
    else:
        infiles = args.infiles

    for f in infiles:

        outputs = get_ship_track_mask(f)
        
        if type(outputs) == xr.Dataset:
            outputs.to_netcdf(f"{f[:-4]}_{args.outfile}_mask.nc")

        else:
                
            # Get the polygons in image coordinates
            #polys = vectoriser(mask, level=0.2, latlon=False)
            
            # Get the polygons in lat/lon coordinates (FIXME this will require an xarray da with the right coordinates)
            #polys = vectoriser(mask, level=0.2, latlon=True)
            
            # TODO: Save to PostGIS DB?
            
            # Save the mask?
            np.savez_compressed(f"{f[:-4]}_{args.outfile}.npz", mask=mask)
            
            if args.show:
                fig, axs = plt.subplots(figsize=(20, 40))
                axs.imshow(data, vmin=0., vmax=1.)
                im=axs.imshow(mask, alpha=0.5, vmin=0, vmax=1)
                plt.savefig(f"{f[:-4]}_{args.outfile}.png")
