#!/usr/bin/env python
"""
Create training images, labels and segmentation annotations given a set of shiptrack locations
"""
import os
import glob
import json
import argparse

import pandas as pd
import numpy as np

import satpy
from satpy import Scene
from satpy.writers import to_image
from shapely.geometry import asLineString
from PIL import Image
from multiprocessing import Pool
from functools import partial

satpy.config.set(config_path=['/home/users/dwatsonparris/satpy_config'])


def main(track_file, output_path, experiment_name, modis_path, overwrite=False, n_processes=1, tar_output=False):
    # Create output directories
    path_output_images = output_path + '/' + experiment_name + '/images/'
    os.makedirs(path_output_images, exist_ok=True)

    path_output_points = output_path + '/' + experiment_name + '/points/'
    os.makedirs(path_output_points, exist_ok=True)

    # Get labels
    df = pd.read_csv(track_file, index_col='datetime', parse_dates=['datetime']).sort_index()

    # Mask missing values
    df = df.replace(-999, np.nan)

    # Process each image
    tasks = list(df.groupby('datetime'))
    with Pool(n_processes) as p:
        p.map(partial(process_safe, modis_path=modis_path, overwrite=overwrite,
                      path_output_images=path_output_images,  
                      path_output_points=path_output_points), tasks)

    # tarball directory for easier downloading
    if tar_output:
        os.system("tar -czvf " + output_path + experiment_name + ".tar.gz" + " " + output_path + experiment_name + "/")


def is_daytime(dt, lat, lon):
    from pyorbital import astronomy
    #lons, lats = scene[0, 0].attrs["area"].get_lonlats() # Get the lon/lats for a single point in the corner
    #angle = astronomy.sun_zenith_angle(scene.attrs["start_time"], lons[0], lats[0])
    angle = astronomy.cos_zen(dt, lon, lat)
    print(dt, lat, lon, angle)
    return angle > 0

def process_safe(grp, *args, **kwargs):
    dt, image_tracks = grp
    try:
        process_dt(dt, image_tracks, *args, **kwargs)
    except Exception as e:
        print("Unable to process {}: {}".format(dt, e))


def process_dt(dt, image_tracks, modis_path, overwrite, path_output_images, path_output_points):
    # Fetch corresponding MODIS type (they'll all be the same for this image)
    mtype = image_tracks.mtype[0]
    source = image_tracks.source[0]
    
    #day_str = image_tracks.daytime[0]
    #daytime = day_str == 'D'

    # Get the lat and lon from the first point after removing any NaNs
    lat, lon = image_tracks.dropna().iloc[0][['lat', 'lon']].values

    daytime = is_daytime(dt, lat, lon)
    day_str = 'D' if daytime else 'N'

    img_id = mtype + dt.strftime("%Y%j.%H%M") + day_str
    # Now fetch MODIS file for selected instrument and time
    filename = "{year}/{month:02d}/{day:02d}/*.A???????.{hour}*.hdf".format(
        year=dt.year, month=dt.month, day=dt.day, hour=dt.strftime("%H%M"))
    file02 = glob.glob(modis_path + mtype.upper() + "021KM/collection61/" + filename)  # calibrated radiances
    file03 = glob.glob(modis_path + mtype.upper() + "03/collection61/" + filename)  # geolocation
    # All MODIS L2 files must exist to proceed
    if (len(file02) > 0) and (len(file03) > 0):

        # Create image
        image_file = path_output_images + img_id + '.png'
        if (not overwrite) and os.path.isfile(image_file):
            print('exists: ' + image_file)
            img = Image.open(image_file)  # Read the file anyway to get its size
        else:
            global_scene = (Scene(reader="modis_l1b", filenames=file02 + file03))
            composite = 'day_microphysics' if daytime else 'night_microphysics'
            global_scene.load([composite], resolution=1000)  # This uses channels 1, 20 and 31)
            img = to_image(global_scene[composite])
            img.stretch("histogram")
            img.save(image_file)

        # Create points file
        text_file = open(path_output_points + img_id + '.txt', "w")
        for _, track in image_tracks.groupby('track_id'):
            # Sort the track points since the order matters
            sorted_points = track.sort_values('point_id')
            track_points = sorted_points.modis_cross.astype(str) + " " + sorted_points.modis_along.astype('str')
            text_file.write(track.source[0] + " " + " ".join(track_points) + "\n")
        text_file.close()

        # Create JSON file for labelling
        json_file = open(path_output_images + img_id + '.json', "w")
        # TODO: I could probably do this in one line with geoPandas...
        shapes = []
        for _, track in image_tracks.groupby('track_id'):
            # Be sure to sort the track points first
            points = asLineString(track.sort_values('point_id')[['modis_cross', 'modis_along']].values)
            poly = points.buffer(5)
            shapes.append({"label": "shiptrack",
                           "points": list(poly.exterior.coords),
                           "group_id": None,
                           "shape_type": "polygon",
                           "flags": {}
                           })
        json.dump({"version": "4.0.0", 
                   'flags': {'CSU': source=='CSU', 
                             'OSU': source=='OSU',
                             'SSI': source=='SSI',
                             'KRL': source=='KRL',
                             'EDG': source=='EDG'},
                   'shapes': shapes,
                   "imagePath": img_id + '.png',  # Use the relative path
                   "imageData": None,
                   "imageHeight": img.height,
                   "imageWidth": img.width
                   }, json_file)
    else:
        print('missing: ' + filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('track_file', metavar='track-file')
    parser.add_argument('output_path', metavar='output-path')
    parser.add_argument('experiment_name', metavar='experiment-name')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--tar-output', action='store_true')
    parser.add_argument('-n', '--n-processes', default=1, type=int)
    parser.add_argument('--modis-path', type=str,
                        default='/neodc/modis/data/')

    args = parser.parse_args()

    main(**vars(args))


