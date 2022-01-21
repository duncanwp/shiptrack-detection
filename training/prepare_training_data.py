#!/usr/bin/env python
"""
Create training images, labels and segmentation annotations given a set of shiptrack locations
"""
import os
import glob
import json
import argparse

import pandas as pd
from satpy import Scene
from satpy.writers import to_image
from shapely.geometry import asLineString
from PIL import Image


def main(track_file, output_path, experiment_name, modis_path, overwrite=False):
    # Create output directories
    path_output_images = output_path + '/' + experiment_name + '/images/'
    os.makedirs(path_output_images, exist_ok=True)

    path_output_points = output_path + '/' + experiment_name + '/points/'
    os.makedirs(path_output_points, exist_ok=True)

    path_output_json = output_path + '/' + experiment_name + '/annotated/'
    os.makedirs(path_output_json, exist_ok=True)

    # Get labels
    df = pd.read_csv(track_file, index_col='datetime', parse_dates=['datetime']).sort_index()
    # Process each image
    for dt, image_tracks in df.groupby('datetime'):
        try:
            process_dt(dt, image_tracks, modis_path, overwrite,
                       path_output_images, path_output_json, path_output_points)
        except Exception as e:
            print("Unable to process {}: {}".format(dt, e))


def process_dt(dt, image_tracks, modis_path, overwrite, path_output_images, path_output_json, path_output_points):
    # Fetch corresponding MODIS type (they'll all be the same for this image)
    mtype = image_tracks.mtype[0]
    img_id = mtype + dt.strftime("%Y%j.%H%M")
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
            global_scene.load(['day_microphysics'])  # This uses channels 1, 20 and 31)
            img = to_image(global_scene['day_microphysics'])
            img.stretch("histogram")
            img.save(image_file)

        # Create points file
        text_file = open(path_output_points + img_id + '.txt', "w")
        for _, track in image_tracks.groupby('track_id'):
            track_points = track.modis_cross.astype(str) + " " + track.modis_along.astype('str')
            text_file.write(track.source[0] + " " + " ".join(track_points) + "\n")
        text_file.close()

        # Create JSON file for labelling
        json_file = open(path_output_json + img_id + '.json', "w")
        # TODO: I could probably do this in one line with geoPandas...
        shapes = []
        for _, track in image_tracks.groupby('track_id'):
            points = asLineString(track[['modis_cross', 'modis_along']].values)
            poly = points.buffer(5)
            shapes.append({"label": "shiptrack",
                           "points": list(poly.exterior.coords),
                           "group_id": None,
                           "shape_type": "polygon",
                           "flags": {}
                           })
        json.dump({"version": "4.0.0", 'flags': {},
                   'shapes': shapes,
                   "imagePath": "../images/" + img_id + '.png',  # Use the relative path
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
    parser.add_argument('--modis-path', type=str,
                        default='/neodc/modis/data/')

    args = parser.parse_args()

    main(**vars(args))

    # tarball directory for easier downloading
    os.system("tar -czvf " + args.output_path + args.experiment_name + ".tar.gz" + " " + args.output_path +
              args.experiment_name + "/")
