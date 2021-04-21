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
from satpy.composites import GenericCompositor
from satpy.writers import to_image
from shapely.geometry import asLineString
import PIL


def main(track_file, output_path, experiment_name, modis_path):
    # Create output directories
    path_output_images = output_path + '/' + experiment_name + '/images/'
    os.makedirs(path_output_images, exist_ok=True)

    path_output_points = output_path + '/' + experiment_name + '/points/'
    os.makedirs(path_output_points, exist_ok=True)

    path_output_json = output_path + '/' + experiment_name + '/annotated/'
    os.makedirs(path_output_json, exist_ok=True)

    # Get labels
    df = pd.read_csv(track_file, index_col='datetime', parse_dates=['datetime'])
    # Process each image
    for dt, image_tracks in df.groupby('datetime'):

        # Fetch corresponding MODIS type (they'll all be the same for this image)
        mtype = image_tracks.mtype[0]

        img_id = mtype + dt.strftime("%Y%j.%H%M")

        # Now fetch MODIS file for selected instrument and time
        filename = "{year/doy/*.A{year}{doy}.{hour}*.hdf".format(
            year=dt.year, doy=dt.doy, hour=dt.time)
        file02 = glob.glob(modis_path + mtype + "021km/" + filename)  # calibrated radiances
        file03 = glob.glob(modis_path + mtype + "03/" + filename)  # geolocation

        # All MODIS L2 files must exist to proceed
        if (len(file02) > 0) and (len(file03) > 0):

            # Create image
            image_file = path_output_images + img_id + '.png'
            if os.path.isfile(image_file):
                print('exists: ' + image_file)
                img = PIL.open(image_file)
            else:
                global_scene = (Scene(reader="modis_l1b", filenames=file02 + file03))
                global_scene.load(['1', '32'], resolution=1000)
                global_scene.load([DatasetID(name='20', modifiers=('nir_emissive',))])
                compositor = GenericCompositor("rgb")
                composite = compositor([global_scene['1'], global_scene['32'], global_scene['20']])
                img = to_image(composite)
                img.stretch_hist_equalize("linear")
                img.save(image_file)

            # Create points file
            text_file = open(path_output_points + img_id + '.txt', "w")
            for _, track in image_tracks.groupby('track_id'):
                track_points = track.modis_cross.astype(str) + " " + track.modis_along.astype('str')
                text_file.write(track.source[0] + " " + " ".join(track_points) + "\n")
            text_file.close()

            # Create JSON file for labelling
            width, height = img.size  # Get the image size for the JSON
            json_file = open(path_output_json + img_id + '.json', "w")
            # TODO: I could probably do this in one line with geoPandas...
            shapes = []
            for _, track in image_tracks.groupby('track_id'):
                points = asLineString(track['modis_cross', 'modis_along'].values)
                poly = points.buffer(5)
                shapes.append({"label": "shiptrack",
                               "points": list(poly.exterior.coords),
                               "group_id": None,
                               "shape_type": "polygon",
                               "flags": {}
                               })
            json.dump({"version": "4.0.0", 'flags': {},
                       'shapes': shapes,
                       "imagePath": image_file,
                       "imageData": None,
                       "imageHeight": height,
                       "imageWidth": width
                       }, json_file)
        else:
            print('missing: ' + filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('track_file', metavar='track-file')
    parser.add_argument('output_path', metavar='output-path')
    parser.add_argument('experiment_name', metavar='experiment-name')
    parser.add_argument('--modis-path', type=str,
                        default='/group_workspaces/cems2/nceo_generic/satellite_data/modis_c61/')

    args = parser.parse_args()

    main(**vars(args))

    # tarball directory for easier downloading
    os.system("tar -czvf " + args.output_path + args.experiment_name + ".tar.gz" + " " + args.output_path +
              args.experiment_name + "/")
