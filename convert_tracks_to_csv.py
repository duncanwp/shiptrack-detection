#---------------------------------------------------------------
# TOP LEVEL CODE
#---------------------------------------------------------------
# PROCESS_TRACK_IMAGES
#
# Description
# General program to read MODIS reflectance and shiptrack hand-logged
# files with some basic plots to illustrate where ship tracks are located
# in the satellite granule. Uses PyTroll to plot/read the MODIS image.
#
# Notes: this code currently crops 250 pixels off the edge of each
#        image. Ship tracks are logged based on the lower left corner
#        starting at x=0 and y=0. The ML algorithm requires the top-left
#        corner to start at x=0 and y=0 therefore we tranform the y-coord
#        for bounding box locations.
#
# Output
# 1) Plots of NIR composite images for each MODIS granule
#       path ---> /images
# 2) Plots of the bounding boxes (used as a sanity check)
#       path ---> /images_bbox/
# 3) Text File: contains the bounding boxes according to DIGITS
#       path --> /labels/
#    (FORMAT: see 
#    https://github.com/NVIDIA/DIGITS/blob/master/digits/extensions/data/objectDetection/README.md)
#
# Example
# python2.7 -i process_track_images.py
#
# History
# 11/12/18, MC: upload initial version of the code to the repo
#---------------------------------------------------------------

#---------------------------------------------------------------
# Libraries
#---------------------------------------------------------------
from subroutines_track_images import *
import datetime
import numpy as np
import os,sys,glob
from netCDF4 import Dataset
from multiprocessing import Pool, Value

import pandas as pd


#---------------------------------------------------------------
# Paths
#---------------------------------------------------------------
#Ship Track Hand-logged Files

path_track_root = '/gws/nopw/j04/aopp/mchristensen/shiptrack/shiptrack_logged_files/combined_v2/'

#---------------------------------------------------------------
# Fetch Ship Track Files
#---------------------------------------------------------------
trackfiles = file_search_tracks( path_track_root )
tfiles = trackfiles['tfiles']
lfiles = trackfiles['lfiles']
fct = len(tfiles)

CT = None

def init(args):
    """ Store the counter ready for later use """
    global CT
    CT = args


def main(i):
    global CT

    # Read track lat/lon locations
    fileInfo = os.path.split(lfiles[i])
    lfilename = fileInfo[0]+'/'+fileInfo[1]
    track_geo = read_osu_shiptrack_file(lfilename)

    # Read track locations from file
    fileInfo = os.path.split(tfiles[i])
    tfilepath = fileInfo[0]
    tfile = fileInfo[1]
    tfilename = tfilepath+'/'+tfile
    track_points = read_osu_shiptrack_file(tfilename)

    # Get the track source
    tSource = tfile[1:4]

    print(tfilename)
    print(tSource)

    # Fetch corresponding MODIS granule
    mtype=''
    if "MOD" in tfile:
        mtype = 'mod'
    if 'MYD' in tfile:
        mtype = 'myd'

    ed=len(tfile)
    YYYY = tfile[ed-16:ed-12]
    DDD  = tfile[ed-12:ed-9]
    HHHH = tfile[ed-8:ed-4]

    print(tfile,'  ',mtype,'  ',YYYY,'  ',DDD,'  ',HHHH)   
    # Now fetch MODIS file for selected instrument and time
    # track_geo['xpt'] is a nested list of points per track
    # I think I need to flatten these and generate track_id and point_id columns. There might be a neat way but I could probably just flatten them with a loop.
    dt = pd.to_datetime(YYYY+DDD+HHHH, format='%Y%j%H%M')
    print(dt)

    track_xpts = np.concatenate(track_points['xpt']).astype(int)
    track_ypts = np.concatenate(track_points['ypt']).astype(int)
    track_lats = np.concatenate(track_geo['ypt']).astype(float)
    track_lons = np.concatenate(track_geo['xpt']).astype(float)
    
    # Create a unique (for this image) track id for each track and copy for all points
    track_ids = np.concatenate([np.repeat(i, len(pts)) for i, pts in enumerate(track_points['xpt'])])
    
    # Create unique point ids for each track so they can keep their order
    point_ids = np.concatenate([np.arange(len(pts)) for i, pts in enumerate(track_points['xpt'])])

    df = pd.DataFrame({"source": tSource, "mtype": mtype, 'datetime': dt,
                       "modis_cross": track_xpts, "modis_along": track_ypts,
                       "lat": track_lats, "lon": track_lons, "track_id": track_ids,
                       "point_id": point_ids})
    return df.set_index('datetime')  # I can't make it with this index otherwise it doens't get broadcast 




p=Pool(initializer=init, initargs=(Value('i', 0),), processes=4)
# Loop over each track file
tracks = p.map(main, range(fct))

df = pd.concat(tracks)
print(df)
df.to_csv('combined_v2.csv')

