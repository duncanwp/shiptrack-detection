import datetime
import numpy as np
import os,sys,glob
from multiprocessing import Pool, Value

import pandas as pd

path_track_root = '/gws/nopw/j04/aopp/mchristensen/shiptrack/shiptrack_logged_files/combined_v2/'

def file_search_tracks(path):
    """
    Specific file search routine to output matching t & l track stat files. 
    There must be a neater way to do this...
    """
    tfiles = file_search( path, '.dat', 't')
    lfiles = file_search( path, '.dat', 'l')

    nfiles=0
    if len(tfiles) <= len(lfiles):
        nfiles = len(tfiles)
        file_set = tfiles
    if len(lfiles) < len(tfiles):
        nfiles = len(lfiles)
        file_set = lfiles

    #Find where they match
    mergedLfiles = []
    mergedTfiles = []
    for i in range(nfiles):
        fileInfo = os.path.split(file_set[i])
        tmpfilepath = fileInfo[0]
        tmpfile = fileInfo[1]
        ed=len(tmpfile)
        Tindex=((np.where(np.array(tfiles) == tmpfilepath+'/t'+tmpfile[1:ed]))[0])[0]
        Lindex=((np.where(np.array(lfiles) == tmpfilepath+'/l'+tmpfile[1:ed]))[0])[0]
        tmpLfile = lfiles[Lindex]
        tmpTfile = tfiles[Tindex]
        mergedLfiles.append(tmpLfile)
        mergedTfiles.append(tmpTfile)

    return mergedTfiles, mergedLfiles


def convert_to_df(tfilename, lfilename):
    """
    Read in a set of tile and map track coordinates and return a single nice dataframe
    """

    tfile = os.path.basename(tfilename)

    # Get the track source
    tSource = tfile[1:4]

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



tfiles, lfiles = trackfiles = file_search_tracks(path_track_root)


p=Pool(processes=4)
# Loop over each track file
tracks = p.starmap(convert_to_df, (tfiles, lfiles))

df = pd.concat(tracks)
print(df)
df.to_csv('combined.csv')

