from glob import glob
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool
from datetime import datetime 
import pandas as pd
import os.path
from skimage.measure import grid_points_in_poly

all_files = list(glob('/Lustre/jupyter_wastelands/jupyter_external_wasteland/anla/shiptrack_results/2020/*/*/*_0.8.gpkg'))

def read_df(f):
    try:
        df = gpd.read_file(f, engine='GPKG', layer='geographic')
        swath_df = gpd.read_file(f, engine='GPKG', layer='swath')
    except Exception as e:
        print(f"{f} raised exception: {e}")
        return None
    
    # Pull the signature from the filename
    signature = os.path.basename(f)[:14]
    # Add this and the raw date to the df
    df['signature'] = signature
    df['time'] = datetime.strptime(signature[1:], "%Y%j%H%M%S")

    df['swath_geometry'] = swath_df['geometry']

    # Do this first since presumably it's quicker than getting the temperatures
    ocean_df = df[df.disjoint(countries.unary_union)].reset_index()
    
    if ocean_df.empty:
        return None

    fname = "_".join(f.split("_")[:2])+"_day_microphysics_.nc"
    brightness_temp = (xr.open_dataarray(fname).sel(bands='B')-273.15)  # Channel 30 (around 10 microns)

    def get_temperature(track):
        import numpy as np
        verts = np.asarray(track.swath_geometry.exterior.coords)
        np_mask = grid_points_in_poly(brightness_temp.shape[::-1], verts).T

        mask = brightness_temp.copy(data=np_mask)
        temperature = float(brightness_temp.where(mask, drop=True).mean().data)
        return temperature

    temps = ocean_df.apply(get_temperature, axis=1)
    ocean_df['temperature'] = temps

    return ocean_df
    
all_shiptracks = process_map(read_df, all_files, max_workers=20, chunksize=4)
all_shiptracks = pd.concat(all_shiptracks)
all_shiptracks.to_file("/Lustre/jupyter_wastelands/jupyter_external_wasteland/anla/shiptrack_results/all_shiptracks_T.gpkg", layer='geographic', driver="GPKG")