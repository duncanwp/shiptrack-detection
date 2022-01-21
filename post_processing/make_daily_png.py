"""
Create daily image of shiptracks given month directory

"""

from os import pathsep
import datashader
from datashader.utils import lnglat_to_meters
import xarray as xr
import argparse
import colorcet
from glob import glob
from datetime import datetime
import dask.dataframe as dd
import re
from tqdm import tqdm

def dataarray_to_points(da:xr.DataArray) -> dd.DataFrame:
    return dd.from_dask_array(
        da.data.flatten(),
        ['shiptrack']
    ).join(
        dd.from_dask_array(
            da['longitude'].data.flatten(),
            ['lon']
        )
    ).join(
        dd.from_dask_array(
            da['latitude'].data.flatten(),
            ['lat']
        )
    ).dropna()

def day_shiptracks_to_png(day, level, latmin=-80, latmax=80):
    

    dt = datetime.strptime(re.findall("\d{4}/\d{2}/\d{2}", day)[0], "%Y/%m/%d")
    
    filename = f"shiptrack_web_mercator_level_{int(level*100)}_%_{dt.year}_{dt.strftime('%j')}.png"

    files = glob(day+"/*.nc")
    
    # open dataarrays and cast to points
    dataframes = []
    for file in files:
        
        try:
            dataframes.append(
                dataarray_to_points(
                    xr.open_dataarray(file,chunks={'x':512,'y':512})
                )
            )
        except:
            print(f'FAILD: {file}')
            
    cdf = dd.concat(dataframes)
        
    # apply threshold
    cdf = cdf[cdf['shiptrack']>level]
    
    # cut off the poles for mercator projection
    cdf = cdf[cdf['lat']>latmin]
    cdf = cdf[cdf['lat']<latmax]

    # datashader canvas
    canvas = datashader.Canvas(
        plot_width=360*8,
        plot_height=180*8,
        x_range=(-20037508.342789244,20037508.342789244),
        y_range=(lnglat_to_meters(0,latmin)[1], lnglat_to_meters(0,latmax)[1])
    )
    
    cdf = cdf.dropna()
    
    cdf['x'], cdf['y'] = lnglat_to_meters(cdf['lon'], cdf['lat'])

    img = datashader.tf.shade(canvas.points(cdf, x='x', y='y'), cmap=colorcet.kr)
    
    img.to_pil().save('/Lustre/user_scratch/anla/shiptrack_request_21_02/results/video_frames/{dt.year}/'+filename)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path',type=str)
    parser.add_argument('--level',type=float, default=0.8)
    parser.add_argument('--latmin', default=-80, type=float)
    parser.add_argument('--latmax', default=80, type=float)

    args = parser.parse_args()

    # get a list of days in the month
    dirs = glob(args.path+'/*')

    # for each day, create a png and store
    for d in tqdm(dirs):
        day_shiptracks_to_png(d, args.level, args.latmin, args.latmax)    
