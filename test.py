import xarray as xr
import numpy as np
import gribscan
import intake
import eccodes
import healpy as hp
import matplotlib.pylab as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.feature as cfeature
import pandas as pd
import os
from joblib import Parallel, delayed
from multiprocessing import Pool
import itertools

from concurrent.futures import ProcessPoolExecutor
cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")



ds = cat.IFS['IFS_9-FESOM_5-production']['3D_hourly_healpix512_2020'].to_dask()
unique_times = ds['cc'].time.values  # Get the time values
time_dates = pd.to_datetime(unique_times)

unique_dates = pd.Series(time_dates).dt.strftime('%Y-%m-%d').unique()
plevs=ds.cc.level.values[7:]

lon = np.linspace(60, 100, 400)
lat = np.linspace(0, 45, 400)
lon2, lat2 = np.meshgrid(lon, lat)
   
data_example=ds['cc'].sel(time=unique_dates[0],level=100).mean("time")
pix = hp.ang2pix(
        hp.npix2nside(len(data_example)), theta=lon2, phi=lat2, nest=True, lonlat=True
        )

def extract_data(data, time,level):
    # lons2, lats2 = np.meshgrid(lon, lat)
    # dd = hp.ang2pix(
    #     hp.npix2nside(len(data)), theta=lons2, phi=lats2, nest=True, lonlat=True
    # )
    
    # # Extract values based on the indices
    val = data.values[pix]
    
    # Add a new axis for time
    val = np.expand_dims(val, axis=0)  # Shape will be (1, lat, lon)
    val=val[np.newaxis, ...]
    # Ensure time1 is a 1D array with the correct shape
    time1 = np.array([time])  # Convert to a 1D array with shape (1,)
    level1 = np.array([level])  # Convert to a 1D array with shape (1,)
    val[val == 0] = np.nan
    
    # Create a DataArray with the new shape
    test = xr.DataArray(val, dims=['time', 'level','lat', 'lon'], name='cloud fraction', coords={'lon': lon, 'lat': lat, 'time': time1,'level':level1})
    ds = xr.Dataset({  'cfc': test, })
    newfile = f"cfc_{time}_{level}.nc"
    print(newfile)
    if os.path.exists(newfile):
        os.remove(newfile)
    
    ds.to_netcdf(newfile, mode='w', format="NETCDF4", engine='h5netcdf',group=None, )
    ds.close()
    return ds
#%%


def process_date_level(args):
    date, lev = args
    if lev >= 100:
        mm = ds['cc'].sel(time=date, level=lev).mean("time")
       
        extract_data(mm, time=date, level=lev)
        return lev, mm
    return None

# Create all combinations of dates and levels
date_level_pairs = itertools.product(unique_dates[38:40], plevs)

# Use multiprocessing Pool
with Pool(processes=4) as pool:  # Adjust number of processes as needed
    results = pool.map(process_date_level, date_level_pairs)
