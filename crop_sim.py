import pgeocode 

import pandas as pd
import numpy as np
import xarray as xr
from math import floor, ceil

def load_ecocrop():
    pass

def load_crop_variables(latin_name):
    pass

def generate_near():
    # Replace 'your_loca_tasmin_file.nc' and 'your_loca_tasmax_file.nc' with your actual file paths
    loca_tasmin_file = 'tasmin.MPI-ESM1-2-HR.ssp245.r1i1p1f1.2015-2044.LOCA_16thdeg_v20220413.s_east.nc'
    loca_tasmax_file = 'tasmax.MPI-ESM1-2-HR.ssp245.r1i1p1f1.2015-2044.LOCA_16thdeg_v20220413.s_east.nc'
    ds_tasmin = xr.open_dataset(loca_tasmin_file)
    ds_tasmax = xr.open_dataset(loca_tasmax_file)
    loca_tasmin = ds_tasmin['tasmin']
    loca_tasmax = ds_tasmax['tasmax']
    near_loca_tasmin = loca_tasmin.sel(time=slice('2022-01-01', '2024-01-01'))
    near_loca_tasmin.to_netcdf("tasmin_near.nc")
    near_loca_tasmax = loca_tasmax.sel(time=slice('2022-01-01', '2024-01-01'))
    near_loca_tasmax.to_netcdf("tasmax_near.nc")

def load_temperature_data(zip_codes):
    

    loca_tasmin_file = "tasmin_near.nc"
    loca_tasmax_file = "tasmax_near.nc"
    
    # Open the NetCDF files using xarray
    ds_tasmin = xr.open_dataset(loca_tasmin_file)
    ds_tasmax = xr.open_dataset(loca_tasmax_file)
    
    # Extract the 'tasmin' and 'tasmax' variables
    loca_tasmin = ds_tasmin['tasmin']
    loca_tasmax = ds_tasmax['tasmax']
    
    # Optionally, select a subset of the data if needed
    # (e.g., by time period or geographic region)
    lat = zip_codes["latitude"].iloc[0]
    lon = zip_codes["longitude"].iloc[0]
    lat_slice = slice(floor(lat), ceil(lat))
    lon_slice = slice(360+floor(lon), 360+ceil(lon))
    time_slice = slice('2022-01-01', '2024-01-01')
    loca_tasmin = loca_tasmin.sel(time=time_slice, lat=lat_slice, lon=lon_slice)
    loca_tasmax = loca_tasmax.sel(time=time_slice, lat=lat_slice, lon=lon_slice)
    
    return (loca_tasmin, loca_tasmax)

def load_zip(zipcode):
    nomi = pgeocode.Nominatim('us')
    postal_code = nomi.query_postal_code("33483")
    zip_codes = pd.DataFrame({'ZIP': ['33483'], 
                         'latitude': [postal_code['latitude']],
                         'longitude': [postal_code['longitude']]})
    return zip_codes