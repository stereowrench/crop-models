import pgeocode 

import pandas as pd
import numpy as np
import xarray as xr
from math import floor, ceil
from scipy.signal import savgol_filter

def load_ecocrop():
    # Replace 'your_ecocrop_file.csv' with your actual file path
    ecocrop_file = 'EcoCrop_DB_UTF8.csv'
    
    # Read the CSV file into a Pandas DataFrame
    # ecocrop_df = pd.read_csv(ecocrop_file, encoding='latin-1')
    ecocrop_df = pd.read_csv(ecocrop_file)
    return ecocrop_df

def load_crop_variables(ecocrop_df, latin_name):
    crop = ecocrop_df[ecocrop_df['ScientificName'] == latin_name]
    tmin = crop['TMIN'].values[0]
    tmax = crop['TMAX'].values[0]
    topt_min = crop['TOPMN'].values[0]
    topt_max = crop['TOPMX'].values[0]
    gmin = crop['GMIN'].values[0]
    gmax = crop['GMAX'].values[0]
    
    return (tmin, tmax, topt_min, topt_max, gmin, gmax)
    
def add_loca_index(zip_codes, loca_tasmin, loca_tasmax):
    def find_nearest_grid_point(lat, lon):
        if lat < 0:
            lat = 360 + lat
        if lon < 0:
            lon = 360 + lon
        distances = np.sqrt((loca_tasmin['lat'] - lat)**2 + (loca_tasmin['lon'] - lon)**2)
    
        return np.unravel_index(distances.argmin(), distances.shape)  # Returns the index of the closest grid point
    zip_codes['loca_index'] = zip_codes.apply(
        lambda row: find_nearest_grid_point(row['latitude'], row['longitude']),
        axis=1
    )
    return zip_codes



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

def calculate_suitability(tasmin, tasmax, tmin, tmax, topt_min, topt_max, frost_tolerance):
    tmin = tmin + 273.15
    tmax = tmax + 273.15
    topt_min = topt_min + 273.15
    topt_max = topt_max + 273.15
    # Basic suitability based on absolute thresholds

    suitability = ((tasmin > (tmin - frost_tolerance)) & (tasmax < tmax)).astype(float)  
    frost_tolerant_range = (tasmin >= (tmin - frost_tolerance)) & (tasmin < tmin)

    # Further refinement based on optimal temperature range
    heat_stress_factor = np.exp(-((tasmax - topt_max) / (tmax - topt_max))**2)  

    suitability *= heat_stress_factor  # Reduce suitability based on heat stress
    
    optimal_range = ((topt_min < tasmin) & (tasmin < topt_max) & (topt_min < tasmax) & (tasmax < topt_max))

    # Linear interpolation between thresholds
    suitability = xr.where(optimal_range, 1.0, suitability)  # Optimal range has perfect suitability (1.0)
    suitability = xr.where(
        (tasmin < topt_min) & ~frost_tolerant_range,
        (tasmin - tmin) / (topt_min - tmin),
        suitability
    )
    suitability = xr.where(
        tasmax > topt_max,
        (tmax - tasmax) / (tmax - topt_max),  # Linear interpolation for values above optimal range
        suitability
    )

    # Ensure suitability is within 0-1 range
    suitability = suitability.clip(0, 1)

    return suitability

def smooth_tas(loca_tasmin, loca_tasmax):
    # Define a threshold for outlier detection (e.g., 3 standard deviations)
    z_threshold = 3
    
    # Calculate z-scores for tasmin and tasmax
    tasmin_zscores = (loca_tasmin - loca_tasmin.mean(dim='time')) / loca_tasmin.std(dim='time')
    tasmax_zscores = (loca_tasmax - loca_tasmax.mean(dim='time')) / loca_tasmax.std(dim='time')
    
    # Mask outliers based on z-score threshold
    loca_tasmin_smoothed = loca_tasmin.where(np.abs(tasmin_zscores) < z_threshold)
    loca_tasmax_smoothed = loca_tasmax.where(np.abs(tasmax_zscores) < z_threshold)
    
    return (loca_tasmin_smoothed, loca_tasmax_smoothed)

def suitability(bolting, loca_tasmin_smoothed, loca_tasmax_smoothed, tmin, tmax, topt_min, topt_max):
    # Calculate daily suitability for the entire LOCA dataset using apply_ufunc
    frost_tolerance = 0
    if bolting:
        tmax = topt_max + 1
    daily_suitability = xr.apply_ufunc(
        calculate_suitability,
        loca_tasmin_smoothed, loca_tasmax_smoothed, tmin, tmax, topt_min, topt_max, frost_tolerance,
        input_core_dims=[["time"], ["time"], [], [], [], [], []],  # Specify that these are scalars
        output_core_dims=[["time"]],  # Output is a single value per input
        vectorize=True,  # Optimize for speed
        dask='parallelized'  # Use Dask for parallel computation (if available)
    )
    return daily_suitability

def calculate_optimal_planting_ranges():
    