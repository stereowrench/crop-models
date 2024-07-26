import pgeocode 

import pandas as pd
import numpy as np
import xarray as xr
import datetime
from math import floor, ceil
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.constants import convert_temperature
from scipy.ndimage import convolve1d


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
    loca_tasmin_file = 'tasmin.MPI-ESM1-2-HR.ssp245.r1i1p1f1.2015-2044.LOCA_16thdeg_v20220413.nc'
    loca_tasmax_file = 'tasmax.MPI-ESM1-2-HR.ssp245.r1i1p1f1.2015-2044.LOCA_16thdeg_v20220413.nc'
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
    lat_slice = slice(floor(lat)-1, ceil(lat)+1)
    lon_slice = slice(360+floor(lon)-1, 360+ceil(lon)+1)
    time_slice = slice('2022-01-01', '2024-01-01')
    loca_tasmin = loca_tasmin.sel(time=time_slice, lat=lat_slice, lon=lon_slice)
    loca_tasmax = loca_tasmax.sel(time=time_slice, lat=lat_slice, lon=lon_slice)
    
    return (loca_tasmin, loca_tasmax)

def load_zip(zipcode):
    nomi = pgeocode.Nominatim('us')
    postal_code = nomi.query_postal_code(zipcode)
    zip_codes = pd.DataFrame({'ZIP': [zipcode], 
                         'latitude': [postal_code['latitude']],
                         'longitude': [postal_code['longitude']]})
    return zip_codes

def calculate_suitability(tasmin, tasmax, tmin, tmax, topt_min, topt_max, frost_tolerance):
    tmin = tmin + 273.15
    tmax = tmax + 273.15
    topt_min = topt_min + 273.15
    topt_max = topt_max + 273.15
    frost_temperature = 273.15
    # max_consecutive_frost_days = 3
    # max_consecutive_nippy_days = 3
    # max_consecutive_heat_days = 3
    
    # Basic suitability based on absolute thresholds
    suitability = ((tasmin > (tmin - frost_tolerance)) & (tasmax < tmax)).astype(float)  
    frost_tolerant_range = (tasmin >= (tmin - frost_tolerance)) & (tasmin < tmin)

    # # Identify frost days (where tasmin is below the adjusted tmin)
    # nippy_days = ((tasmin < tmin) & (tasmin > 273.15)).astype(int)
    # nippy_days = pd.Series(nippy_days)
    
    # frost_days = (tasmin < frost_temperature).astype(int)
    # frost_days = pd.Series(frost_days)

    # heat_days = (tasmax > tmax).astype(int)
    # heat_days = pd.Series(heat_days)
    
    # # Calculate consecutive frost days using pandas rolling and sum
    # consecutive_frost_days = frost_days.rolling(window=7, min_periods=1).sum().to_numpy()
    # consecutive_nippy_days = nippy_days.rolling(window=7, min_periods=1).sum().to_numpy()
    # consecutive_heat_days = heat_days.rolling(window=7, min_periods=1).sum().to_numpy()
    
    # plot_nippy(consecutive_nippy_days)

    # Convert back to xarray DataArray if needed
    # consecutive_frost_days = xr.DataArray(consecutive_frost_days, coords=[tasmin.time], dims="time")

    # Further refinement based on optimal temperature range
    # heat_stress_factor = np.exp(-((tasmax - topt_max) / (tmax - topt_max))**2)  

    # suitability *= heat_stress_factor  # Reduce suitability based on heat stress
    
    optimal_range = ((topt_min < tasmin) & (tasmin < topt_max) & (topt_min < tasmax) & (tasmax < topt_max))

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

    # old_suitability = suitability.copy()
    
    # suitability = xr.where(
    #     ((consecutive_nippy_days <= max_consecutive_nippy_days) & (consecutive_nippy_days > 0)),
    #     np.where(suitability < 0.2, 0.2, suitability),
    #     suitability
    # )
  
    # suitability = xr.where(
    #     ((consecutive_frost_days <= max_consecutive_frost_days) & (consecutive_frost_days > 0)),
    #     np.where(suitability < 0.2, 0.2, suitability),
    #     suitability
    # )

    # suitability = xr.where(
    #     (consecutive_heat_days > 0)& (consecutive_heat_days <= max_consecutive_heat_days),
    #     np.where(suitability < 0.2, 0.2, suitability),
    #     suitability
    # )
    
    # suitability = xr.where(
    #     ,
    #     np.where(old_suitability < 0.2, 0.3, old_suitability),
    #     0
    # )

    # suitability = xr.where(
    #     (consecutive_heat_days <= max_consecutive_heat_days) & (consecutive_heat_days > 0),
    #     np.where(old_suitability < 0.2, 0.3, old_suitability),
    #     0
    # )

    # suitability = xr.where(
    #     (consecutive_frost_days == 0) & (consecutive_nippy_days == 0) & (consecutive_heat_days == 0),
    #     old_suitability,
    #     suitability
    # )
    
    # Ensure suitability is within 0-1 range
    suitability = suitability.clip(0, 1)

    return suitability

# def plot_nippy(nippy):
    # plt.figure(figsize=(12, 6))
    # plt.plot(nippy)
    # plt.plot(ltime_values, , marker='o', linestyle='-', color='blue')
    # plt.plot(utime_values, utemp_values, marker='o', linestyle='-', color='green')
    # plt.axhline(y = ftmin, color = 'r', linestyle = '-') 
    # plt.axhline(y = ftmax, color = 'r', linestyle = '-') 
    # plt.axhline(y = ftopt_min, color = 'y', linestyle = '-') 
    # plt.axhline(y = ftopt_max, color = 'y', linestyle = '-')
    # plt.show()
    
def smooth_tas(loca_tasmin, loca_tasmax):
    # Define a threshold for outlier detection (e.g., 3 standard deviations)
    # z_threshold = 
    
    # # Calculate z-scores for tasmin and tasmax
    # tasmin_zscores = (loca_tasmin - loca_tasmin.mean(dim='time')) / loca_tasmin.std(dim='time')
    # tasmax_zscores = (loca_tasmax - loca_tasmax.mean(dim='time')) / loca_tasmax.std(dim='time')
    
    # # Mask outliers based on z-score threshold
    # loca_tasmin_smoothed = loca_tasmin.where(np.abs(tasmin_zscores) < z_threshold)
    # loca_tasmax_smoothed = loca_tasmax.where(np.abs(tasmax_zscores) < z_threshold)

    from scipy.signal import savgol_filter, butter, lfilter
    # suit = suit.rolling(time=14, center=True).mean()

    #     suit = xr.apply_ufunc(
    #         savgol_filter,
    #         suit,
    #         kwargs={'window_length': 30, 'polyorder': 2, 'axis': 0},  
    #         dask='parallelized' 
    #     )
    
    loca_tasmin_smoothed =  loca_tasmin.rolling(time=30, center=True).mean().interpolate_na(dim="time")
    loca_tasmax_smoothed =  loca_tasmax.rolling(time=30, center=True).mean().interpolate_na(dim="time")

    num_iterations = 1  # Adjust as needed
    for _ in range(num_iterations):
        loca_tasmin_smoothed = xr.apply_ufunc(savgol_filter, loca_tasmin_smoothed,
            kwargs={'window_length': 30, 'polyorder': 2, 'axis': 0},  
            dask='parallelized' 
        )
        loca_tasmax_smoothed = xr.apply_ufunc(savgol_filter, loca_tasmax_smoothed,
            kwargs={'window_length': 30, 'polyorder': 2, 'axis': 0},  
            dask='parallelized' 
        )
    
    return (loca_tasmin_smoothed, loca_tasmax_smoothed)

def calc_suitability(bolting, loca_tasmin_smoothed, loca_tasmax_smoothed, tmin, tmax, topt_min, topt_max, frost_tolerance):
    # Calculate daily suitability for the entire LOCA dataset using apply_ufunc
    # frost_tolerance = 0
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

def calculate_season_suitability(gmin, gmax, daily_suitability, day_lengths, min_day, max_day):
    growing_season_suitability = {}
    for window_size in range(int(gmin), int(gmax) + 1, 10):
        # Extend the data for circular rolling
        # season_suitability = daily_suitability.rolling(time=window_size, min_periods=window_size, center=False).mean()
        # season_suitability = daily_suitability[::-1].rolling(time=window_size, min_periods=window_size, center=True).mean()[::-1]
        # season_suitability = daily_suitability.rolling(time=7, min_periods=7).mean().fillna(0)
        # season_suitability = daily_suitability.rolling(time=7, min_periods=7).mean().interpolate_na(dim="time")
        season_suitability = daily_suitability.interpolate_na(dim="time")
        # alpha=0.3
        # span=10
        # season_suitability = daily_suitability.rolling_exp(time=span, window_type="span").mean()
        season_suitability = xr.where((day_lengths > min_day) & (day_lengths < max_day), season_suitability, 0)
    
        # Slice out the original data's suitability after the circular rolling
        growing_season_suitability[window_size] = season_suitability.where(season_suitability > 0).fillna(0)
    return growing_season_suitability
    
def calculate_optimal_planting_ranges(growing_season_suitability, lat, lon, cutoff):
    
    def chunk_contiguous_dates(dates):
        """Chunks a pandas Series of dates into contiguous sections.
    
        Args:
            dates (pd.Series): A pandas Series of datetime values.
    
        Returns:
            list: A list of lists, where each inner list represents a contiguous section of dates.
        """
        # Calculate differences between consecutive dates
        diffs = dates.diff()
    
        # Identify boundaries where the difference is not 1 day (indicating a break in the sequence)
        boundaries = diffs.ne(pd.to_timedelta("1 days")).cumsum() 
    
        # Group dates by boundaries and convert each group to a list
        contiguous_sections = dates.groupby(boundaries).apply(list).tolist()
    
        return contiguous_sections
        
    # Find optimal planting date RANGES for each growing season length
    optimal_planting_ranges = {}
    for window_size, suitability in growing_season_suitability.items():
        suitability = suitability.isel(lat=lat,lon=lon)
        # cutoff = xr.where(suitability > 0, suitability, 0).where(suitability < 1, suitability, 0).quantile(0.2)
        # print(f"cutoff {cutoff}")
        # daily_suitability_smoothed = suitability.interpolate_na(dim="time", limit=7).rolling(time=30, center=True).mean()
        daily_suitability_smoothed = suitability
        suitable_dates = daily_suitability_smoothed.where(daily_suitability_smoothed > cutoff).interpolate_na(dim="time",limit=7).dropna(dim="time")

        ranges = []
        current_range = None

        suitable_dates_series = pd.Series(suitable_dates.time.values)
        new_ranges = chunk_contiguous_dates(suitable_dates_series)
    
        for dates in new_ranges:
            ds = dates[0]
            de = dates[-1]
            days = (de - ds).days
    
            if days >= 360:
                ranges.append([pd.Timestamp("20230101"), pd.Timestamp("20231231")])
            elif de.date() >= datetime.date(2024,1,1):
                pass
            elif ds.date() <= datetime.date(2022,1,1):
                pass
            else:
                # print([days, window_size, ds, de])
                if days >= window_size:
                    # print(days - window_size)
                    if days - window_size < 14:
                        pass
                    else:
                        ranges.append([ds, de - pd.Timedelta(window_size + 7, "d")])
    
        optimal_planting_ranges[window_size] = ranges
    
    return optimal_planting_ranges

def plot_planting(loca_tasmin_smoothed, loca_tasmax_smoothed, tmin, tmax, topt_min, topt_max, view_window, optimal_planting_ranges, lat, lon, crop_name, day_lengths, dates, zip_code):
    ltime_values = loca_tasmin_smoothed.isel(lat=lat,lon=lon).time.values
    ltemp_values = loca_tasmin_smoothed.isel(lat=lat,lon=lon).values
    ltemp_values = convert_temperature(ltemp_values, "K", "F")
    
    utime_values = loca_tasmax_smoothed.isel(lat=lat,lon=lon).time.values
    utemp_values = loca_tasmax_smoothed.isel(lat=lat,lon=lon).values
    utemp_values = convert_temperature(utemp_values, "K", "F")
    
    ftmin = convert_temperature(tmin, "C", "F")
    ftmax = convert_temperature(tmax, "C", "F")
    ftopt_min = convert_temperature(topt_min, "C", "F")
    ftopt_max = convert_temperature(topt_max, "C", "F")
    
    # print(temp_values)
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(ltime_values, ltemp_values, linestyle='-', color='blue')
    ax1.plot(utime_values, utemp_values, linestyle='-', color='green')
    ax1.axhline(y = ftmin, color = 'r', linestyle = '-') 
    ax1.axhline(y = ftmax, color = 'r', linestyle = '-') 
    ax1.axhline(y = ftopt_min, color = 'y', linestyle = '-') 
    ax1.axhline(y = ftopt_max, color = 'y', linestyle = '-')

    ax2 = ax1.twinx()
    ax2.plot(dates, day_lengths, color="pink")
    
    ax = plt.gca()
    
    for dates in optimal_planting_ranges[view_window]:
        [d1,d2] = dates
        ax1.add_patch(Rectangle((d1, 40), d2-d1, 95-40))
        ax1.add_patch(Rectangle((d2, 40), datetime.timedelta(days=view_window), 95-40, color="yellow"))
    
    # Create legend elements
    blue_line = mlines.Line2D([0], [0], color='blue', lw=2, label='Lower Temperature Threshold')
    day_line = mlines.Line2D([0], [0], color='pink', lw=2, label='Day length')
    green_line = mlines.Line2D([0], [0], color='green', lw=2, label='Upper Temperature Threshold')
    red_line = mlines.Line2D([0], [0], color='red', lw=2, linestyle='-', label='Absolute Temperature Limit')
    yellow_line = mlines.Line2D([0], [0], color='yellow', lw=2, linestyle='-', label='Optimal Temperature Range')
    blue_patch = mpatches.Patch(color='blue', alpha=0.5, label='Suitable Planting Period')
    yellow_patch = mpatches.Patch(color='yellow', alpha=0.5, label=f'Growing Season Length: {view_window}')
    
    # Add legend
    plt.legend(handles=[blue_line, day_line, green_line, red_line, yellow_line, blue_patch, yellow_patch], loc='upper left', bbox_to_anchor=(1.04, 1))
    
    # Add labels and title
    plt.title(f'Minimum Daily Temperature Over Time ({crop_name}) @ {zip_code}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature (°F)')
    # plt.grid(axis='y', linestyle='--')
    fig.tight_layout()
    # Show the plot
    plt.show()

def plot_suitability(view_window, growing_season_suitability, daily_suitability, lat, lon, crop_name, cutoff):
    window_size = view_window
    daily_suitability_smoothed = growing_season_suitability[window_size].interpolate_na(dim="time", limit=3).rolling(time=30, center=True).mean()
    dssi = daily_suitability_smoothed.isel(lat=lat,lon=lon)
    # cutoff = xr.where(dssi > 0, dssi, 0).where(dssi < 1, dssi, 0).quantile(0.2)
    # cutoff = 0.2
    # print(f"cutoff {cutoff}")
    daily_suitability_smoothed = daily_suitability_smoothed.where(daily_suitability_smoothed > cutoff)
    
    
    plt.figure(figsize=(12, 6))
    plt.axhline(y = cutoff, color = 'y', linestyle = '-') 
    plt.plot(daily_suitability.isel(lat=lat,lon=lon).time.values, daily_suitability.isel(lat=lat,lon=lon), linestyle='-', color='purple')
    plt.plot(daily_suitability_smoothed.isel(lat=lat,lon=lon).time.values, daily_suitability_smoothed.isel(lat=lat,lon=lon), linestyle='-', color='yellow')
    # plt.plot(daily_suitability.isel(lat=lat,lon=lon).time.values, daily_suitability.isel(lat=lat,lon=lon), marker='o', linestyle='-', color='purple')
    # plt.plot(growing_season_suitability[105].isel(lat=lat,lon=lon).time.values, growing_season_suitability[105].isel(lat=lat,lon=lon), marker='o', linestyle='-', color='purple')
    plt.plot(growing_season_suitability[window_size].isel(lat=lat,lon=lon).time.values, growing_season_suitability[window_size].isel(lat=lat,lon=lon),  linestyle='-', color='green')
    # Add labels and title
    plt.title('Suitability ' + crop_name)
    plt.xlabel('Time')
    plt.ylabel('Suitability')  # Assuming tasmin is in Celsius; change if necessary
    plt.grid(axis='y', linestyle='--')
    
    # Show the plot
    plt.show()

def return_first(m1, d1, m2, d2):
    if m1 == m2:
        if d1 > d2:
            return (m2, d2)
        else:
            return (m1, d1)
    elif m1 > m2:
        return (m2, d2)
    else:
        return (m1, d1)
        
def return_last(m1, d1, m2, d2):
    if m1 == m2:
        if d1 < d2:
            return (m2, d2)
        else:
            return (m1, d1)
    elif m1 < m2:
        return (m2, d2)
    else:
        return (m1, d1)

def fix_wraparound(w):
    (m1, d1, m2, d2) = w
    if m1 > 12:
        m1 -= 12
    if m2 > 12:
        m2 -= 12

    return (m1, d1, m2, d2)


def merge_overlapping_monthday_ranges(date_ranges):
    """Converts date ranges to month-day format and merges overlapping ranges.

    Args:
        date_ranges (list): List of date ranges, each as [start_date, end_date].

    Returns:
        list: List of merged month-day ranges, each as (start_month, start_day, end_month, end_day).
    """

    # month_day_ranges = [(date.month, date.day) for start, end in date_ranges for date in [start, end]]
    # month_day_ranges.sort()  

    month_day_ranges = [(start.month, start.day, end.month, end.day) for [start, end] in date_ranges]
    month_day_ranges.sort()
    
    merged_ranges = []
    current_range = month_day_ranges[0]
    
    for start_month2, start_day2, end_month2, end_day2 in month_day_ranges[1:]:
        start_month1, start_day1, end_month1, end_day1 = current_range
        if end_month1 < start_month1:
            end_month1 += 12
        if end_month2 < start_month2:
            end_month2 += 12
        # Range 1 starts before Range 2 ends, and Range 1 ends after Range 2 starts:
        if ((start_month1 < end_month2 or (start_month1 == end_month2 and start_day1 <= end_day2)) and
            (end_month1 > start_month2 or (end_month1 == start_month2 and end_day1 >= start_day2))):
            # Overlapping range - extend the end
            current_range = fix_wraparound(return_first(start_month1, start_day1, start_month2, start_day2) + return_last(end_month1, end_day1, end_month2, end_day2))
            # current_range = (start_month1, start_day1, end_month2, end_day2)
        # Range 1 is fully contained within Range 2:
        elif ((start_month1 >= start_month2 and start_day1 >= start_day2) and
            (end_month1 <= end_month2 and end_day1 <= end_day2)):
            current_range = fix_wraparound((start_month2, start_day2, end_month2, end_day2))
        # Range 2 is fully contained within Range 1:
        elif ((start_month2 >= start_month1 and start_day2 >= start_day1) and
            (end_month2 <= end_month1 and end_day2 <= end_day1)):
            current_range = fix_wraparound((start_month1, start_day1, end_month1, end_day1))
        elif (end_month2 > 12 and end_month1 <= 12):
            start_month1 += 12
            end_month1 += 12
            if ((start_month1 < end_month2 or (start_month1 == end_month2 and start_day1 <= end_day2)) and
                (end_month1 > start_month2 or (end_month1 == start_month2 and end_day1 >= start_day2))):
                # Overlapping range - extend the end
                current_range = fix_wraparound(return_first(start_month1, start_day1, start_month2, start_day2) + return_last(end_month1, end_day1, end_month2, end_day2))
                # current_range = (start_month1, start_day1, end_month2, end_day2)
            # Range 1 is fully contained within Range 2:
            elif ((start_month1 >= start_month2 and start_day1 >= start_day2) and
                (end_month1 <= end_month2 and end_day1 <= end_day2)):
                current_range = fix_wraparound((start_month2, start_day2, end_month2, end_day2))
            # Range 2 is fully contained within Range 1:
            elif ((start_month2 >= start_month1 and start_day2 >= start_day1) and
                (end_month2 <= end_month1 and end_day2 <= end_day1)):
                current_range = fix_wraparound((start_month1, start_day1, end_month1, end_day1))
        elif (end_month2 <= 12 and end_month1 > 12):
            start_month2 += 12
            end_month2 += 12
            if ((start_month1 < end_month2 or (start_month1 == end_month2 and start_day1 <= end_day2)) and
                (end_month1 > start_month2 or (end_month1 == start_month2 and end_day1 >= start_day2))):
                # Overlapping range - extend the end
                current_range = fix_wraparound(return_first(start_month1, start_day1, start_month2, start_day2) + return_last(end_month1, end_day1, end_month2, end_day2))
                # current_range = (start_month1, start_day1, end_month2, end_day2)
            # Range 1 is fully contained within Range 2:
            elif ((start_month1 >= start_month2 and start_day1 >= start_day2) and
                (end_month1 <= end_month2 and end_day1 <= end_day2)):
                current_range = fix_wraparound((start_month2, start_day2, end_month2, end_day2))
            # Range 2 is fully contained within Range 1:
            elif ((start_month2 >= start_month1 and start_day2 >= start_day1) and
                (end_month2 <= end_month1 and end_day2 <= end_day1)):
                current_range = fix_wraparound((start_month1, start_day1, end_month1, end_day1))
        else:
            # New range - add the current range and start a new one
            merged_ranges.append(current_range)
            current_range = fix_wraparound((start_month2, start_day2, end_month2, end_day2))

    merged_ranges.append(current_range)  # Add the last range

    return merged_ranges

def generate_day_lengths(zip_codes):
    import ephem
    import datetime
    
    observer = ephem.Observer()
    observer.lat = str(zip_codes['latitude'].values[0])
    observer.lon = str(zip_codes['longitude'].values[0])
    observer.pressure = 0  # Disable atmospheric refraction for more accurate sunrise/sunset
    sun = ephem.Sun()
    
    start_date = datetime.date(2022, 1, 1)  # Start of the year
    end_date = datetime.date(2024, 1, 1)  # End of the year
    dates = []
    day_lengths = []
    
    current_date = start_date
    while current_date <= end_date:
        observer.date = current_date
        try:
            sunrise = observer.next_rising(sun, use_center=True)  # Use center of the sun for accuracy
            sunset = observer.next_setting(sun, use_center=True)
    
            # Convert sunrise and sunset to local timezone
            sunrise = ephem.localtime(sunrise)#.replace(tzinfo=timezone)
            sunset = ephem.localtime(sunset)#.replace(tzinfo=timezone)
    
            # Ensure sunset is after sunrise
            if sunset < sunrise:
                sunset += datetime.timedelta(days=1)
    
            day_length = sunset - sunrise
            dates.append(current_date)
            day_lengths.append(day_length.total_seconds() / 3600)  # Convert to hours
        except ephem.AlwaysUpError:
            # Handle polar days
            dates.append(current_date)
            day_lengths.append(24)
        except ephem.NeverUpError:
            # Handle polar nights
            dates.append(current_date)
            day_lengths.append(0)
        current_date += datetime.timedelta(days=1)

    day_lengths = np.array(day_lengths)
    
    return day_lengths, dates

def generate_ranges(suit, lat, lon, view_window, show, crop_name, zip_code):
    from scipy.signal import savgol_filter, butter, lfilter
    suit = suit.rolling(time=14, center=True).mean()
    num_iterations = 4  # Adjust as needed
    for _ in range(num_iterations):
        suit = xr.apply_ufunc(
            savgol_filter,
            suit,
            kwargs={'window_length': 30, 'polyorder': 2, 'axis': 0},  
            dask='parallelized' 
        )
    suit = suit.clip(0,1)
    # Define filter parameters
    # order = 4             # Filter order (higher order = sharper cutoff)
    # cutoff_freq = 0.05    # Cutoff frequency (adjust to control smoothing strength)
    # nyquist_freq = 0.5    # Nyquist frequency (half the sampling rate)
    
    # # Normalize cutoff frequency
    # Wn = cutoff_freq / nyquist_freq 
    
    # # Design the Butterworth filter (we'll use a lowpass filter for smoothing)
    # b,a  = butter(order, Wn, btype='lowpass', output='sos')  
    
    # # Apply the filter to the suitability data
    # suit = xr.apply_ufunc(
    #     lfilter, 
    #     b,a,suit,
    #     kwargs={'axis': 0},   # Filter along the 'time' axis
    #     input_core_dims=[[], [], ['time']],  # Core dimension for suitability, none for sos
    #     output_core_dims=[['time']], 
    #     dask='parallelized' 
    # )
    # suit = suit.rolling(time=7, center=True).mean()
    # bef = suit
    # aft = xr.where(suit < 0.01, 0, suit)
    # suit = xr.where(suit > 0, suit, 0).where(suit < 1, suit, 0)
    
    from scipy.signal import find_peaks, peak_widths
    import pandas as pd
    x = suit.isel(lat=lat,lon=lon)
    peaks, _ = find_peaks(x, width=view_window/2)  # Adjust as needed
    # print(peaks)
    peak_times = x.time[peaks].values
    suitability_values = x.values.flatten()
    # suitability_values = suitability_values[~np.isnan(suitability_values)]
    if show:
        plt.figure(figsize=(12, 6))
        plt.title(f"Suitability for {crop_name} @ {zip_code}")
        plt.plot(x.time, x.values)
        plt.plot(peak_times, suitability_values[peaks], "x", color="red", label="Peaks")
    
    widths, width_heights, left_ips, right_ips = peak_widths(
        suitability_values, peaks, rel_height=0.9
    )  #
    
    # Since xarray works with datetime, the peak widths must be converted to the same
    left_edges = [peak_times[i] - pd.to_timedelta(abs(x - left_ips[i]), unit='D') for i, x in enumerate(peaks)]
    right_edges = [peak_times[i] + pd.to_timedelta(abs(x - right_ips[i]), unit='D') for i, x in enumerate(peaks)]
    # Plot peak widths as horizontal lines
    plant_ranges = []
    for left, right, height, widths in zip(left_edges, right_edges, width_heights, widths):
        if height < 0.03:
            pass
        else:
            right_adjusted = right + pd.to_timedelta(-view_window, unit="D")
            if left.date() >= right_adjusted.date():
                pass
            else:
                if right - left >= pd.to_timedelta(365, unit="D"):
                    plant_ranges.append([pd.to_datetime("2023-01-01"), pd.to_datetime("2023-12-31")])
                    plt.hlines(height, left, right, color="green", linestyle="--")
                    break
                else:
                    plant_ranges.append([left, right_adjusted])
            if show:
                plt.hlines(height, left, right, color="green", linestyle="--")
    if show:
        plt.show()
    return plant_ranges

# def get_day_length_from_lat(lat):
    

def all_in_one(zipcode, crop_name, bolting, min_day, max_day):
    frost_tolerance = 0
    zip_codes = load_zip(zipcode)
    loca_tasmin, loca_tasmax = load_temperature_data(zip_codes)
    ecocrop_df = load_ecocrop()
    tmin, tmax, topt_min, topt_max, gmin, gmax = load_crop_variables(ecocrop_df, crop_name)
    zip_codes = add_loca_index(zip_codes, loca_tasmin, loca_tasmax)
    lat, lon = zip_codes['loca_index'].values[0]
    min_days = loca_tasmin.groupby("time.dayofyear").mean("time")
    max_days = loca_tasmax.groupby("time.dayofyear").mean("time")

    start_year = 2022
    end_year = 2024
    time_index = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-1-1", freq="D")
    two_year_tasmin = xr.DataArray(
        coords={"time": time_index, "lat": min_days.lat, "lon": min_days.lon},
        dims=["time", "lat", "lon"],
    )
    
    two_year_tasmax = xr.DataArray(
        coords={"time": time_index, "lat": max_days.lat, "lon": max_days.lon},
        dims=["time", "lat", "lon"],
    )
    
    # Fill values from min_days for each day of year 
    for t in two_year_tasmin.time:
        day_of_year = t.dt.dayofyear
        # month = t.dt.month
        two_year_tasmin.loc[t, :, :] = min_days.sel(dayofyear=day_of_year)
        two_year_tasmax.loc[t, :, :] = max_days.sel(dayofyear=day_of_year)


    
    loca_tasmin_smoothed, loca_tasmax_smoothed = smooth_tas(two_year_tasmin, two_year_tasmax)
    day_lengths, dates = generate_day_lengths(zip_codes)
    daily_suitability = calc_suitability(bolting, loca_tasmin_smoothed, loca_tasmax_smoothed, tmin, tmax, topt_min, topt_max, frost_tolerance)
    # cutoff = 0.1
    # min_day, max_day = get_day_length_from_lat(zip_codes['latitude'])
    growing_season_suitability = calculate_season_suitability(gmin, gmax, daily_suitability, day_lengths, min_day, max_day)
    # optimal_planting_ranges = crop_sim.calculate_optimal_planting_ranges(growing_season_suitability, lat, lon, cutoff)
    ranges = {}
    acc_ranges = {}
    for window_size, suitability in growing_season_suitability.items():
        show = False
        if window_size == next(iter(growing_season_suitability)):
            show = True
        suit = growing_season_suitability[window_size]
        loc_ranges = generate_ranges(suit, lat, lon, window_size, show, crop_name, zipcode)
        acc_ranges[window_size] = loc_ranges

        if len(loc_ranges) > 0:
            ranges[window_size] = merge_overlapping_monthday_ranges(loc_ranges)
        else:
            ranges[window_size] = []
    
    for window_size, suitability in growing_season_suitability.items():
        if window_size == next(iter(growing_season_suitability)):
            plot_planting(loca_tasmin_smoothed, loca_tasmax_smoothed, tmin, tmax, topt_min, topt_max, window_size, acc_ranges, lat, lon, crop_name, day_lengths, dates, zipcode)

    return ranges



    