#--------------------------------------------------------------------------------------------------------------
# This is a Python3: anaconda script


#--------------------------------------------------------------------------------------------------------------
import numpy as np
import os
import copy
import math
import re
import csv
import shutil
import datetime
import pandas as pd
import scipy
import os.path
import subprocess



#--------------------------------------------------------------------------------------------------------------
def npnan(x,y):
    #this function creates the np.nan 2d-array (np.nan should be float)
    array_2d = np.zeros((x,y), float) 
    array_2d[:] = np.nan
    return array_2d

#--------------------------------------------------------------------------------------------------------------
def finddate(year,month,day,var):
    #given year month day and find the index number in the var array
    #if the year/month/day is not in the var, return nan value
    #input year/month/day: int
    var = var[:,:3]        #only contains the date
    var = var.astype(int)  #float to int
    date = np.array([year,month,day])
    temp = np.where(np.all(var==date, axis=1))  #return a tuple
    try:
        m = int(temp[0])
    except:
        m = np.nan
        print('cannot find the date, return NaN value')
    return m



#--------------------------------------------------------------------------------------------------------------
# User inputs
def extract_AM_data(pixel_file, am_files_dict, r_file, td, forcing_type="Daymet"):
    """
    Extract Annual Maximum data from DHSVM output
    
    Parameters:
    -----------
    pixel_file : str
        Path to Pixel.CENTER file
    am_files_dict : dict
        Dictionary with keys like 'am_1h_W_veg', 'am_1h_P', etc.
    r_file : str
        Path to R script
    td : str
        Temporary directory
    forcing_type : str
        Type of forcing data: "Daymet", "WRF_historical", "WRF_medium", "WRF_high", 
        "CESM_hist_LE2", "CESM_hist_LE4", etc.
    
    Returns:
    --------
    Dictionary with AM data for each duration
    """
    
    # Determine time step and date range based on forcing type
    if forcing_type == "Daymet":
        time_step = 3  # 3-hourly
        s_date = datetime.datetime(1989, 6, 1, 0, 0, 0)
        e_date = datetime.datetime(2021, 9, 30, 21, 0, 0)
        durations = [24, 48, 72]  # hours
    else:  # WRF or CESM (all are hourly)
        time_step = 1  # 1-hourly
        if "futu" in forcing_type or "medium" in forcing_type or "high" in forcing_type:
            # Future scenarios
            s_date = datetime.datetime(2033, 5, 31, 16, 0, 0)
            e_date = datetime.datetime(2065, 9, 30, 15, 0, 0)
        else:
            # Historical scenarios
            s_date = datetime.datetime(1989, 5, 31, 16, 0, 0)
            e_date = datetime.datetime(2021, 9, 30, 15, 0, 0)
        durations = [1, 3, 6, 12, 24, 48, 72]  # hours

    #----------------------------------------------------------------------------------------------------------
    # 1. read time-step Pixel.CENTER output
    lines = [line.rstrip('\n') for line in open(pixel_file)]
    lines = lines[2:]  # remove the first 2 header lines

    num_steps = len(lines)
    # year, mon, day, hour,  W_veg (mm), P (mm), P_int (mm), SWE (mm)
    output_timestep = npnan(num_steps, 8)

    # fill the date
    for t in range(num_steps):
        temp_date = s_date + datetime.timedelta(hours=t*time_step)
        output_timestep[t, 0] = float(temp_date.year)
        output_timestep[t, 1] = float(temp_date.month)
        output_timestep[t, 2] = float(temp_date.day)
        output_timestep[t, 3] = float(temp_date.hour)

    # print(output_timestep[:3, :], flush=True)
    # print(output_timestep[-3:, :], flush=True)


    # load the timestep data
    count = 0
    for line in lines:
        item = line.split()
        output_timestep[count, 4] = float(item[2])        # W_veg  (mm)
        output_timestep[count, 5] = float(item[3]) * 1000 # Precip (mm)
        output_timestep[count, 6] = float(item[5]) * 1000 # P_int (mm)
        output_timestep[count, 7] = float(item[13]) * 1000# SWE (mm)
        count += 1

    # post-processing: no negatives
    for k in range(4, 8):
        output_timestep[output_timestep[:, k] <= 0, k] = 0

    #----------------------------------------------------------------------------------------------------------
    # 2. Aggregate data for different durations
    aggregated_data = {}
    
    for duration in durations:
        steps_per_duration = duration // time_step
        
        if duration == time_step:
            # No aggregation needed for 1-hour data when time_step is 1
            aggregated_data[duration] = output_timestep.copy()
        else:
            # Aggregate to the specified duration
            num_periods = len(output_timestep) // steps_per_duration
            # 0-year, 1-mon, 2-day, 3-hour, 4-W_veg, 5-P, 6-P_int, 7-SWE
            agg_output = npnan(num_periods, 8)
            
            for k in range(num_periods):
                start_idx = k * steps_per_duration
                end_idx = (k + 1) * steps_per_duration
                
                # Use the timestamp of the last timestep in the aggregation window
                agg_output[k, 0:4] = output_timestep[end_idx - 1, 0:4]
                # Sum W_veg, P, P_int
                agg_output[k, 4] = np.sum(output_timestep[start_idx:end_idx, 4])
                agg_output[k, 5] = np.sum(output_timestep[start_idx:end_idx, 5])
                agg_output[k, 6] = np.sum(output_timestep[start_idx:end_idx, 6])
                # SWE: use the last value
                agg_output[k, 7] = output_timestep[end_idx - 1, 7]
            
            aggregated_data[duration] = agg_output

    # 🔴 IMPORTANT: Filter to complete water years (Oct 1 onwards for first year)
    for duration in durations:
        data = aggregated_data[duration]
        first_year = int(data[0, 0])
        # Drop all data before October 1st of the first year
        mask = ~((data[:, 0] == first_year) & (data[:, 1] < 10))
        aggregated_data[duration] = data[mask]

    #----------------------------------------------------------------------------------------------------------
    # 3. Extract annual maximum data (AM) by WATER YEAR
    
    am_results = {}
    
    for duration in durations:
        data = aggregated_data[duration]
        
        # Add water year column
        water_years = np.zeros(len(data))
        for i in range(len(data)):
            yr = int(data[i, 0])
            mon = int(data[i, 1])
            water_years[i] = yr + 1 if mon >= 10 else yr
        
        # Get unique water years
        wy_list = sorted(np.unique(water_years))
        num_year = len(wy_list)
        
        # Initialize AM outputs: cols = year, mon, day, AM
        am_W_veg = npnan(num_year, 4)
        am_P = npnan(num_year, 4)
        am_P_int = npnan(num_year, 4)
        am_swe = npnan(num_year, 4)
        
        # Compute AM for each water year
        count = 0
        for wy in wy_list:
            wy_data = data[water_years == wy, :]
            
            if len(wy_data) > 0:
                # W_veg
                am_W_veg[count, 3] = np.max(wy_data[:, 4])
                am_W_veg[count, 0:3] = wy_data[np.argmax(wy_data[:, 4]), 0:3]
                
                # P
                am_P[count, 3] = np.max(wy_data[:, 5])
                am_P[count, 0:3] = wy_data[np.argmax(wy_data[:, 5]), 0:3]
                
                # P_int
                am_P_int[count, 3] = np.max(wy_data[:, 6])
                am_P_int[count, 0:3] = wy_data[np.argmax(wy_data[:, 6]), 0:3]
                
                # SWE (only for 24h duration to avoid duplication)
                if duration == 24:
                    am_swe[count, 3] = np.max(wy_data[:, 7])
                    am_swe[count, 0:3] = wy_data[np.argmax(wy_data[:, 7]), 0:3]
            
            count += 1
        
        am_results[f'{duration}h'] = {
            'W_veg': am_W_veg,
            'P': am_P,
            'P_int': am_P_int
        }
        
        if duration == 24:
            am_results['swe'] = am_swe

    #----------------------------------------------------------------------------------------------------------
    # 4. Write AM data to files (R will read and delete them)
    for duration in durations:
        dur_key = f'{duration}h'
        np.savetxt(am_files_dict[f'am_{dur_key}_W_veg'], am_results[dur_key]['W_veg'], fmt='%d %d %d %5.4f')
        np.savetxt(am_files_dict[f'am_{dur_key}_P'], am_results[dur_key]['P'], fmt='%d %d %d %5.4f')

    # print(f"After writing AM files — ls -l {td}", flush=True)
    # subprocess.run(['ls', '-l', td], check=False)

    # print out the first and last 3 lines of the AM files
    # for key, label in [('am_24h_P', 'am_24h_P'), ('am_72h_W_veg', 'am_72h_W_veg')]:
    #     path = am_files_dict.get(key)
    #     if not path or not os.path.exists(path):
    #         print(f"{label}: skip (missing): {path}", flush=True)
    #         continue
    #     with open(path, 'r', errors='replace') as f:
    #         am_lines = f.readlines()
    #     print(f"{label} ({path}) — first 3 lines:", flush=True)
    #     for row in am_lines[:3]:
    #         print(row.rstrip('\n\r'), flush=True)
    #     print(f"{label} — last 3 lines:", flush=True)
    #     for row in am_lines[-3:]:
    #         print(row.rstrip('\n\r'), flush=True)




    #----------------------------------------------------------------------------------------------------------
    # 5. Start generating NG-IDF curves: run R script from Python
    # The R script needs to run in the temp directory where the AM files are located
    
    print(f"Running R script in directory: {td}", flush=True)
    print(f"R script path: {r_file}", flush=True)
    
    # List AM files before running R
    print(f"AM files in {td}:", flush=True)
    subprocess.run(['ls', '-lh', td], check=False)
    
    # Run R script with the temp directory as working directory
    # This way R can find the am_*h_* files
    cmd = f'cd {td} && Rscript {r_file}'
    print(f"Running command: {cmd}", flush=True)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"R script stdout:\n{result.stdout}", flush=True)
    print(f"R script stderr:\n{result.stderr}", flush=True)
    print(f"R script exit code: {result.returncode}", flush=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"R script failed with exit code {result.returncode}")
    
    # List IDF files after running R
    print(f"IDF files in {td} after R:", flush=True)
    subprocess.run(['ls', '-lh', td], check=False)

    # print out the first and last 3 lines of the IDF files (written under td by get_IDF.R)
    # for key, label in [('IDF_24h_P', 'IDF_24h_P'), ('IDF_72h_W_veg', 'IDF_72h_W_veg')]:
    #     path = os.path.join(td, key)
    #     if not os.path.exists(path):
    #         print(f"{label}: skip (missing): {path}", flush=True)
    #         continue
    #     with open(path, 'r', errors='replace') as f:
    #         idf_lines = f.readlines()
    #     print(f"{label} ({path}) — first 3 lines:", flush=True)
    #     for row in idf_lines[:3]:
    #         print(row.rstrip('\n\r'), flush=True)
    #     print(f"{label} — last 3 lines:", flush=True)
    #     for row in idf_lines[-3:]:
    #         print(row.rstrip('\n\r'), flush=True)


    # Return results as a dictionary
    return am_results
