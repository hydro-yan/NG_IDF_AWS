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
def extract_AM_data(pixel_file, am_24h_W_veg_file, am_48h_W_veg_file, am_72h_W_veg_file,
                    am_24h_P_file, am_48h_P_file, am_72h_P_file, r_file, td):

    # sim period: year month, day, hours, min, second
    s_date = datetime.datetime(1989, 6,  1,  0,  0, 0)
    e_date = datetime.datetime(2021, 9, 30, 21, 0, 0)

    #----------------------------------------------------------------------------------------------------------
    # 1. read 3-hourly Pixel.CENTER output
    lines = [line.rstrip('\n') for line in open(pixel_file)]
    lines = lines[2:]  # remove the first 2 header lines

    num_3h = len(lines)
    # year, mon, day, hour,  W_veg (mm), P (mm), P_int (mm), SWE (mm)
    output_3h = npnan(num_3h, 8)

    # fill the date
    for t in range(num_3h):
        temp_date = s_date + datetime.timedelta(hours=t*3)
        output_3h[t, 0] = float(temp_date.year)
        output_3h[t, 1] = float(temp_date.month)
        output_3h[t, 2] = float(temp_date.day)
        output_3h[t, 3] = float(temp_date.hour)

    # load the 3h data
    count = 0
    for line in lines:
        item = line.split()
        output_3h[count, 4] = float(item[2])        # W_veg  (mm)
        output_3h[count, 5] = float(item[3]) * 1000 # Precip (mm)
        output_3h[count, 6] = float(item[5]) * 1000 # P_int (mm)
        output_3h[count, 7] = float(item[13]) * 1000# SWE (mm)
        count += 1

    # post-processing: no negatives
    for k in range(4, 8):
        output_3h[output_3h[:, k] <= 0, k] = 0

    # remove the last day (incomplete)
    output_3h_new = output_3h[0:num_3h - 8, :]

    #----------------------------------------------------------------------------------------------------------
    # 2. aggregate 3h into 24h daily
    num_3h = len(output_3h_new[:, 0])
    num_day = int(num_3h / 8)

    # 0-year, 1-mon, 2-day, 3-W_veg, 4-P, 5-P_int, 6-SWE
    output_daily = npnan(num_day, 7)
    for k in range(num_day):
        output_daily[k, 0:3] = output_3h_new[k*8, 0:3]
        output_daily[k, 3]   = np.sum(output_3h_new[k*8:(k+1)*8, 4])
        output_daily[k, 4]   = np.sum(output_3h_new[k*8:(k+1)*8, 5])
        output_daily[k, 5]   = np.sum(output_3h_new[k*8:(k+1)*8, 6])
        output_daily[k, 6]   = output_3h_new[k*8, 7]

    output_24h = output_daily

    # 🔴 IMPORTANT: drop all data before 1989-10-01 (we only want complete water years)
    mask = ~((output_24h[:, 0] == 1989) & (output_24h[:, 1] < 10))
    output_24h = output_24h[mask]

    # recompute num_day after filtering
    num_day = len(output_24h)

    #----------------------------------------------------------------------------------------------------------
    # 2b. Aggregate into 48h and 72h using filtered daily data

    # 48h
    output_48h = npnan(num_day - 1, 6)
    for k in range(num_day - 1):
        # date = second day of the 48h window
        output_48h[k, 0:3] = output_24h[k+1, 0:3]
        output_48h[k, 3]   = output_24h[k+1, 3] + output_24h[k, 3]
        output_48h[k, 4]   = output_24h[k+1, 4] + output_24h[k, 4]
        output_48h[k, 5]   = output_24h[k+1, 5] + output_24h[k, 5]

    # 72h
    output_72h = npnan(num_day - 2, 6)
    for k in range(num_day - 2):
        # date = third day of the 72h window
        output_72h[k, 0:3] = output_24h[k+2, 0:3]
        output_72h[k, 3]   = output_24h[k+2, 3] + output_24h[k+1, 3] + output_24h[k, 3]
        output_72h[k, 4]   = output_24h[k+2, 4] + output_24h[k+1, 4] + output_24h[k, 4]
        output_72h[k, 5]   = output_24h[k+2, 5] + output_24h[k+1, 5] + output_24h[k, 5]

    #----------------------------------------------------------------------------------------------------------
    # 3. extract annual maximum data (AM) by WATER YEAR

    # 3a. Water year for 24h daily data
    # wy_daily: cols = year, mon, day, W, P, Pint, SWE, WY
    wy_daily = npnan(len(output_24h), 8)
    wy_daily[:, :7] = output_24h[:, :7]

    for i in range(len(output_24h)):
        yr  = int(output_24h[i, 0])
        mon = int(output_24h[i, 1])
        wy  = yr + 1 if mon >= 10 else yr
        wy_daily[i, 7] = wy

    # 3b. Water year for 48h and 72h
    water_year_48h = np.zeros(len(output_48h))
    for i in range(len(output_48h)):
        yr  = int(output_48h[i, 0])
        mon = int(output_48h[i, 1])
        water_year_48h[i] = yr + 1 if mon >= 10 else yr

    water_year_72h = np.zeros(len(output_72h))
    for i in range(len(output_72h)):
        yr  = int(output_72h[i, 0])
        mon = int(output_72h[i, 1])
        water_year_72h[i] = yr + 1 if mon >= 10 else yr

    # 3c. List of water years (now first one should be WY1990)
    wy_list = sorted(np.unique(wy_daily[:, 7]))
    num_year = len(wy_list)

    # Initialize AM outputs: cols = year, mon, day, AM
    am_24h_W_veg = npnan(num_year, 4)
    am_24h_P     = npnan(num_year, 4)
    am_24h_P_int = npnan(num_year, 4)

    am_48h_W_veg = npnan(num_year, 4)
    am_48h_P     = npnan(num_year, 4)
    am_48h_P_int = npnan(num_year, 4)

    am_72h_W_veg = npnan(num_year, 4)
    am_72h_P     = npnan(num_year, 4)
    am_72h_P_int = npnan(num_year, 4)

    am_swe       = npnan(num_year, 4)

    # 3d. Compute AM for each WY
    count = 0
    for wy in wy_list:

        # 24h daily records belonging to this water year
        wy_data_24h = wy_daily[wy_daily[:, 7] == wy, :]

        # 48h & 72h records with this water year label
        wy_data_48h = output_48h[water_year_48h == wy, :]
        wy_data_72h = output_72h[water_year_72h == wy, :]

        # --- 24h AM ---
        # W_veg
        am_24h_W_veg[count, 3]   = np.max(wy_data_24h[:, 3])
        am_24h_W_veg[count, 0:3] = wy_data_24h[np.argmax(wy_data_24h[:, 3]), 0:3]

        # P
        am_24h_P[count, 3]       = np.max(wy_data_24h[:, 4])
        am_24h_P[count, 0:3]     = wy_data_24h[np.argmax(wy_data_24h[:, 4]), 0:3]

        # P_int
        am_24h_P_int[count, 3]   = np.max(wy_data_24h[:, 5])
        am_24h_P_int[count, 0:3] = wy_data_24h[np.argmax(wy_data_24h[:, 5]), 0:3]

        # SWE
        am_swe[count, 3]         = np.max(wy_data_24h[:, 6])
        am_swe[count, 0:3]       = wy_data_24h[np.argmax(wy_data_24h[:, 6]), 0:3]

        # --- 48h AM ---
        if len(wy_data_48h) > 0:
            am_48h_W_veg[count, 3]   = np.max(wy_data_48h[:, 3])
            am_48h_W_veg[count, 0:3] = wy_data_48h[np.argmax(wy_data_48h[:, 3]), 0:3]

            am_48h_P[count, 3]       = np.max(wy_data_48h[:, 4])
            am_48h_P[count, 0:3]     = wy_data_48h[np.argmax(wy_data_48h[:, 4]), 0:3]

            am_48h_P_int[count, 3]   = np.max(wy_data_48h[:, 5])
            am_48h_P_int[count, 0:3] = wy_data_48h[np.argmax(wy_data_48h[:, 5]), 0:3]

        # --- 72h AM ---
        if len(wy_data_72h) > 0:
            am_72h_W_veg[count, 3]   = np.max(wy_data_72h[:, 3])
            am_72h_W_veg[count, 0:3] = wy_data_72h[np.argmax(wy_data_72h[:, 3]), 0:3]

            am_72h_P[count, 3]       = np.max(wy_data_72h[:, 4])
            am_72h_P[count, 0:3]     = wy_data_72h[np.argmax(wy_data_72h[:, 4]), 0:3]

            am_72h_P_int[count, 3]   = np.max(wy_data_72h[:, 5])
            am_72h_P_int[count, 0:3] = wy_data_72h[np.argmax(wy_data_72h[:, 5]), 0:3]

        count += 1

    #----------------------------------------------------------------------------------------------------------
    # 4. write AM data to files (R will read and delete them)
    np.savetxt(am_24h_W_veg_file, am_24h_W_veg, fmt='%d %d %d %5.4f')
    np.savetxt(am_48h_W_veg_file, am_48h_W_veg, fmt='%d %d %d %5.4f')
    np.savetxt(am_72h_W_veg_file, am_72h_W_veg, fmt='%d %d %d %5.4f')

    np.savetxt(am_24h_P_file,     am_24h_P,     fmt='%d %d %d %5.4f')
    np.savetxt(am_48h_P_file,     am_48h_P,     fmt='%d %d %d %5.4f')
    np.savetxt(am_72h_P_file,     am_72h_P,     fmt='%d %d %d %5.4f')

    #----------------------------------------------------------------------------------------------------------
    # 5. start generating NG-IDF curves: run R script from Python
    lines = [line.rstrip('\n') for line in open(r_file)]

    # copy and update R script paths
    file = open(r_file, "r")
    replacement = ""
    for line in file:
        line = line.strip()
        if line[0:10] == "file_name1":
            line = "file_name1 <- sprintf('%s/am_%%s_%%s', duration[d], variable[v])"  % (td,)
        if line[0:10] == "file_name2":
            line = "file_name2 <- sprintf('%s/IDF_%%s_%%s', duration[d], variable[v])" % (td,)
        replacement = replacement + line + "\n"
    file.close()
    fout = open(r_file, "w")
    fout.write(replacement)
    fout.close()

    cmd = 'Rscript %s' % (r_file,)
    p = subprocess.call(cmd, stdout=subprocess.PIPE, shell=True)

    return am_24h_P, am_48h_P, am_72h_P, am_24h_W_veg, am_48h_W_veg, am_72h_W_veg, am_swe
