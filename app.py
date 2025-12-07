from flask import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import subprocess
import os
import os.path
from dhsvm_idf import generate_ng_idf
from extract_AM import extract_AM_data
import tempfile
import shutil
from gen_figures import generate_fig


# ----------------------------------------------------------------------------------------------------------------------------
# Helper: Create np.nan array
def npnan(x,y):
    #this function creates the np.nan 2d-array (np.nan should be float)
    array_2d = np.zeros((x,y), float) 
    array_2d[:] = np.nan
    return array_2d

# Helper: Read IDF file
def read_idf(file, data):
    # file is the IDF path
    # data is a np array (3x7)
    lines = [line.rstrip('\n') for line in open(file)]  
    count = 0
    for line in lines:
        item = line.split() 
        for k in range(len(item)):
            data[count, k] = float(item[k])  
        count += 1  
    return data

# Helper: Pack tuple results into a dictionary for easier template rendering
def structure_results(results, year_offset=0):
    (
        P_IDF_24h, P_IDF_48h, P_IDF_72h,
        NG_IDF_24h, NG_IDF_48h, NG_IDF_72h,
        fig24_code, fig48_code, fig72_code,
        am_24h_P, am_48h_P, am_72h_P,
        am_24h_W, am_48h_W, am_72h_W,
        am_swe
    ) = results

    # Helper to process date arrays: round, convert to int, and add offset
    def process_dates(arr_in, offset):
        # Slice first 3 cols (Year, Month, Day)
        dates = np.round(arr_in[:, :3]).astype(int)
        # Add offset to Year column (index 0)
        dates[:, 0] += offset
        return dates

    return {
        "P_24h": P_IDF_24h, "P_48h": P_IDF_48h, "P_72h": P_IDF_72h,
        "NG_24h": NG_IDF_24h, "NG_48h": NG_IDF_48h, "NG_72h": NG_IDF_72h,
        "fig24": fig24_code, "fig48": fig48_code, "fig72": fig72_code,
        "am_24h_P": am_24h_P, "am_48h_P": am_48h_P, "am_72h_P": am_72h_P,
        "am_24h_W": am_24h_W, "am_48h_W": am_48h_W, "am_72h_W": am_72h_W,
        "am_swe": am_swe,
        
        # Apply the year offset here
        "am_24h_P_date": process_dates(am_24h_P, year_offset),
        "am_48h_P_date": process_dates(am_48h_P, year_offset),
        "am_72h_P_date": process_dates(am_72h_P, year_offset),
        "am_24h_W_date": process_dates(am_24h_W, year_offset),
        "am_48h_W_date": process_dates(am_48h_W, year_offset),
        "am_72h_W_date": process_dates(am_72h_W, year_offset),
        "am_swe_date":   process_dates(am_swe,   year_offset)
    }





# ----------------------------------------------------------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/NG_IDF", methods=["GET", "POST"])

def NG_IDF():

    if request.method == "POST":

        input_data = dict()

        # user input
        for i in range(1, 11):
            key = f"Value{i}"
            input_data[f"value{i}"] = float(request.form[key])

        lat = input_data['value1']
        lon = input_data['value2']
        scenario = request.form.get("scenario", "historical")

        # ---------------------------------------------
        # BRANCH: HISTORICAL OR FUTURE
        # ---------------------------------------------

        if scenario == "historical":

            # Run historical Daymet workflow (same as before)
            results = get_NG_IDF(input_data, forcing_type="Daymet")

            data = structure_results(results)

            # Render historical results using existing out.html
            return render_template("out.html",
                                   lat=lat, lon=lon,
                                   **data) # Unpack all keys in data to template variables

        else:

            # Run Multi-Scenario Workflow
            # 1. Historical Baseline (WRF Historical)
            res_hist = get_NG_IDF(input_data, forcing_type="WRF_historical")
            hist_data = structure_results(res_hist, year_offset=0)

            # 2. Future Medium (WRF Medium)
            res_med = get_NG_IDF(input_data, forcing_type="WRF_medium")
            med_data = structure_results(res_med, year_offset=44)

            # 3. Future High (WRF High)
            res_high = get_NG_IDF(input_data, forcing_type="WRF_high")
            high_data = structure_results(res_high, year_offset=44)


            # 4. Build Comprehensive Summary Table
            # Indices: 2yr=0, 5yr=30, 10yr=40, 25yr=46, 50yr=48, 100yr=49, 500yr=50
            aris = [2, 5, 10, 25, 50, 100, 500]
            indices = [0, 30, 40, 46, 48, 49, 50]
            durations = ["24h", "48h", "72h"]
            
            summary_rows = []

            for dur in durations:
                key = f"NG_{dur}" # e.g. "NG_24h"
                
                # Get base arrays for this duration
                base_arr = hist_data[key]
                med_arr = med_data[key]
                high_arr = high_data[key]

                for ari, idx in zip(aris, indices):
                    # Values (Row 0 is magnitude)
                    val_hist = base_arr[0, idx]
                    val_med = med_arr[0, idx]
                    val_high = high_arr[0, idx]

                    # Diffs
                    diff_med = val_med - val_hist
                    diff_high = val_high - val_hist
                    
                    # Percents
                    pct_med = (diff_med / val_hist * 100) if val_hist != 0 else 0.0
                    pct_high = (diff_high / val_hist * 100) if val_hist != 0 else 0.0

                    summary_rows.append({
                        "duration": dur,
                        "ari": ari,
                        "hist": val_hist,
                        "med": val_med,
                        "med_diff": diff_med,
                        "med_pct": pct_med,
                        "high": val_high,
                        "high_diff": diff_high,
                        "high_pct": pct_high
                    })

            return render_template("out_future.html",
                                   lat=lat, lon=lon,
                                   hist=hist_data,
                                   med=med_data,
                                   high=high_data,
                                   summary_rows=summary_rows)

    return render_template("NG_IDF.html")





# ----------------------------------------------------------------------------------------------------------------------------
# read the values and start processing 
def get_NG_IDF(input_data, forcing_type="Daymet"):

    # 0-lat, 1-lon, 2-LAI, 3-Height (m), 4-land cover fraction, 5-Land cover type, 6-Rain LAI Multiplier, 7-Snow LAI Multiplier, 8-Max Snow Intercp (m), 9-Snow Intercp Effi
    value = []  
    for i in range(1, 11):
        value.append(input_data[f"value{i}"])

    # print(value)  # this line doesnot work in docker container 



    # # -----------------------------------------------------
    # # load met data; test docker volume file
    # met_path = '/met/data_25_-120'   # mount docker volume path "/met"
    # lines = [line.rstrip('\n') for line in open(met_path)]    
    # met_data =npnan(2,4)
    # count = 0
    # for line in lines:
    #   item = line.split() 
    #   for k in range(len(item)):
    #       met_data[count, k] = float(item[k])
    #   count += 1
    # print(met_data)


    # -----------------------------------------------------
    # run DHSVM, output the Pixel.Center file here
    bas_par = np.array([value[0], value[1], value[2], value[3], value[4], value[5]])
    adv_par = np.array([value[6], value[7], value[8], value[9]])    

    # ----------------------------------------------
    # location for the forcing file, need update when moving to AWS
    # Define Met Paths
    folder_map = {
        "Daymet": "./met/Daymet/",
        "WRF_historical": "./met/WRF_historical/",
        "WRF_medium": "./met/WRF_medium/",
        "WRF_high": "./met/WRF_high/"
    }
    met_path = folder_map.get(forcing_type, "./met/Daymet/")




    with tempfile.TemporaryDirectory() as td:
        # --------------------------------------------------------------------------------------------------
        # create files inside the temporal file, need to define them before using them
        pixel_file = os.path.join(td, 'Pixel.CENTER')       # e.g., /tmp/tmp3rnej769/Pixel.CENTE
        #config_file_tmp = os.path.join(td, 'Input.Snotel.T4_tmp')
        #config_file = os.path.join(td, 'Input.Snotel.T4')
        tmp_fil1 = os.path.join(td, 'Mass.Final.Balance')
        tmp_fil2 = os.path.join(td, 'Mass.Balance')
        tmp_fil3 = os.path.join(td, 'Stream.Flow')
        tmp_fil4 = os.path.join(td, 'Streamflow.Only')
        tmp_fil5 = os.path.join(td, 'Aggregated.Values') 

        am_24h_W_veg_file = os.path.join(td, 'am_24h_W_veg')
        am_48h_W_veg_file = os.path.join(td, 'am_48h_W_veg')
        am_72h_W_veg_file = os.path.join(td, 'am_72h_W_veg')

        am_24h_P_file = os.path.join(td, 'am_24h_P')
        am_48h_P_file = os.path.join(td, 'am_48h_P')
        am_72h_P_file = os.path.join(td, 'am_72h_P')

        IDF_24h_P_file = os.path.join(td, 'IDF_24h_P')
        IDF_48h_P_file = os.path.join(td, 'IDF_48h_P')
        IDF_72h_P_file = os.path.join(td, 'IDF_72h_P')

        IDF_24h_W_veg_file = os.path.join(td, 'IDF_24h_W_veg')
        IDF_48h_W_veg_file = os.path.join(td, 'IDF_48h_W_veg')
        IDF_72h_W_veg_file = os.path.join(td, 'IDF_72h_W_veg')

        fig_24h_file  = os.path.join(td, 'fig_24h.png')
        fig_48h_file  = os.path.join(td, 'fig_48h.png')
        fig_72h_file  = os.path.join(td, 'fig_72h.png')
        

        config_file = generate_ng_idf(bas_par, adv_par, td, met_path, forcing_type)
        # print(config_file)
    
        os.system('dos2unix ' + config_file)
        #os.system('./dhsvm/no_sat_dump/DHSVM3.2 ' + config_file)
        os.system('./dhsvm/no_sat_dump/DHSVM3.2 ' + config_file + ' > /dev/null 2>&1')
        #os.system('ls -l ' + td)

        # create r file in the temp folder and copy the lines into the new temp r file
        r_file = os.path.join(td, 'get_IDF.R')
        shutil.copy2('./get_IDF.R', r_file)

        # ----------------------------------------------------
        # pixel_file = './example_output/Pixel.CENTER'   
        pixel_file = os.path.join(td, 'Pixel.CENTER')




        am_24h_P, am_48h_P, am_72h_P, am_24h_W_veg, am_48h_W_veg, am_72h_W_veg, am_swe = extract_AM_data(pixel_file, am_24h_W_veg_file, am_48h_W_veg_file, am_72h_W_veg_file, am_24h_P_file, am_48h_P_file, am_72h_P_file, r_file, td)

        # load the P-IDF and NG-IDF data from R output
        # 1st row: estimated value; 2nd row: 5% quantile; 3rd row: 95% quantile
        # column: 51 probabilities (0.50 to 0.99 plus 0.998)
        # 2-yr (0), 5-yr (30), 10-yr (40), 25-yr (46), 50-yr (48), 100-yr (49), 500-yr (50)
        num_pro = 51;
        P_IDF_24h = npnan(3, num_pro);  NG_IDF_24h = npnan(3, num_pro);
        P_IDF_48h = npnan(3, num_pro);  NG_IDF_48h = npnan(3, num_pro);
        P_IDF_72h = npnan(3, num_pro);  NG_IDF_72h = npnan(3, num_pro);

        # load the data
        P_IDF_24h = read_idf(IDF_24h_P_file, P_IDF_24h)
        P_IDF_48h = read_idf(IDF_48h_P_file, P_IDF_48h)
        P_IDF_72h = read_idf(IDF_72h_P_file, P_IDF_72h)

        NG_IDF_24h = read_idf(IDF_24h_W_veg_file, NG_IDF_24h)
        NG_IDF_48h = read_idf(IDF_48h_W_veg_file, NG_IDF_48h)
        NG_IDF_72h = read_idf(IDF_72h_W_veg_file, NG_IDF_72h)


        # gen figures
        fig24_code, fig48_code, fig72_code = generate_fig(P_IDF_24h, P_IDF_48h, P_IDF_72h, NG_IDF_24h, NG_IDF_48h, NG_IDF_72h, fig_24h_file, fig_48h_file, fig_72h_file)


    P_IDF_24h = np.round(P_IDF_24h, 2); P_IDF_48h = np.round(P_IDF_48h, 2); P_IDF_72h = np.round(P_IDF_72h, 2)
    NG_IDF_24h = np.round(NG_IDF_24h, 2); NG_IDF_48h = np.round(NG_IDF_48h, 2); NG_IDF_72h = np.round(NG_IDF_72h, 2)

    am_24h_P = np.round(am_24h_P, 2); am_48h_P = np.round(am_48h_P, 2); am_72h_P = np.round(am_72h_P, 2)
    am_24h_W_veg = np.round(am_24h_W_veg, 2); am_48h_W_veg = np.round(am_48h_W_veg, 2); am_72h_W_veg = np.round(am_72h_W_veg, 2)

    am_swe = np.round(am_swe, 2)



    return P_IDF_24h, P_IDF_48h, P_IDF_72h, NG_IDF_24h, NG_IDF_48h, NG_IDF_72h, fig24_code, fig48_code, fig72_code, am_24h_P, am_48h_P, am_72h_P, am_24h_W_veg, am_48h_W_veg, am_72h_W_veg, am_swe



if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")

