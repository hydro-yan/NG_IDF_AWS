
import os
import numpy as np
#import matplotlib.pyplot as plt
import re
import pandas as pd

## Part I - Update Config File

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    
def find_lat_lon(input_lat, input_lon):
    
    if (input_lat > 48.90625 + 1.0/32) or input_lat < (25.15625 + 1.0/32):
        print('lat outside boundary')

    if (input_lon > -67.09375 - 1.0/32) or input_lon < (-124.59375 - 1.0/32):
        print('lat outside boundary')
        
    all_lat = np.arange(25.15625, 48.90625 +1.0/16, 1.0/16)
    all_lon = np.arange(-124.59375, -67.09375 + 1.0/16, 1.0/16)
    
    lat = find_nearest(all_lat, input_lat)  #locate nearest Livneh grid center - lat
    lon = find_nearest(all_lon, input_lon)  #locate nearest Livneh grid center - lon
    
    lat_lon = "{:.5f}".format(lat) + '_' + "{:.5f}".format(lon)
    
    return lat_lon
   
def find_cover(cover_idx):
    if cover_idx == 1:
        return "Open"
    elif cover_idx == 2:
        return "Evergreen"
    elif cover_idx == 3:
        return "Deciduous"
    elif cover_idx == 4:
        return "Mixed"
    elif cover_idx == 5:
        return "Crop"
    elif cover_idx == 6:
        return "Grass"
    elif cover_idx == 7:
        return "Shrub"
    elif cover_idx == 8:
        return "Pasture"
    elif cover_idx == 9:
        return "Wetland"
    else:
        return "Cover Type Not Found, please insert number between 1-9. "


   
def update_config_file_basic(cover_type = [], lat_lon=[], fc = -9999, lai = -9999, o_height = -9999, 
                             snow_item=[], veg_item=[], td = [], met_path = []):
    
    config_file_tmp = os.path.join(td, 'Input.Snotel.T4_tmp')
    # Input Templates 
    with open("./example_config/Input.Snotel.T4", 'r') as file:
        # read a list of lines into data
        data = file.readlines()

    # Output Config 
    outF = open(config_file_tmp,"w")
            
    ################################
    # Snow Parameters
    ################################

    rthresh = float(snow_item['Train'])
    sthresh = float(snow_item['TSnow'])

    amax = float(snow_item['amax'])
    acclamb = float(snow_item['acc_lmbda'])
    mellamb = float(snow_item['melt_lmbda']) 
    
    sw_cap = float(veg_item['SW_cap'])

    dft_over_lai = float(veg_item['LAI_Over'])
    dft_under_lai = float(veg_item['LAI_Under'])
    dft_over_height = float(veg_item['H_Over'])
    dft_under_height = float(veg_item['H_Under']) 
    lai_multi = np.array(veg_item.iloc[: , -12:])[0]
    
    ################################
    # Update Parameters
    ################################
    for line in data:
        ###########   Met ##############
        if 'Station File 1' in line:
#             tmp = re.split(r'\s+',line)
#             tmp[4] = met_path + 'data_' +lat_lon
#             tmp2 = ' '.join(tmp[:])
            tmp2 = 'Station File 1 =  ./met/data_47.71875_-123.71875'
            outF.write(tmp2)
            outF.write("\n")  
                
         ###########   Snow  ##############
        
        elif 'Rain Threshold' in line:
            if rthresh != -9999:
                #print line
                tmp = re.split(r'\s+',line)
                #print tmp[:]

                tmp[3] = str(rthresh)
                tmp2 = ' '.join(tmp[:])
               # print tmp2
                outF.write(tmp2)
                outF.write("\n")  
            else:
                outF.write(line)

        elif 'Snow Threshold' in line:
            if sthresh != -9999:
                #print line
                tmp = re.split(r'\s+',line)
                #print tmp[:]

                tmp[3] = str(sthresh)
                tmp2 = ' '.join(tmp[:])
               # print tmp2
                outF.write(tmp2)
                outF.write("\n")  
            else:
                outF.write(line)
        elif 'Fresh Snow Albedo' in line:
            if amax != -9999: 
                tmp = re.split(r'\s+',line)
                tmp[4] = str('%.5f' % amax)
                tmp2 = ' '.join(tmp[:])
                outF.write(tmp2)
                outF.write("\n")  
                del tmp, tmp2
            else:
                outF.write(line)

        elif 'Albedo Accumulation Lambda' in line:
            if acclamb != -9999: 
                tmp = re.split(r'\s+',line)
                tmp[4] = str('%.5f' % acclamb)
                tmp2 = ' '.join(tmp[:])
                outF.write(tmp2)
                outF.write("\n")  
                del tmp, tmp2
            else:
                outF.write(line)

        elif 'Albedo Melting Lambda' in line:
            if mellamb != -9999: 
                tmp = re.split(r'\s+',line)
                tmp[4] = str('%.5f' % mellamb)
                tmp2 = ' '.join(tmp[:])
                outF.write(tmp2)
                outF.write("\n")  
                del tmp, tmp2
            else:
                outF.write(line)
        
         ###########   Vegetation ##############
        elif 'Fractional Cover' in line:
            #print line
            tmp = re.split(r'\s+',line)
            #print tmp[:]
            if (tmp[4]!='') and (tmp[4]!='#0.00'):
                tmp[4] = str(fc)
                tmp2 = ' '.join(tmp[:])
                #print tmp2
                outF.write(tmp2)
                outF.write("\n")  
            else:
                outF.write(line)
                
        elif 'Overstory Monthly LAI ' in line:          
            tmp = re.split(r'\s+',line)

            if lai != -9999:
                o_lai = lai
            else:
                o_lai = dft_over_lai

            for mon in range(5,17):
                tmp[mon] = str('%.4f' % (o_lai* lai_multi[mon-5]))

            tmp2 = ' '.join(tmp[:])
            outF.write(tmp2)
            outF.write("\n")
        elif 'Understory Monthly LAI ' in line:
            tmp = re.split(r'\s+',line)
            for mon in range(5,17):
                tmp[mon] = str('%.4f' % (dft_under_lai* lai_multi[mon-5]))
                
            tmp2 = ' '.join(tmp[:])
            #print tmp2
            outF.write(tmp2)
            outF.write("\n")
            
        elif 'Height' in line and('Reference Height' not in line ):
            tmp = re.split(r'\s+',line)
            
            if o_height != -9999:
                o_height = o_height
            else: 
                o_height = dft_over_height
            
            tmp[3] = ' '.join((str(o_height), str(dft_under_height)))
            
            tmp2 = ' '.join(tmp[:4])
            #print tmp2
            outF.write(tmp2)
            outF.write("\n")  
        elif 'Output Directory' in line:
            tmp = re.split(r'\s+',line)
            tmp[3] = str(td) + '/'
            tmp2 = ' '.join(tmp[:])
            outF.write(tmp2)
            outF.write("\n")
        else:
            outF.write(line) 
    
    outF.close()

    print("configuration file updated")
    return config_file_tmp
    
     
  
def update_config_file_under_only(cover_type = [], lat_lon=[], fc = -9999, lai = -9999, u_height = -9999, 
                                  snow_item=[], veg_item=[], td =[], met_path = []):
    # Input Templates 
    config_file_tmp = os.path.join(td, 'Input.Snotel.T4_tmp')
    with open("./example_config/Input.Snotel.T4", 'r') as file:
        # read a list of lines into data
        data = file.readlines()

    # Output Config 
    outF = open( config_file_tmp ,"w")
            
    ################################
    # Snow Parameters
    ################################

    rthresh = float(snow_item['Train'])
    sthresh = float(snow_item['TSnow'])

    amax = float(snow_item['amax'])
    acclamb = float(snow_item['acc_lmbda'])
    mellamb = float(snow_item['melt_lmbda']) 
    
    sw_cap = float(veg_item['SW_cap'])

    #dft_over_lai = veg_item['LAI_Over']
    dft_under_lai = float(veg_item['LAI_Under'])
    #dft_over_height = veg_item['H_Over']
    dft_under_height = float(veg_item['H_Under']) 
    lai_multi = np.array(veg_item.iloc[: , -12:])[0]
    
    ################################
    # Update Parameters
    ################################
    for line in data:
        ###########   Met ##############

        if 'Station File 1' in line:
#             tmp = re.split(r'\s+',line)
#             tmp[4] = met_path + 'data_' +lat_lon
#             tmp2 = ' '.join(tmp[:])
            tmp2 = 'Station File 1 =  ./met/data_47.71875_-123.71875'
            outF.write(tmp2)
            outF.write("\n")  
        
         ###########   Snow  ##############
        
        elif 'Rain Threshold' in line:
            if rthresh != -9999:
                #print line
                tmp = re.split(r'\s+',line)
                #print tmp[:]

                tmp[3] = str(rthresh)
                tmp2 = ' '.join(tmp[:])
               # print tmp2
                outF.write(tmp2)
                outF.write("\n")  
            else:
                outF.write(line)

        elif 'Snow Threshold' in line:
            if sthresh != -9999:
                #print line
                tmp = re.split(r'\s+',line)
                #print tmp[:]

                tmp[3] = str(sthresh)
                tmp2 = ' '.join(tmp[:])
               # print tmp2
                outF.write(tmp2)
                outF.write("\n")  
            else:
                outF.write(line)
        elif 'Fresh Snow Albedo' in line:
            if amax != -9999: 
                tmp = re.split(r'\s+',line)
                tmp[4] = str('%.5f' % amax)
                tmp2 = ' '.join(tmp[:])
                outF.write(tmp2)
                outF.write("\n")  
                del tmp, tmp2
            else:
                outF.write(line)

        elif 'Albedo Accumulation Lambda' in line:
            if acclamb != -9999: 
                tmp = re.split(r'\s+',line)
                tmp[4] = str('%.5f' % acclamb)
                tmp2 = ' '.join(tmp[:])
                outF.write(tmp2)
                outF.write("\n")  
                del tmp, tmp2
            else:
                outF.write(line)

        elif 'Albedo Melting Lambda' in line:
            if mellamb != -9999: 
                tmp = re.split(r'\s+',line)
                tmp[4] = str('%.5f' % mellamb)
                tmp2 = ' '.join(tmp[:])
                outF.write(tmp2)
                outF.write("\n")  
                del tmp, tmp2
            else:
                outF.write(line)
        
         ###########   Vegetation ##############
            
        elif 'Overstory Present' in line:
            outF.write('Overstory Present        1 = FALSE')
            outF.write("\n")
        
        elif 'Overstory Monthly LAI ' in line:
            outF.write('Overstory Monthly LAI    1 = ')
            outF.write("\n")

        elif 'Overstory Monthly Alb' in line:
            outF.write('Overstory Monthly Alb    1 = ')
            outF.write("\n")

        elif 'Fractional Cover' in line:
            outF.write('Fractional Coverage      1 = ')
            outF.write("\n")
                        
        elif 'Height' in line and('Reference Height' not in line ):
            tmp = re.split(r'\s+',line)       
            if u_height != -9999:            
                tmp[3] = str(u_height)
            else: 
                tmp[3] = str(dft_under_height)
                
            tmp2 = ' '.join(tmp[:4])
            outF.write(tmp2)
            outF.write("\n")
            del tmp, tmp2
        elif 'Maximum Resistance' in line:
            tmp = re.split(r'\s+', line)
            tmp[4] = tmp[5]
            tmp2 = ' '.join(tmp[0:5])
            outF.write(tmp2)
            outF.write("\n")
            del tmp, tmp2

        elif 'Minimum Resistance' in line:
            tmp = re.split(r'\s+', line)
            tmp[4] = tmp[5]
            tmp2 = ' '.join(tmp[0:5])
            outF.write(tmp2)
            outF.write("\n")
            del tmp, tmp2

        elif 'Moisture Threshold' in line:
            tmp = re.split(r'\s+', line)
            tmp[4] = tmp[5]
            tmp2 = ' '.join(tmp[0:5])
            outF.write(tmp2)
            outF.write("\n")
            del tmp, tmp2

        elif 'Vapor Pressure Deficit' in line:
            tmp = re.split(r'\s+', line)
            tmp[5] = tmp[6]
            tmp2 = ' '.join(tmp[0:6])
            outF.write(tmp2)
            outF.write("\n")
            del tmp, tmp2
        elif 'Rpc' in line:
            tmp = re.split(r'\s+', line)
            tmp[3] = tmp[4]
            tmp2 = ' '.join(tmp[0:4])
            outF.write(tmp2)
            outF.write("\n")
            del tmp, tmp2
        elif 'Moisture Threshold' in line:
            tmp = re.split(r'\s+', line)
            tmp[4] = tmp[5]
            tmp2 = ' '.join(tmp[0:5])
            outF.write(tmp2)
            outF.write("\n")
            del tmp, tmp2
        elif 'Overstory Root Fraction' in line:
            outF.write('Overstory Root Fraction  1 = ')
            outF.write("\n")
        elif 'Output Directory' in line:
            tmp = re.split(r'\s+',line)
            tmp[3] = str(td) + '/'
            tmp2 = ' '.join(tmp[:])
            outF.write(tmp2)
            outF.write("\n")
        else:
            outF.write(line) 
    
    outF.close()

    print("configuration file updated")
    return config_file_tmp

  
def update_config_file_open(cover_type = [], lat_lon=[], snow_item=[], veg_item=[], td = [], met_path = []):
    
    config_file_tmp = os.path.join(td, 'Input.Snotel.T4_tmp')
    # Input Templates 
    with open("./example_config/Input.Snotel.T4", 'r') as file:
        # read a list of lines into data
        data = file.readlines()

    # Output Config 
    outF = open(config_file_tmp,"w")
            
    ################################
    # Snow Parameters
    ################################

    rthresh = float(snow_item['Train'])
    sthresh = float(snow_item['TSnow'])

    amax = float(snow_item['amax'])
    acclamb = float(snow_item['acc_lmbda'])
    mellamb = float(snow_item['melt_lmbda']) 
    
    sw_cap = float(veg_item['SW_cap'])
    
    ################################
    # Update Parameters
    ################################
    for line in data:
        ###########   Met ##############

        if 'Station File 1' in line:
#             tmp = re.split(r'\s+',line)
#             tmp[4] = met_path + 'data_' +lat_lon
#             tmp2 = ' '.join(tmp[:])
            tmp2 = 'Station File 1 =  ./met/data_47.71875_-123.71875'
            outF.write(tmp2)
            outF.write("\n")  
        
         ###########   Snow  ##############
        
        elif 'Rain Threshold' in line:
            if rthresh != -9999:
                #print line
                tmp = re.split(r'\s+',line)
                #print tmp[:]

                tmp[3] = str(rthresh)
                tmp2 = ' '.join(tmp[:])
               # print tmp2
                outF.write(tmp2)
                outF.write("\n")  
            else:
                outF.write(line)

        elif 'Snow Threshold' in line:
            if sthresh != -9999:
                #print line
                tmp = re.split(r'\s+',line)
                #print tmp[:]

                tmp[3] = str(sthresh)
                tmp2 = ' '.join(tmp[:])
               # print tmp2
                outF.write(tmp2)
                outF.write("\n")  
            else:
                outF.write(line)
        elif 'Fresh Snow Albedo' in line:
            if amax != -9999: 
                tmp = re.split(r'\s+',line)
                tmp[4] = str('%.5f' % amax)
                tmp2 = ' '.join(tmp[:])
                outF.write(tmp2)
                outF.write("\n")  
                del tmp, tmp2
            else:
                outF.write(line)

        elif 'Albedo Accumulation Lambda' in line:
            if acclamb != -9999: 
                tmp = re.split(r'\s+',line)
                tmp[4] = str('%.5f' % acclamb)
                tmp2 = ' '.join(tmp[:])
                outF.write(tmp2)
                outF.write("\n")  
                del tmp, tmp2
            else:
                outF.write(line)

        elif 'Albedo Melting Lambda' in line:
            if mellamb != -9999: 
                tmp = re.split(r'\s+',line)
                tmp[4] = str('%.5f' % mellamb)
                tmp2 = ' '.join(tmp[:])
                outF.write(tmp2)
                outF.write("\n")  
                del tmp, tmp2
            else:
                outF.write(line)
        
         ###########   Vegetation ##############
            
        elif 'Overstory Present' in line:
            outF.write('Overstory Present        1 = FALSE')
            outF.write("\n")
        elif 'Understory Present' in line:
            outF.write('Understory Present        1 = FALSE')
            outF.write("\n")
        elif 'Output Directory' in line:
            tmp = re.split(r'\s+',line)
            tmp[3] = str(td) + '/'
            tmp2 = ' '.join(tmp[:])
            outF.write(tmp2)
            outF.write("\n")
        else:
            outF.write(line) 
    
    outF.close()

    print("configuration file updated")
    return config_file_tmp
    
def find_snow_param(grid_lat, grid_lon):
    snow_param_file = './example_config/param_5cluster_ensembleMean_CONUS.csv'
    snow_param = pd.read_csv(snow_param_file,names=['lat','lon','acc_lmbda','melt_lmbda','amax','TSnow','Train','cluster'])
    snow_item = snow_param[(snow_param['lat'] == float(grid_lat) ) & (snow_param['lon'] == float(grid_lon))]
    
    return snow_item
    
def find_default_advan(cover, cluster):
    if cover == 'Open':
        cover = 'Evergreen'

    veg_param_file = './example_config/veg_param_type_cluster.csv'
    veg_param = pd.read_csv(veg_param_file, skiprows=[0,1], 
                            names=['cover_type','cluster','LAI_Over','LAI_Under','H_Over','H_Under','RA_multi','SN_multi','SW_cap','MaxSnowInt','IntEffi', '', 
                                   'LAI_R_1', 'LAI_R_2', 'LAI_R_3', 'LAI_R_4', 'LAI_R_5', 'LAI_R_6', 'LAI_R_7', 'LAI_R_8', 'LAI_R_9', 'LAI_R_10', 'LAI_R_11', 'LAI_R_12'])
    
    veg_sub = veg_param[veg_param['cover_type'] == cover]
    veg_item = veg_sub[veg_sub['cluster'] == cluster]
    
    return veg_item

def update_config_file_advanced (adv_par = [-9999, -9999, -9999, -9999], veg_item = [], td = []):
    config_file_tmp = os.path.join(td, 'Input.Snotel.T4_tmp')
    config_file = os.path.join(td, 'Input.Snotel.T4') 
    # Input Templates 
    with open(config_file_tmp, 'r') as file:
        # read a list of lines into data
        data = file.readlines()

    # Output Config 
    outF = open(config_file,"w")
    
    rain_lai = float(adv_par[0])
    snow_lai = float(adv_par[1])
    max_int = float(adv_par[2])
    snow_eff = float(adv_par[3])
    
    if rain_lai == -9999: 
        rain_lai = float(veg_item['RA_multi'])
    if snow_lai == -9999: 
        snow_lai = float(veg_item['SN_multi'])
    if max_int == -9999: 
        max_int = float(veg_item['MaxSnowInt'])
    if snow_eff == -9999: 
        snow_eff = float(veg_item['IntEffi'])
    
    # Update parameters  10 
    for line in data:
        ########## 
        if 'Rain LAI Multiplier' in line:
            if rain_lai != -9999: 
                tmp = re.split(r'\s+',line)
                tmp[4] = str('%.5f' % rain_lai)
                tmp2 = ' '.join(tmp[:])
                outF.write(tmp2)
                outF.write("\n")  
                del tmp, tmp2
            else:
                outF.write(line)

        elif 'Snow LAI Multiplier' in line:
            if snow_lai != -9999: 
                tmp = re.split(r'\s+', line)
                tmp[4] = str('%.5f' % snow_lai)
                tmp2 = ' '.join(tmp[:])
                outF.write(tmp2)
                outF.write("\n")  
                del tmp, tmp2
            else:
                outF.write(line)

        ################################
        # Vegetation
        ################################
        elif 'Max Snow Int Capacity' in line:
            if max_int != -9999: 
                tmp = re.split(r'\s+',line)
                tmp[6] = str('%.3f' % max_int)
                tmp2 = ' '.join(tmp[:])
                outF.write(tmp2)
                outF.write("\n")  
                del tmp, tmp2
            else:
                outF.write(line)

        elif 'Snow Interception Eff' in line:
            if snow_eff != -9999: 
                tmp = re.split(r'\s+', line)
                tmp[5] = str('%.3f' % snow_eff)
                tmp2 = ' '.join(tmp[:])
                outF.write(tmp2)
                outF.write("\n")  
                del tmp, tmp2
            else:
                outF.write(line)
        else:
            outF.write(line) 
    
    outF.close()

    print("configuration file updated - advanced")
    return config_file


def generate_ng_idf(base_par, adv_par, td, met_path):
    #config_file_tmp = os.path.join(td, 'Input.Snotel.T4_tmp')
    #config_file = os.path.join(td, 'Input.Snotel.T4') 
    print(base_par)
    # Basic Parameter initialization
    input_lat = float(base_par[0])
    input_lon = float(base_par[1])
    
    lat_lon = find_lat_lon(input_lat, input_lon)
    
    grid_lat = re.split('_',lat_lon)[0]
    grid_lon = re.split('_',lat_lon)[1]
    
    lai = float(base_par[2])
    o_height = float(base_par[3])
    fc = float(base_par[4])
    
    cover_type = find_cover(float(base_par[5]))
    
    # Identify snow parameters: acc_lmbda, melt_lmbda, amax, TSnow, Train, cluster
    snow_item = find_snow_param(grid_lat, grid_lon)
    
    # Cluster information
    cluster = snow_item['cluster']
    print(cluster)
    print(cover_type)
    
    
    # Initialize vegetation characteristic 
    
    veg_item = find_default_advan(cover_type, int(cluster))
    
    #print veg_item
    
    # Advanced Parameter initialization
    if cover_type in ['Evergreen', 'Deciduous', 'Mixed','Wetland']:
        new_config_tmp = update_config_file_basic (cover_type, lat_lon, fc, lai, o_height, snow_item, veg_item, td, met_path)
    elif cover_type in ['Crop','Grass','Shrub','Pasture']:
        new_config_tmp = update_config_file_under_only(cover_type, lat_lon, fc, lai, o_height, snow_item, veg_item, td, met_path)
    elif cover_type == 'Open':
        new_config_tmp = update_config_file_open(cover_type, lat_lon, snow_item, veg_item, td, met_path)
    
    new_config = update_config_file_advanced (adv_par, veg_item, td)
    
    return new_config    