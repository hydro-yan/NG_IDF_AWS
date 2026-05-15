
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64


def npnan(x,y):
    #this function creates the np.nan 2d-array (np.nan should be float)
    array_2d = np.zeros((x,y), float) 
    array_2d[:] = np.nan
    return array_2d

def read_idf(file, data):
    # file is the IDF path
    # data is a np array (3x51)
    lines = [line.rstrip('\n') for line in open(file)]  
    count = 0
    for line in lines:
        item = line.split() 
        for k in range(len(item)):
            data[count, k] = float(item[k])  
        count += 1  
    return data

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue())


def generate_fig_single(P_IDF, NG_IDF, fig_file, duration_label):
    """
    Generate a single IDF figure for a given duration
    
    Parameters:
    -----------
    P_IDF : np.array (3 x 51)
        Precipitation IDF data
    NG_IDF : np.array (3 x 51)
        Net Groundwater IDF data
    fig_file : str
        Output file path
    duration_label : str
        Label for the duration (e.g., "24-hour", "1-hour")
    
    Returns:
    --------
    str : Base64 encoded image string
    """
    
    # The 51 probabilities used to compute the IDF curves
    probs = np.concatenate([np.arange(0.50, 1.00, 0.01), np.array([0.998])])
    
    # ARI tick positions
    ari_ticks = np.array([2, 5, 10, 25, 50, 100, 500])
    ari_tick_labels = ['2', '5', '10', '25', '50', '100', '500']
    
    # P-IDF first row is estimate only (no CI)
    P_curve = P_IDF[0, :]
    NG_curve = NG_IDF[0, :]
    
    # y-limits across both curves
    ymin = min(np.nanmin(P_curve), np.nanmin(NG_curve))
    ymax = max(np.nanmax(P_curve), np.nanmax(NG_curve))
    margin = 0.05 * (ymax - ymin)
    ymin -= margin
    ymax += margin
    
    plt.figure(figsize=(6, 5))
    
    # Plot using probabilities but log-scale defined by ARI ticks
    plt.semilogx(1.0 / (1.0 - probs), P_curve, color='blue', lw=2, label='PREC-IDF')
    plt.semilogx(1.0 / (1.0 - probs), NG_curve, color='pink', lw=2, label='NG-IDF')
    
    # Set x-ticks to ARI values
    plt.xticks(ari_ticks, ari_tick_labels)
    plt.xlabel('Average Recurrence Interval (years)')
    plt.ylabel(f'Magnitude (mm) - {duration_label}')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.ylim(ymin, ymax)
    plt.legend(loc='upper left')
    plt.title(f'IDF Curves - {duration_label}')
    
    plt.savefig(fig_file, dpi=150)
    plt.close()
    
    # Return base64 image string
    img = base64.b64encode(open(fig_file, "rb").read()).decode('utf-8')
    return "data:image/png;base64," + img


def generate_fig(P_IDF_24h, P_IDF_48h, P_IDF_72h,
                 NG_IDF_24h, NG_IDF_48h, NG_IDF_72h,
                 fig24_file, fig48_file, fig72_file):
    """
    Generate three IDF figures for 24h, 48h, and 72h durations
    (Backward compatibility function for Daymet)
    """
    
    fig24_code = generate_fig_single(P_IDF_24h, NG_IDF_24h, fig24_file, "24-hour")
    fig48_code = generate_fig_single(P_IDF_48h, NG_IDF_48h, fig48_file, "48-hour")
    fig72_code = generate_fig_single(P_IDF_72h, NG_IDF_72h, fig72_file, "72-hour")
    
    return fig24_code, fig48_code, fig72_code


def generate_figs_multiple(idf_data_dict, fig_files_dict, durations):
    """
    Generate multiple IDF figures for different durations
    
    Parameters:
    -----------
    idf_data_dict : dict
        Dictionary with keys like 'P_1h', 'NG_1h', 'P_3h', 'NG_3h', etc.
    fig_files_dict : dict
        Dictionary with keys like 'fig_1h', 'fig_3h', etc.
    durations : list
        List of duration strings like ['1h', '3h', '6h', '12h', '24h', '48h', '72h']
    
    Returns:
    --------
    dict : Dictionary with keys like 'fig_1h', 'fig_3h', etc. containing base64 image strings
    """
    
    fig_codes = {}
    
    for dur in durations:
        P_key = f'P_{dur}'
        NG_key = f'NG_{dur}'
        fig_key = f'fig_{dur}'
        
        if P_key in idf_data_dict and NG_key in idf_data_dict:
            fig_code = generate_fig_single(
                idf_data_dict[P_key],
                idf_data_dict[NG_key],
                fig_files_dict[fig_key],
                f"{dur.replace('h', '-hour')}"
            )
            fig_codes[fig_key] = fig_code
    
    return fig_codes
