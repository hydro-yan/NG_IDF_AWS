
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
    plt.xlabel('Average Recurrence Interval (years)', fontsize=10, fontweight='bold')
    plt.ylabel(f'Magnitude (mm) - {duration_label}', fontsize=10, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.ylim(ymin, ymax)
    plt.legend(loc='upper left')
    #plt.title(f'IDF Curves - {duration_label}')
    
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


def generate_am_timeseries_plots(am_results, durations, fig_file_combined, fig_file_w):
    """
    Generate combined time series plot for Annual Maximum P and W
    
    Parameters:
    -----------
    am_results : dict
        Dictionary with AM data for each duration
    durations : list
        List of duration strings like ['24h', '48h', '72h']
    fig_file_combined : str
        Output file path for combined P and W time series
    fig_file_w : str
        Not used (kept for backward compatibility)
    
    Returns:
    --------
    tuple : (base64_combined, base64_combined) - returns same plot twice for compatibility
    """
    
    # Create combined figure for P and W
    fig, axes = plt.subplots(1, len(durations), figsize=(15, 4.5))
    if len(durations) == 1:
        axes = [axes]
    
    for idx, dur in enumerate(durations):
        am_p = am_results[dur]['P']
        am_w = am_results[dur]['W_veg']
        
        years_p = am_p[:, 0].astype(int)
        values_p = am_p[:, 3]
        
        years_w = am_w[:, 0].astype(int)
        values_w = am_w[:, 3]
        
        # Plot P in blue (same as PREC-IDF)
        axes[idx].plot(years_p, values_p, 'o-', color='blue', linewidth=2, markersize=4, label='AM P')
        
        # Plot W in pink (same as NG-IDF)
        axes[idx].plot(years_w, values_w, 's-', color='pink', linewidth=2, markersize=4, label='AM W')
        axes[idx].set_xlabel('Water Year', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel(f'Magnitude (mm) - {dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
        #axes[idx].set_title(f'{dur.replace("h", "-hour")} Duration', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(fig_file_combined, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return base64 encoded string (return twice for backward compatibility)
    img_combined = base64.b64encode(open(fig_file_combined, "rb").read()).decode('utf-8')
    encoded = "data:image/png;base64," + img_combined
    
    return encoded, encoded


def generate_swe_timeseries_plot(am_swe, fig_file):
    """
    Generate time series plot for Annual Maximum SWE
    
    Parameters:
    -----------
    am_swe : np.array
        Array with shape (N, 4) containing [year, month, day, swe_value]
    fig_file : str
        Output file path
    
    Returns:
    --------
    str : Base64 encoded image string
    """
    
    years = am_swe[:, 0].astype(int)
    values = am_swe[:, 3]
    
    plt.figure(figsize=(10, 4))
    plt.plot(years, values, 'o-', color='purple', linewidth=2, markersize=5)
    plt.xlabel('Water Year', fontsize=10, fontweight='bold')
    plt.ylabel('AM Snow Water Equivalent (mm)', fontsize=10, fontweight='bold')
    #plt.title('Annual Maximum SWE Time Series', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return base64 encoded string
    img = base64.b64encode(open(fig_file, "rb").read()).decode('utf-8')
    return "data:image/png;base64," + img


def generate_multi_scenario_idf_plots(hist_data, med_data, high_data, durations, fig_file_prec, fig_file_ng):
    """
    Generate multi-scenario IDF comparison plots in 2x4 grid
    
    Parameters:
    -----------
    hist_data : dict
        Historical scenario IDF data
    med_data : dict
        Medium emissions scenario IDF data
    high_data : dict
        High emissions scenario IDF data
    durations : list
        List of duration strings like ['1h', '3h', '6h', '12h', '24h', '48h', '72h']
    fig_file_prec : str
        Output file path for PREC-IDF comparison
    fig_file_ng : str
        Output file path for NG-IDF comparison
    
    Returns:
    --------
    tuple : (base64_prec, base64_ng) encoded image strings
    """
    
    # Probabilities for IDF curves
    probs = np.concatenate([np.arange(0.50, 1.00, 0.01), np.array([0.998])])
    ari_values = 1.0 / (1.0 - probs)
    
    # ARI tick positions
    ari_ticks = np.array([2, 5, 10, 25, 50, 100, 500])
    ari_tick_labels = ['2', '5', '10', '25', '50', '100', '500']
    
    # Create PREC-IDF comparison plot in 2x4 grid
    n_dur = len(durations)
    nrows = 2
    ncols = 4
    fig_prec, axes_prec = plt.subplots(nrows, ncols, figsize=(16, 8))
    axes_prec = axes_prec.flatten()
    
    for idx, dur in enumerate(durations):
        hist_curve = hist_data[f'P_{dur}'][0, :]
        med_curve = med_data[f'P_{dur}'][0, :]
        high_curve = high_data[f'P_{dur}'][0, :]
        
        axes_prec[idx].semilogx(ari_values, hist_curve, 'o-', color='blue', lw=2, markersize=3, label='Historical')
        axes_prec[idx].semilogx(ari_values, med_curve, 's-', color='orange', lw=2, markersize=3, label='Medium')
        axes_prec[idx].semilogx(ari_values, high_curve, '^-', color='red', lw=2, markersize=3, label='High')
        
        axes_prec[idx].set_xticks(ari_ticks)
        axes_prec[idx].set_xticklabels(ari_tick_labels)
        axes_prec[idx].set_xlabel('ARI (years)', fontsize=9)
        axes_prec[idx].set_ylabel('Magnitude (mm)', fontsize=9)
        axes_prec[idx].set_title(f'{dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
        axes_prec[idx].grid(True, alpha=0.3)
        axes_prec[idx].legend(loc='best', fontsize=8)
    
    # Hide the last subplot if we have 7 durations
    if n_dur == 7:
        axes_prec[7].axis('off')
    
    #plt.suptitle('PREC-IDF Curves: Climate Scenario Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save to file or memory
    if fig_file_prec:
        plt.savefig(fig_file_prec, dpi=150, bbox_inches='tight')
        img_prec = base64.b64encode(open(fig_file_prec, "rb").read()).decode('utf-8')
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_prec = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Create NG-IDF comparison plot in 2x4 grid
    fig_ng, axes_ng = plt.subplots(nrows, ncols, figsize=(16, 8))
    axes_ng = axes_ng.flatten()
    
    for idx, dur in enumerate(durations):
        hist_curve = hist_data[f'NG_{dur}'][0, :]
        med_curve = med_data[f'NG_{dur}'][0, :]
        high_curve = high_data[f'NG_{dur}'][0, :]
        
        axes_ng[idx].semilogx(ari_values, hist_curve, 'o-', color='blue', lw=2, markersize=3, label='Historical')
        axes_ng[idx].semilogx(ari_values, med_curve, 's-', color='orange', lw=2, markersize=3, label='Medium')
        axes_ng[idx].semilogx(ari_values, high_curve, '^-', color='red', lw=2, markersize=3, label='High')
        
        axes_ng[idx].set_xticks(ari_ticks)
        axes_ng[idx].set_xticklabels(ari_tick_labels)
        axes_ng[idx].set_xlabel('ARI (years)', fontsize=9)
        axes_ng[idx].set_ylabel('Magnitude (mm)', fontsize=9)
        axes_ng[idx].set_title(f'{dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
        axes_ng[idx].grid(True, alpha=0.3)
        axes_ng[idx].legend(loc='best', fontsize=8)
    
    # Hide the last subplot if we have 7 durations
    if n_dur == 7:
        axes_ng[7].axis('off')
    
    #plt.suptitle('NG-IDF Curves: Climate Scenario Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save to file or memory
    if fig_file_ng:
        plt.savefig(fig_file_ng, dpi=150, bbox_inches='tight')
        img_ng = base64.b64encode(open(fig_file_ng, "rb").read()).decode('utf-8')
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_ng = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return "data:image/png;base64," + img_prec, "data:image/png;base64," + img_ng


def generate_cesm_ensemble_am_plots(hist_le2_am, hist_le4_am, hist_le7_am, hist_le9_am,
                                     futu_le2_am, futu_le4_am, futu_le7_am, futu_le9_am,
                                     durations, fig_file_p, fig_file_w):
    """
    Generate CESM multi-ensemble AM time series plots in 2x4 grid
    Shows all 4 historical and 4 future ensemble members
    
    Parameters:
    -----------
    hist_le2_am, hist_le4_am, hist_le7_am, hist_le9_am : dict
        Historical ensemble member AM results
    futu_le2_am, futu_le4_am, futu_le7_am, futu_le9_am : dict
        Future ensemble member AM results
    durations : list
        List of duration strings
    fig_file_p : str
        Output file for P comparison
    fig_file_w : str
        Output file for W comparison
    
    Returns:
    --------
    tuple : (base64_p, base64_w) encoded image strings
    """
    
    try:
        # Color schemes for ensemble members
        hist_colors = ['blue', 'cyan', 'navy', 'dodgerblue']
        futu_colors = ['red', 'orange', 'darkred', 'coral']
        hist_labels = ['Hist LE2', 'Hist LE4', 'Hist LE7', 'Hist LE9']
        futu_labels = ['Futu LE2', 'Futu LE4', 'Futu LE7', 'Futu LE9']
        
        # Create combined P plot in 2x4 grid
        n_dur = len(durations)
        nrows = 2
        ncols = 4
        fig_p, axes_p = plt.subplots(nrows, ncols, figsize=(16, 8))
        axes_p = axes_p.flatten()
        
        for idx, dur in enumerate(durations):
            # Plot all 4 historical ensemble members
            for i, (data, color, label) in enumerate(zip(
                [hist_le2_am, hist_le4_am, hist_le7_am, hist_le9_am],
                hist_colors,
                hist_labels
            )):
                p = data[dur]['P']
                axes_p[idx].plot(p[:, 0].astype(int), p[:, 3], '-', color=color, lw=1.5,
                               markersize=3, alpha=0.8, label=label)
            
            # Plot all 4 future ensemble members
            for i, (data, color, label) in enumerate(zip(
                [futu_le2_am, futu_le4_am, futu_le7_am, futu_le9_am],
                futu_colors,
                futu_labels
            )):
                p = data[dur]['P']
                axes_p[idx].plot(p[:, 0].astype(int), p[:, 3], '--', color=color, lw=1.5,
                               markersize=3, alpha=0.8, label=label)
            
            axes_p[idx].set_xlabel('Water Year', fontsize=9)
            axes_p[idx].set_ylabel('AM Precipitation (mm)', fontsize=9)
            axes_p[idx].set_title(f'{dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
            axes_p[idx].grid(True, alpha=0.3)
            axes_p[idx].tick_params(axis='x', rotation=45)
            axes_p[idx].legend(loc='best', fontsize=6, ncol=2)
        
        # Hide the last subplot if we have 7 durations
        if n_dur == 7:
            axes_p[7].axis('off')
        
        plt.tight_layout()
        
        # Save to file or memory
        if fig_file_p:
            plt.savefig(fig_file_p, dpi=150, bbox_inches='tight')
            img_p = base64.b64encode(open(fig_file_p, "rb").read()).decode('utf-8')
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_p = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Create combined W plot in 2x4 grid
        fig_w, axes_w = plt.subplots(nrows, ncols, figsize=(16, 8))
        axes_w = axes_w.flatten()
        
        for idx, dur in enumerate(durations):
            # Plot all 4 historical ensemble members
            for i, (data, color, label) in enumerate(zip(
                [hist_le2_am, hist_le4_am, hist_le7_am, hist_le9_am],
                hist_colors,
                hist_labels
            )):
                w = data[dur]['W_veg']
                axes_w[idx].plot(w[:, 0].astype(int), w[:, 3], '-', color=color, lw=1.5,
                               markersize=3, alpha=0.8, label=label)
            
            # Plot all 4 future ensemble members
            for i, (data, color, label) in enumerate(zip(
                [futu_le2_am, futu_le4_am, futu_le7_am, futu_le9_am],
                futu_colors,
                futu_labels
            )):
                w = data[dur]['W_veg']
                axes_w[idx].plot(w[:, 0].astype(int), w[:, 3], '--', color=color, lw=1.5,
                               markersize=3, alpha=0.8, label=label)
            
            axes_w[idx].set_xlabel('Water Year', fontsize=9)
            axes_w[idx].set_ylabel('AM Water for Runoff (mm)', fontsize=9)
            axes_w[idx].set_title(f'{dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
            axes_w[idx].grid(True, alpha=0.3)
            axes_w[idx].tick_params(axis='x', rotation=45)
            axes_w[idx].legend(loc='best', fontsize=6, ncol=2)
        
        # Hide the last subplot if we have 7 durations
        if n_dur == 7:
            axes_w[7].axis('off')
        
        plt.tight_layout()
        
        # Save to file or memory
        if fig_file_w:
            plt.savefig(fig_file_w, dpi=150, bbox_inches='tight')
            img_w = base64.b64encode(open(fig_file_w, "rb").read()).decode('utf-8')
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_w = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return "data:image/png;base64," + img_p, "data:image/png;base64," + img_w
    
    except Exception as e:
        print(f"ERROR in generate_cesm_ensemble_am_plots: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Return empty base64 strings on error
        return "", ""


def generate_cesm_ensemble_swe_plot(hist_le2_swe, hist_le4_swe, hist_le7_swe, hist_le9_swe,
                                     futu_le2_swe, futu_le4_swe, futu_le7_swe, futu_le9_swe,
                                     fig_file):
    """
    Generate CESM multi-ensemble SWE time series plot
    Shows all 4 historical and 4 future ensemble members
    
    Parameters:
    -----------
    hist_le2_swe, hist_le4_swe, hist_le7_swe, hist_le9_swe : np.array
        Historical ensemble member SWE data
    futu_le2_swe, futu_le4_swe, futu_le7_swe, futu_le9_swe : np.array
        Future ensemble member SWE data
    fig_file : str
        Output file path
    
    Returns:
    --------
    str : Base64 encoded image string
    """
    
    try:
        # Color schemes for ensemble members
        hist_colors = ['blue', 'cyan', 'navy', 'dodgerblue']
        futu_colors = ['red', 'orange', 'darkred', 'coral']
        hist_labels = ['Hist LE2', 'Hist LE4', 'Hist LE7', 'Hist LE9']
        futu_labels = ['Futu LE2', 'Futu LE4', 'Futu LE7', 'Futu LE9']
        
        plt.figure(figsize=(10, 5))
        
        # Plot all 4 historical ensemble members
        for i, (swe, color, label) in enumerate(zip(
            [hist_le2_swe, hist_le4_swe, hist_le7_swe, hist_le9_swe],
            hist_colors,
            hist_labels
        )):
            plt.plot(swe[:, 0].astype(int), swe[:, 3], '-', color=color, lw=2,
                    markersize=4, alpha=0.8, label=label)
        
        # Plot all 4 future ensemble members
        for i, (swe, color, label) in enumerate(zip(
            [futu_le2_swe, futu_le4_swe, futu_le7_swe, futu_le9_swe],
            futu_colors,
            futu_labels
        )):
            plt.plot(swe[:, 0].astype(int), swe[:, 3], '--', color=color, lw=2,
                    markersize=4, alpha=0.8, label=label)
        
        plt.xlabel('Water Year', fontsize=11, fontweight='bold')
        plt.ylabel('AM Snow Water Equivalent (mm)', fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend(loc='best', fontsize=9, ncol=2)
        plt.tight_layout()
        
        # Save to file or memory
        if fig_file:
            plt.savefig(fig_file, dpi=150, bbox_inches='tight')
            img = base64.b64encode(open(fig_file, "rb").read()).decode('utf-8')
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return "data:image/png;base64," + img
    
    except Exception as e:
        print(f"ERROR in generate_cesm_ensemble_swe_plot: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Return empty base64 string on error
        return ""


def generate_multi_scenario_am_plots(hist_am, med_am, high_am, durations, fig_file_p, fig_file_w):
    """
    Generate multi-scenario AM time series plots in 2x4 grid
    
    Parameters:
    -----------
    hist_am : dict
        Historical AM results
    med_am : dict
        Medium emissions AM results
    high_am : dict
        High emissions AM results
    durations : list
        List of duration strings
    fig_file_p : str
        Output file for P comparison
    fig_file_w : str
        Output file for W comparison
    
    Returns:
    --------
    tuple : (base64_p, base64_w) encoded image strings
    """
    
    # Create combined P plot in 2x4 grid
    n_dur = len(durations)
    nrows = 2
    ncols = 4
    fig_p, axes_p = plt.subplots(nrows, ncols, figsize=(16, 8))
    axes_p = axes_p.flatten()
    
    for idx, dur in enumerate(durations):
        hist_p = hist_am[dur]['P']
        med_p = med_am[dur]['P']
        high_p = high_am[dur]['P']
        
        axes_p[idx].plot(hist_p[:, 0].astype(int), hist_p[:, 3], 'o-', color='blue', lw=2, markersize=3, label='Historical')
        axes_p[idx].plot(med_p[:, 0].astype(int), med_p[:, 3], 's-', color='orange', lw=2, markersize=3, label='Medium')
        axes_p[idx].plot(high_p[:, 0].astype(int), high_p[:, 3], '^-', color='red', lw=2, markersize=3, label='High')
        
        axes_p[idx].set_xlabel('Water Year', fontsize=9)
        axes_p[idx].set_ylabel('AM Precipitation (mm)', fontsize=9)
        axes_p[idx].set_title(f'{dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
        axes_p[idx].grid(True, alpha=0.3)
        axes_p[idx].tick_params(axis='x', rotation=45)
        axes_p[idx].legend(loc='best', fontsize=8)
    
    # Hide the last subplot if we have 7 durations
    if n_dur == 7:
        axes_p[7].axis('off')
    
    #plt.suptitle('Annual Maximum Precipitation: Climate Scenario Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save to file or memory
    if fig_file_p:
        plt.savefig(fig_file_p, dpi=150, bbox_inches='tight')
        img_p = base64.b64encode(open(fig_file_p, "rb").read()).decode('utf-8')
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_p = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Create combined W plot in 2x4 grid
    fig_w, axes_w = plt.subplots(nrows, ncols, figsize=(16, 8))
    axes_w = axes_w.flatten()
    
    for idx, dur in enumerate(durations):
        hist_w = hist_am[dur]['W_veg']
        med_w = med_am[dur]['W_veg']
        high_w = high_am[dur]['W_veg']
        
        axes_w[idx].plot(hist_w[:, 0].astype(int), hist_w[:, 3], 'o-', color='blue', lw=2, markersize=3, label='Historical')
        axes_w[idx].plot(med_w[:, 0].astype(int), med_w[:, 3], 's-', color='orange', lw=2, markersize=3, label='Medium')
        axes_w[idx].plot(high_w[:, 0].astype(int), high_w[:, 3], '^-', color='red', lw=2, markersize=3, label='High')
        
        axes_w[idx].set_xlabel('Water Year', fontsize=9)
        axes_w[idx].set_ylabel('AM Water for Runoff (mm)', fontsize=9)
        axes_w[idx].set_title(f'{dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
        axes_w[idx].grid(True, alpha=0.3)
        axes_w[idx].tick_params(axis='x', rotation=45)
        axes_w[idx].legend(loc='best', fontsize=8)
    
    # Hide the last subplot if we have 7 durations
    if n_dur == 7:
        axes_w[7].axis('off')
    
    #plt.suptitle('Annual Maximum Water for Runoff: Climate Scenario Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save to file or memory
    if fig_file_w:
        plt.savefig(fig_file_w, dpi=150, bbox_inches='tight')
        img_w = base64.b64encode(open(fig_file_w, "rb").read()).decode('utf-8')
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_w = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return "data:image/png;base64," + img_p, "data:image/png;base64," + img_w


def generate_multi_scenario_swe_plot(hist_swe, med_swe, high_swe, fig_file):
    """
    Generate multi-scenario SWE time series plot (smaller size)
    
    Parameters:
    -----------
    hist_swe : np.array
        Historical SWE data
    med_swe : np.array
        Medium emissions SWE data
    high_swe : np.array
        High emissions SWE data
    fig_file : str
        Output file path
    
    Returns:
    --------
    str : Base64 encoded image string
    """
    
    plt.figure(figsize=(8, 4))
    
    plt.plot(hist_swe[:, 0].astype(int), hist_swe[:, 3], 'o-', color='blue', lw=2, markersize=4, label='Historical')
    plt.plot(med_swe[:, 0].astype(int), med_swe[:, 3], 's-', color='orange', lw=2, markersize=4, label='Medium')
    plt.plot(high_swe[:, 0].astype(int), high_swe[:, 3], '^-', color='red', lw=2, markersize=4, label='High')
    
    plt.xlabel('Water Year', fontsize=11)
    plt.ylabel('AM Snow Water Equivalent (mm)', fontsize=11)
    #plt.title('Annual Maximum SWE: Climate Scenario Comparison', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    # Save to file or memory
    if fig_file:
        plt.savefig(fig_file, dpi=150, bbox_inches='tight')
        img = base64.b64encode(open(fig_file, "rb").read()).decode('utf-8')
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return "data:image/png;base64," + img


def generate_cesm_ensemble_idf_plots(hist_le2, hist_le4, hist_le7, hist_le9,
                                      futu_le2, futu_le4, futu_le7, futu_le9,
                                      durations, fig_file_prec, fig_file_ng):
    """
    Generate CESM multi-ensemble IDF comparison plots in 2x4 grid
    Shows all 4 historical ensemble members and all 4 future ensemble members
    
    Parameters:
    -----------
    hist_le2, hist_le4, hist_le7, hist_le9 : dict
        Historical ensemble member IDF data
    futu_le2, futu_le4, futu_le7, futu_le9 : dict
        Future ensemble member IDF data
    durations : list
        List of duration strings like ['1h', '3h', '6h', '12h', '24h', '48h', '72h']
    fig_file_prec : str
        Output file path for PREC-IDF comparison
    fig_file_ng : str
        Output file path for NG-IDF comparison
    
    Returns:
    --------
    tuple : (base64_prec, base64_ng) encoded image strings
    """
    
    # Probabilities for IDF curves
    probs = np.concatenate([np.arange(0.50, 1.00, 0.01), np.array([0.998])])
    ari_values = 1.0 / (1.0 - probs)
    
    # ARI tick positions
    ari_ticks = np.array([2, 5, 10, 25, 50, 100, 500])
    ari_tick_labels = ['2', '5', '10', '25', '50', '100', '500']
    
    # Color schemes for ensemble members
    hist_colors = ['blue', 'cyan', 'navy', 'dodgerblue']
    futu_colors = ['red', 'orange', 'darkred', 'coral']
    hist_labels = ['Hist LE2', 'Hist LE4', 'Hist LE7', 'Hist LE9']
    futu_labels = ['Futu LE2', 'Futu LE4', 'Futu LE7', 'Futu LE9']
    
    # Create PREC-IDF comparison plot in 2x4 grid
    n_dur = len(durations)
    nrows = 2
    ncols = 4
    fig_prec, axes_prec = plt.subplots(nrows, ncols, figsize=(16, 8))
    axes_prec = axes_prec.flatten()
    
    for idx, dur in enumerate(durations):
        # Plot all 4 historical ensemble members
        for i, (data, color, label) in enumerate(zip(
            [hist_le2, hist_le4, hist_le7, hist_le9],
            hist_colors,
            hist_labels
        )):
            curve = data[f'P_{dur}'][0, :]
            axes_prec[idx].semilogx(ari_values, curve, '-', color=color, lw=1.5, 
                                   alpha=0.8, label=label)
        
        # Plot all 4 future ensemble members
        for i, (data, color, label) in enumerate(zip(
            [futu_le2, futu_le4, futu_le7, futu_le9],
            futu_colors,
            futu_labels
        )):
            curve = data[f'P_{dur}'][0, :]
            axes_prec[idx].semilogx(ari_values, curve, '--', color=color, lw=1.5,
                                   alpha=0.8, label=label)
        
        axes_prec[idx].set_xticks(ari_ticks)
        axes_prec[idx].set_xticklabels(ari_tick_labels)
        axes_prec[idx].set_xlabel('ARI (years)', fontsize=9)
        axes_prec[idx].set_ylabel('Magnitude (mm)', fontsize=9)
        axes_prec[idx].set_title(f'PREC-IDF {dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
        axes_prec[idx].grid(True, alpha=0.3)
        axes_prec[idx].legend(loc='best', fontsize=6, ncol=2)
    
    # Hide the last subplot if we have 7 durations
    if n_dur == 7:
        axes_prec[7].axis('off')
    
    plt.tight_layout()
    
    # Save to file or memory
    if fig_file_prec:
        plt.savefig(fig_file_prec, dpi=150, bbox_inches='tight')
        img_prec = base64.b64encode(open(fig_file_prec, "rb").read()).decode('utf-8')
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_prec = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Create NG-IDF comparison plot in 2x4 grid
    fig_ng, axes_ng = plt.subplots(nrows, ncols, figsize=(16, 8))
    axes_ng = axes_ng.flatten()
    
    for idx, dur in enumerate(durations):
        # Plot all 4 historical ensemble members
        for i, (data, color, label) in enumerate(zip(
            [hist_le2, hist_le4, hist_le7, hist_le9],
            hist_colors,
            hist_labels
        )):
            curve = data[f'NG_{dur}'][0, :]
            axes_ng[idx].semilogx(ari_values, curve, '-', color=color, lw=1.5,
                                 alpha=0.8, label=label)
        
        # Plot all 4 future ensemble members
        for i, (data, color, label) in enumerate(zip(
            [futu_le2, futu_le4, futu_le7, futu_le9],
            futu_colors,
            futu_labels
        )):
            curve = data[f'NG_{dur}'][0, :]
            axes_ng[idx].semilogx(ari_values, curve, '--', color=color, lw=1.5,
                                 alpha=0.8, label=label)
        
        axes_ng[idx].set_xticks(ari_ticks)
        axes_ng[idx].set_xticklabels(ari_tick_labels)
        axes_ng[idx].set_xlabel('ARI (years)', fontsize=9)
        axes_ng[idx].set_ylabel('Magnitude (mm)', fontsize=9)
        axes_ng[idx].set_title(f'NG-IDF {dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
        axes_ng[idx].grid(True, alpha=0.3)
        axes_ng[idx].legend(loc='best', fontsize=6, ncol=2)
    
    # Hide the last subplot if we have 7 durations
    if n_dur == 7:
        axes_ng[7].axis('off')
    
    plt.tight_layout()
    
    # Save to file or memory
    if fig_file_ng:
        plt.savefig(fig_file_ng, dpi=150, bbox_inches='tight')
        img_ng = base64.b64encode(open(fig_file_ng, "rb").read()).decode('utf-8')
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_ng = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return "data:image/png;base64," + img_prec, "data:image/png;base64," + img_ng
