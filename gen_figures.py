
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

def style_grid(ax, minor=False):
    """Draw a clearly visible dashed grid behind the data on the given axes."""
    ax.set_axisbelow(True)
    ax.grid(True, which='major', color='#6e6e6e', linestyle='--', linewidth=0.8, alpha=0.75)
    if minor:
        ax.grid(True, which='minor', color='#a0a0a0', linestyle=':', linewidth=0.5, alpha=0.6)


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


def generate_fig_single(P_IDF, NG_IDF, fig_file, duration_label,
                        atlas14_values=None, display_inches=False):
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
    atlas14_values : np.array or None
        Seven NOAA Atlas 14 magnitudes in inches at 2, 5, 10, 25, 50, 100,
        and 500-year return periods. When provided, they are overlaid as a
        reference series.
    display_inches : bool
        Convert P-IDF and NG-IDF curves from millimetres to inches for display.
    
    Returns:
    --------
    str : Base64 encoded image string
    """
    
    # The 51 probabilities used to compute the IDF curves
    probs = np.concatenate([np.arange(0.50, 1.00, 0.01), np.array([0.998])])
    
    # ARI tick positions
    ari_ticks = np.array([2, 5, 10, 25, 50, 100, 500])
    ari_tick_labels = ['2', '5', '10', '25', '50', '100', '500']
    
    # P-IDF first row is estimate only (no CI).
    unit_factor = 1.0 / 25.4 if display_inches else 1.0
    unit_label = "in" if display_inches else "mm"
    P_curve = P_IDF[0, :] * unit_factor
    NG_curve = NG_IDF[0, :] * unit_factor
    atlas14_values = (
        np.asarray(atlas14_values, dtype=float)
        if atlas14_values is not None else None
    )
    
    # y-limits across both curves
    plotted_values = [P_curve, NG_curve]
    if atlas14_values is not None:
        plotted_values.append(atlas14_values)
    ymin = min(np.nanmin(values) for values in plotted_values)
    ymax = max(np.nanmax(values) for values in plotted_values)
    margin = 0.05 * (ymax - ymin)
    ymin -= margin
    ymax += margin
    
    plt.figure(figsize=(6, 5))
    
    # Plot using probabilities but log-scale defined by ARI ticks
    plt.semilogx(1.0 / (1.0 - probs), P_curve, color='blue', lw=2,
                 label='PREC-IDF (Daymet)')
    plt.semilogx(1.0 / (1.0 - probs), NG_curve, color='pink', lw=2, label='NG-IDF')
    if atlas14_values is not None:
        plt.semilogx(ari_ticks, atlas14_values, color='#d95f02', lw=1.5,
                     marker='o', markersize=4, label='PREC-IDF (Atlas 14)')
    
    # Set x-ticks to ARI values
    plt.xticks(ari_ticks, ari_tick_labels)
    plt.xlabel('Average Recurrence Interval (years)', fontsize=10, fontweight='bold')
    plt.ylabel(f'Magnitude ({unit_label}) - {duration_label}', fontsize=10, fontweight='bold')
    style_grid(plt.gca(), minor=True)
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


def generate_figs_multiple(idf_data_dict, fig_files_dict, durations,
                           atlas14_idf=None, display_inches=False):
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
    atlas14_idf : dict or None
        Optional NOAA Atlas 14 values in inches keyed by duration.
    display_inches : bool
        Whether to display the P-IDF and NG-IDF curves in inches.
    
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
                f"{dur.replace('h', '-hour')}",
                atlas14_values=(atlas14_idf or {}).get(dur),
                display_inches=display_inches,
            )
            fig_codes[fig_key] = fig_code
    
    return fig_codes


def generate_am_timeseries_plots(am_results, durations, fig_file_combined,
                                 fig_file_w, display_inches=False):
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
    display_inches : bool
        Convert annual maximum precipitation and runoff values from millimetres
        to inches for display.
    
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
        unit_factor = 1.0 / 25.4 if display_inches else 1.0
        unit_label = "in" if display_inches else "mm"
        values_p = am_p[:, 3] * unit_factor
        
        years_w = am_w[:, 0].astype(int)
        values_w = am_w[:, 3] * unit_factor
        
        # Plot P in blue (same as PREC-IDF)
        axes[idx].plot(years_p, values_p, 'o-', color='blue', linewidth=2, markersize=4, label='AM P')
        
        # Plot W in pink (same as NG-IDF)
        axes[idx].plot(years_w, values_w, 's-', color='pink', linewidth=2, markersize=4, label='AM W')
        axes[idx].set_xlabel('Water Year', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel(f'Magnitude ({unit_label}) - {dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
        #axes[idx].set_title(f'{dur.replace("h", "-hour")} Duration', fontsize=11, fontweight='bold')
        style_grid(axes[idx])
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(fig_file_combined, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return base64 encoded string (return twice for backward compatibility)
    img_combined = base64.b64encode(open(fig_file_combined, "rb").read()).decode('utf-8')
    encoded = "data:image/png;base64," + img_combined
    
    return encoded, encoded


def generate_swe_timeseries_plot(am_swe, fig_file, display_inches=False):
    """
    Generate time series plot for Annual Maximum SWE
    
    Parameters:
    -----------
    am_swe : np.array
        Array with shape (N, 4) containing [year, month, day, swe_value]
    fig_file : str
        Output file path
    display_inches : bool
        Convert annual maximum snow water equivalent values from millimetres to
        inches for display.
    
    Returns:
    --------
    str : Base64 encoded image string
    """
    
    years = am_swe[:, 0].astype(int)
    unit_factor = 1.0 / 25.4 if display_inches else 1.0
    unit_label = "in" if display_inches else "mm"
    values = am_swe[:, 3] * unit_factor
    
    plt.figure(figsize=(10, 4))
    plt.plot(years, values, 'o-', color='purple', linewidth=2, markersize=5)
    plt.xlabel('Water Year', fontsize=10, fontweight='bold')
    plt.ylabel(f'AM Snow Water Equivalent ({unit_label})', fontsize=10, fontweight='bold')
    #plt.title('Annual Maximum SWE Time Series', fontsize=12, fontweight='bold')
    style_grid(plt.gca())
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return base64 encoded string
    img = base64.b64encode(open(fig_file, "rb").read()).decode('utf-8')
    return "data:image/png;base64," + img


def generate_multi_scenario_idf_plots(hist_data, med_data, high_data, durations, fig_file_prec, fig_file_ng, display_inches=False):
    """
    Generate multi-scenario IDF comparison plots in 2x4 grid
    
    Parameters:
    -----------
    hist_data : dict
        Historical scenario IDF data
    med_data : dict
        Moderate (+1.72 C) scenario IDF data
    high_data : dict
        Severe (+2.36 C) scenario IDF data
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
    
    unit_factor = 1.0 / 25.4 if display_inches else 1.0
    unit_label = 'in' if display_inches else 'mm'

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
        hist_curve = hist_data[f'P_{dur}'][0, :] * unit_factor
        med_curve = med_data[f'P_{dur}'][0, :] * unit_factor
        high_curve = high_data[f'P_{dur}'][0, :] * unit_factor
        
        axes_prec[idx].semilogx(ari_values, hist_curve, 'o-', color='blue', lw=2, markersize=3, label='Baseline Sim.')
        axes_prec[idx].semilogx(ari_values, med_curve, 's-', color='orange', lw=2, markersize=3, label='Moderate (+1.72\u00b0C)')
        axes_prec[idx].semilogx(ari_values, high_curve, '^-', color='red', lw=2, markersize=3, label='Severe (+2.36\u00b0C)')
        
        axes_prec[idx].set_xticks(ari_ticks)
        axes_prec[idx].set_xticklabels(ari_tick_labels)
        axes_prec[idx].set_xlabel('ARI (years)', fontsize=9)
        axes_prec[idx].set_ylabel(f'Magnitude ({unit_label})', fontsize=9)
        axes_prec[idx].set_title(f'{dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
        style_grid(axes_prec[idx])
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
        hist_curve = hist_data[f'NG_{dur}'][0, :] * unit_factor
        med_curve = med_data[f'NG_{dur}'][0, :] * unit_factor
        high_curve = high_data[f'NG_{dur}'][0, :] * unit_factor
        
        axes_ng[idx].semilogx(ari_values, hist_curve, 'o-', color='blue', lw=2, markersize=3, label='Baseline Sim.')
        axes_ng[idx].semilogx(ari_values, med_curve, 's-', color='orange', lw=2, markersize=3, label='Moderate (+1.72\u00b0C)')
        axes_ng[idx].semilogx(ari_values, high_curve, '^-', color='red', lw=2, markersize=3, label='Severe (+2.36\u00b0C)')
        
        axes_ng[idx].set_xticks(ari_ticks)
        axes_ng[idx].set_xticklabels(ari_tick_labels)
        axes_ng[idx].set_xlabel('ARI (years)', fontsize=9)
        axes_ng[idx].set_ylabel(f'Magnitude ({unit_label})', fontsize=9)
        axes_ng[idx].set_title(f'{dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
        style_grid(axes_ng[idx])
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


# Near-term (2034-2049) domain-mean annual air-temperature change (deg C) of each
# CESM-LE member; same values as the near-term spatial maps.
CESM_NEAR_TERM_DMAAT = {"LE2": 1.51, "LE4": 1.81, "LE7": 1.69, "LE9": 2.45}

# Mid-century (2034-2065) member warming; same values as the mid-century spatial maps.
CESM_MID_CENTURY_DMAAT = {"LE2": 1.89, "LE4": 2.11, "LE7": 1.61, "LE9": 2.61}


def _cesm_member_labels(member_dtair=None):
    """Legend labels for the four CESM-LE members (historical, future)."""
    members = ["LE2", "LE4", "LE7", "LE9"]
    if member_dtair is None:
        return (["Hist " + m for m in members], ["Futu " + m for m in members])
    return (
        ["Baseline " + m for m in members],
        ["Future %s (%+.2f\u00b0C)" % (m, member_dtair[m]) for m in members],
    )


def _cesm_grid(ax, member_dtair=None):
    """Visible dashed grid for the near-term style, original faint grid otherwise."""
    if member_dtair is None:
        ax.grid(True, alpha=0.3)
    else:
        style_grid(ax)


def generate_cesm_ensemble_am_plots(hist_le2_am, hist_le4_am, hist_le7_am, hist_le9_am,
                                     futu_le2_am, futu_le4_am, futu_le7_am, futu_le9_am,
                                     durations, fig_file_p, fig_file_w, member_dtair=None, display_inches=False):
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
        unit_factor = 1.0 / 25.4 if display_inches else 1.0
        unit_label = 'in' if display_inches else 'mm'

        # Color schemes for ensemble members
        hist_colors = ['blue', 'cyan', 'navy', 'dodgerblue']
        futu_colors = ['red', 'orange', 'darkred', 'coral']
        hist_labels, futu_labels = _cesm_member_labels(member_dtair)
        
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
                axes_p[idx].plot(p[:, 0].astype(int), p[:, 3] * unit_factor, '-', color=color, lw=1.5,
                               markersize=3, alpha=0.8, label=label)
            
            # Plot all 4 future ensemble members
            for i, (data, color, label) in enumerate(zip(
                [futu_le2_am, futu_le4_am, futu_le7_am, futu_le9_am],
                futu_colors,
                futu_labels
            )):
                p = data[dur]['P']
                axes_p[idx].plot(p[:, 0].astype(int), p[:, 3] * unit_factor, '--', color=color, lw=1.5,
                               markersize=3, alpha=0.8, label=label)
            
            axes_p[idx].set_xlabel('Water Year', fontsize=9)
            axes_p[idx].set_ylabel(f'AM Precipitation ({unit_label})', fontsize=9)
            axes_p[idx].set_title(f'{dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
            _cesm_grid(axes_p[idx], member_dtair)
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
                axes_w[idx].plot(w[:, 0].astype(int), w[:, 3] * unit_factor, '-', color=color, lw=1.5,
                               markersize=3, alpha=0.8, label=label)
            
            # Plot all 4 future ensemble members
            for i, (data, color, label) in enumerate(zip(
                [futu_le2_am, futu_le4_am, futu_le7_am, futu_le9_am],
                futu_colors,
                futu_labels
            )):
                w = data[dur]['W_veg']
                axes_w[idx].plot(w[:, 0].astype(int), w[:, 3] * unit_factor, '--', color=color, lw=1.5,
                               markersize=3, alpha=0.8, label=label)
            
            axes_w[idx].set_xlabel('Water Year', fontsize=9)
            axes_w[idx].set_ylabel(f'AM Water for Runoff ({unit_label})', fontsize=9)
            axes_w[idx].set_title(f'{dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
            _cesm_grid(axes_w[idx], member_dtair)
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
                                     fig_file, member_dtair=None, display_inches=False):
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
        unit_factor = 1.0 / 25.4 if display_inches else 1.0
        unit_label = 'in' if display_inches else 'mm'

        # Color schemes for ensemble members
        hist_colors = ['blue', 'cyan', 'navy', 'dodgerblue']
        futu_colors = ['red', 'orange', 'darkred', 'coral']
        hist_labels, futu_labels = _cesm_member_labels(member_dtair)
        
        plt.figure(figsize=(10, 5))
        
        # Plot all 4 historical ensemble members
        for i, (swe, color, label) in enumerate(zip(
            [hist_le2_swe, hist_le4_swe, hist_le7_swe, hist_le9_swe],
            hist_colors,
            hist_labels
        )):
            plt.plot(swe[:, 0].astype(int), swe[:, 3] * unit_factor, '-', color=color, lw=2,
                    markersize=4, alpha=0.8, label=label)
        
        # Plot all 4 future ensemble members
        for i, (swe, color, label) in enumerate(zip(
            [futu_le2_swe, futu_le4_swe, futu_le7_swe, futu_le9_swe],
            futu_colors,
            futu_labels
        )):
            plt.plot(swe[:, 0].astype(int), swe[:, 3] * unit_factor, '--', color=color, lw=2,
                    markersize=4, alpha=0.8, label=label)
        
        plt.xlabel('Water Year', fontsize=11, fontweight='bold')
        plt.ylabel(f'AM Snow Water Equivalent ({unit_label})', fontsize=11, fontweight='bold')
        _cesm_grid(plt.gca(), member_dtair)
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


def _pgw_year_shift(base_arr, fut_arr):
    """Years between a baseline AM series and its PGW counterpart (normally 44).

    PGW runs replay the baseline weather, so subtracting this shift from the
    future water years lines every event up with its baseline year.
    """
    return int(np.nanmin(fut_arr[:, 0])) - int(np.nanmin(base_arr[:, 0]))


def generate_multi_scenario_am_plots(hist_am, med_am, high_am, durations, fig_file_p, fig_file_w, display_inches=False):
    """
    Generate multi-scenario AM time series plots in 2x4 grid
    
    Parameters:
    -----------
    hist_am : dict
        Historical AM results
    med_am : dict
        Moderate (+1.72 C) scenario AM results
    high_am : dict
        Severe (+2.36 C) scenario AM results
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
    
    unit_factor = 1.0 / 25.4 if display_inches else 1.0
    unit_label = 'in' if display_inches else 'mm'

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
        shift_m = _pgw_year_shift(hist_p, med_p)
        shift_h = _pgw_year_shift(hist_p, high_p)
        
        axes_p[idx].plot(hist_p[:, 0].astype(int), hist_p[:, 3] * unit_factor, 'o-', color='blue', lw=1.0, markersize=2.5, label='Baseline Sim.')
        axes_p[idx].plot(med_p[:, 0].astype(int) - shift_m, med_p[:, 3] * unit_factor, 's-', color='orange', lw=1.0, markersize=2.5, label='Moderate (+1.72\u00b0C)')
        axes_p[idx].plot(high_p[:, 0].astype(int) - shift_h, high_p[:, 3] * unit_factor, '^-', color='red', lw=1.0, markersize=2.5, label='Severe (+2.36\u00b0C)')
        
        axes_p[idx].set_xlabel(f'Baseline-equivalent Water Year (PGW runs shifted -{shift_m} yr)', fontsize=8)
        axes_p[idx].set_ylabel(f'AM Precipitation ({unit_label})', fontsize=9)
        axes_p[idx].set_title(f'{dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
        style_grid(axes_p[idx])
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
        shift_m = _pgw_year_shift(hist_w, med_w)
        shift_h = _pgw_year_shift(hist_w, high_w)
        
        axes_w[idx].plot(hist_w[:, 0].astype(int), hist_w[:, 3] * unit_factor, 'o-', color='blue', lw=1.0, markersize=2.5, label='Baseline Sim.')
        axes_w[idx].plot(med_w[:, 0].astype(int) - shift_m, med_w[:, 3] * unit_factor, 's-', color='orange', lw=1.0, markersize=2.5, label='Moderate (+1.72\u00b0C)')
        axes_w[idx].plot(high_w[:, 0].astype(int) - shift_h, high_w[:, 3] * unit_factor, '^-', color='red', lw=1.0, markersize=2.5, label='Severe (+2.36\u00b0C)')
        
        axes_w[idx].set_xlabel(f'Baseline-equivalent Water Year (PGW runs shifted -{shift_m} yr)', fontsize=8)
        axes_w[idx].set_ylabel(f'AM Water for Runoff ({unit_label})', fontsize=9)
        axes_w[idx].set_title(f'{dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
        style_grid(axes_w[idx])
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


def generate_multi_scenario_swe_plot(hist_swe, med_swe, high_swe, fig_file, display_inches=False):
    """
    Generate multi-scenario SWE time series plot (smaller size)
    
    Parameters:
    -----------
    hist_swe : np.array
        Historical SWE data
    med_swe : np.array
        Moderate (+1.72 C) scenario SWE data
    high_swe : np.array
        Severe (+2.36 C) scenario SWE data
    fig_file : str
        Output file path
    
    Returns:
    --------
    str : Base64 encoded image string
    """
    
    unit_factor = 1.0 / 25.4 if display_inches else 1.0
    unit_label = 'in' if display_inches else 'mm'

    shift_m = _pgw_year_shift(hist_swe, med_swe)
    shift_h = _pgw_year_shift(hist_swe, high_swe)

    plt.figure(figsize=(8, 4))
    
    plt.plot(hist_swe[:, 0].astype(int), hist_swe[:, 3] * unit_factor, 'o-', color='blue', lw=1.0, markersize=3, label='Baseline Sim.')
    plt.plot(med_swe[:, 0].astype(int) - shift_m, med_swe[:, 3] * unit_factor, 's-', color='orange', lw=1.0, markersize=3, label='Moderate (+1.72\u00b0C)')
    plt.plot(high_swe[:, 0].astype(int) - shift_h, high_swe[:, 3] * unit_factor, '^-', color='red', lw=1.0, markersize=3, label='Severe (+2.36\u00b0C)')
    
    plt.xlabel(f'Baseline-equivalent Water Year (PGW runs shifted -{shift_m} yr)', fontsize=10)
    plt.ylabel(f'AM Snow Water Equivalent ({unit_label})', fontsize=11)
    #plt.title('Annual Maximum SWE: Climate Scenario Comparison', fontsize=12, fontweight='bold')
    style_grid(plt.gca())
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
                                      durations, fig_file_prec, fig_file_ng, member_dtair=None, display_inches=False):
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
    
    unit_factor = 1.0 / 25.4 if display_inches else 1.0
    
    unit_label = 'in' if display_inches else 'mm'

    
    # Color schemes for ensemble members
    hist_colors = ['blue', 'cyan', 'navy', 'dodgerblue']
    futu_colors = ['red', 'orange', 'darkred', 'coral']
    hist_labels, futu_labels = _cesm_member_labels(member_dtair)
    
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
            curve = data[f'P_{dur}'][0, :] * unit_factor
            axes_prec[idx].semilogx(ari_values, curve, '-', color=color, lw=1.5, 
                                   alpha=0.8, label=label)
        
        # Plot all 4 future ensemble members
        for i, (data, color, label) in enumerate(zip(
            [futu_le2, futu_le4, futu_le7, futu_le9],
            futu_colors,
            futu_labels
        )):
            curve = data[f'P_{dur}'][0, :] * unit_factor
            axes_prec[idx].semilogx(ari_values, curve, '--', color=color, lw=1.5,
                                   alpha=0.8, label=label)
        
        axes_prec[idx].set_xticks(ari_ticks)
        axes_prec[idx].set_xticklabels(ari_tick_labels)
        axes_prec[idx].set_xlabel('ARI (years)', fontsize=9)
        axes_prec[idx].set_ylabel(f'Magnitude ({unit_label})', fontsize=9)
        axes_prec[idx].set_title(f'PREC-IDF {dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
        _cesm_grid(axes_prec[idx], member_dtair)
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
            curve = data[f'NG_{dur}'][0, :] * unit_factor
            axes_ng[idx].semilogx(ari_values, curve, '-', color=color, lw=1.5,
                                 alpha=0.8, label=label)
        
        # Plot all 4 future ensemble members
        for i, (data, color, label) in enumerate(zip(
            [futu_le2, futu_le4, futu_le7, futu_le9],
            futu_colors,
            futu_labels
        )):
            curve = data[f'NG_{dur}'][0, :] * unit_factor
            axes_ng[idx].semilogx(ari_values, curve, '--', color=color, lw=1.5,
                                 alpha=0.8, label=label)
        
        axes_ng[idx].set_xticks(ari_ticks)
        axes_ng[idx].set_xticklabels(ari_tick_labels)
        axes_ng[idx].set_xlabel('ARI (years)', fontsize=9)
        axes_ng[idx].set_ylabel(f'Magnitude ({unit_label})', fontsize=9)
        axes_ng[idx].set_title(f'NG-IDF {dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
        _cesm_grid(axes_ng[idx], member_dtair)
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


def generate_cesm_ensemble_mean_idf_plots(hist_list, futu_list, durations, member_dtair=None,
                                          display_inches=False, kind='magnitude'):
    """
    CESM ensemble-mean IDF summary in a 2x4 grid (PREC-IDF and NG-IDF).

    The ensemble mean is taken across the four members at each ARI. The shaded
    band is the min-max range across members.

    Parameters
    ----------
    hist_list, futu_list : list of dict
        Baseline / future results (structure_results output), one per member.
    durations : list
        Duration strings such as ['1h', '3h', ..., '72h'].
    member_dtair : dict or None
        Member air-temperature changes (deg C); used only for the legend.
    display_inches : bool
        Convert magnitudes from mm to inches.
    kind : str
        'magnitude' -> baseline and future ensemble mean with member range.
        'change'    -> percent change of the ensemble mean (future mean vs
                       baseline mean) with the range of member changes.

    Returns
    -------
    tuple : (base64_prec, base64_ng) image strings
    """

    try:
        unit_factor = 1.0 / 25.4 if display_inches else 1.0
        unit_label = 'in' if display_inches else 'mm'

        probs = np.concatenate([np.arange(0.50, 1.00, 0.01), np.array([0.998])])
        ari_values = 1.0 / (1.0 - probs)
        ari_ticks = np.array([2, 5, 10, 25, 50, 100, 500])
        ari_tick_labels = ['2', '5', '10', '25', '50', '100', '500']

        if member_dtair:
            mean_dtair = float(np.mean(list(member_dtair.values())))
            futu_mean_label = 'Future ensemble mean (avg %+.2f\u00b0C)' % mean_dtair
        else:
            futu_mean_label = 'Future ensemble mean'

        def build(prefix, name):
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()

            for idx, dur in enumerate(durations):
                ax = axes[idx]
                hist = np.array([np.asarray(d[f'{prefix}_{dur}'][0, :], dtype=float)
                                 for d in hist_list]) * unit_factor
                futu = np.array([np.asarray(d[f'{prefix}_{dur}'][0, :], dtype=float)
                                 for d in futu_list]) * unit_factor
                ax.set_xscale('log')

                if kind == 'change':
                    with np.errstate(divide='ignore', invalid='ignore'):
                        member_chg = (futu - hist) / hist * 100.0
                        mean_chg = (futu.mean(axis=0) - hist.mean(axis=0)) / hist.mean(axis=0) * 100.0
                    ax.fill_between(ari_values, np.nanmin(member_chg, axis=0),
                                    np.nanmax(member_chg, axis=0), color='red',
                                    alpha=0.2, label='Member range (min-max)')
                    ax.plot(ari_values, mean_chg, '-', color='red', lw=2.5,
                            label='Ensemble-mean change')
                    ax.axhline(0.0, color='black', lw=0.8)
                    ax.set_ylabel('Change vs baseline (%)', fontsize=9)
                else:
                    ax.fill_between(ari_values, hist.min(axis=0), hist.max(axis=0),
                                    color='blue', alpha=0.2, label='Baseline member range')
                    ax.fill_between(ari_values, futu.min(axis=0), futu.max(axis=0),
                                    color='red', alpha=0.2, label='Future member range')
                    ax.plot(ari_values, hist.mean(axis=0), '-', color='blue', lw=2.5,
                            label='Baseline ensemble mean')
                    ax.plot(ari_values, futu.mean(axis=0), '--', color='red', lw=2.5,
                            label=futu_mean_label)
                    ax.set_ylabel(f'Magnitude ({unit_label})', fontsize=9)

                ax.set_xticks(ari_ticks)
                ax.set_xticklabels(ari_tick_labels)
                ax.set_xlabel('ARI (years)', fontsize=9)
                ax.set_title(f'{name} {dur.replace("h", "-hour")}', fontsize=10, fontweight='bold')
                style_grid(ax)
                ax.legend(loc='best', fontsize=6)

            # Hide the last subplot if we have 7 durations
            if len(durations) == 7:
                axes[7].axis('off')

            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return "data:image/png;base64," + img

        return build('P', 'PREC-IDF'), build('NG', 'NG-IDF')

    except Exception as e:
        print(f"ERROR in generate_cesm_ensemble_mean_idf_plots: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return "", ""
