
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
    # data is a np array (3x7)
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



# def generate_fig(P_IDF_24h, P_IDF_48h, P_IDF_72h, NG_IDF_24h, NG_IDF_48h, NG_IDF_72h, P_fig_file, NG_fig_file):
#   # ----------------------------------------------------------------------------------------------------------------------------
#   # load the P-IDF and NG-IDF data from R output
#   # 1st row: estimated value; 2nd row: 5% quantile; 3rd row: 95% quantile
#   # 1-7 column: 2-yr, 5-yr, 10-yr, 25-yr, 50-yr, 100-yr, 500-yr

#   all_values = np.concatenate([P_IDF_24h.flatten(), P_IDF_48h.flatten(), P_IDF_72h.flatten(),NG_IDF_24h.flatten(), NG_IDF_48h.flatten(), NG_IDF_72h.flatten()])
#   all_values = all_values[~np.isnan(all_values)]  # remove NaNs
#   ymin, ymax = np.min(all_values), np.max(all_values)
#   margin = 0.05 * (ymax - ymin)  # add 5% padding
#   ymin -= margin
#   ymax += margin

#   # generate PREC-IDF figure
#   x = [2, 5, 10, 25, 50, 100, 500]
#   plt.fill_between(x, P_IDF_24h[1,:], P_IDF_24h[2,:],facecolor='blue',label='24h 90% Confidence Interval',alpha=0.2)
#   plt.semilogx(x, P_IDF_24h[0,:],label = '24h', color='blue')
#   plt.fill_between(x, P_IDF_48h[1,:], P_IDF_48h[2,:],facecolor='red',label='48h 90% Confidence Interval',alpha=0.2)
#   plt.semilogx(x, P_IDF_48h[0,:],label = '48h', color='red')
#   plt.fill_between(x, P_IDF_72h[1,:], P_IDF_72h[2,:],facecolor='green',label='72h 90% Confidence Interval',alpha=0.2)
#   plt.semilogx(x, P_IDF_72h[0,:],label = '72h', color='green')
#   plt.xticks(x,['2', '5', '10', '25', '50', '100', '500'] )
#   plt.grid(True)
#   plt.xlabel('Average Recurrence Interval (years)')
#   plt.ylabel('PREC-IDF (mm)')
#   plt.ylim(ymin, ymax)
#   plt.legend(loc='upper left')
#   plt.savefig(P_fig_file)
#   plt.close()

#   # generate NG-IDF figure
#   x = [2, 5, 10, 25, 50, 100, 500]
#   plt.fill_between(x, NG_IDF_24h[1,:], NG_IDF_24h[2,:],facecolor='blue',label='24h 90% Confidence Interval',alpha=0.2)
#   plt.semilogx(x, NG_IDF_24h[0,:],label = '24h', color='blue')
#   plt.fill_between(x, NG_IDF_48h[1,:], NG_IDF_48h[2,:],facecolor='red',label='48h 90% Confidence Interval',alpha=0.2)
#   plt.semilogx(x, NG_IDF_48h[0,:],label = '48h', color='red')
#   plt.fill_between(x, NG_IDF_72h[1,:], NG_IDF_72h[2,:],facecolor='green',label='72h 90% Confidence Interval',alpha=0.2)
#   plt.semilogx(x, NG_IDF_72h[0,:],label = '72h', color='green')
#   plt.xticks(x,['2', '5', '10', '25', '50', '100', '500'] )
#   plt.grid(True)
#   plt.xlabel('Average Recurrence Interval (years)')
#   plt.ylabel('NG-IDF (mm)')
#   plt.ylim(ymin, ymax)
#   plt.legend(loc='upper left')
#   plt.savefig(NG_fig_file)
#   plt.close()



#   image_data = base64.b64encode(open(P_fig_file, "rb").read())  # b'\x89PNG\r\n\x1a\n\x00\x00\x00\r'; String representation of bytes object includes leading "b" and quotes,  
#   image_data = image_data.decode('utf-8')
#   P_code = "data:image/png;base64," + image_data

#   image_data = base64.b64encode(open(NG_fig_file, "rb").read())  # b'\x89PNG\r\n\x1a\n\x00\x00\x00\r'; String representation of bytes object includes leading "b" and quotes,  
#   image_data = image_data.decode('utf-8')
#   NG_code = "data:image/png;base64," + image_data

#   return P_code, NG_code





def generate_fig(P_IDF_24h, P_IDF_48h, P_IDF_72h,
                 NG_IDF_24h, NG_IDF_48h, NG_IDF_72h,
                 fig24_file, fig48_file, fig72_file):



    # ------------------------------------------------------------------------------------------
    # The 51 probabilities used to compute the IDF curves
    probs = np.concatenate([np.arange(0.50, 1.00, 0.01), np.array([0.998])])

    # Convert probabilities to return periods (not used for plotting, only for reference)
    # T = 1 / (1 - p)

    # ARI tick positions
    ari_ticks = np.array([2, 5, 10, 25, 50, 100, 500])
    ari_tick_labels = ['2', '5', '10', '25', '50', '100', '500']

    # ------------------------------------------------------------------------------------------
    # Helper to plot a single duration
    def plot_single(P_IDF, NG_IDF, fig_file, duration_label):

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
        plt.ylabel('Magnitude (mm)')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.ylim(ymin, ymax)
        plt.legend(loc='upper left')

        plt.savefig(fig_file, dpi=150)
        plt.close()

        # Return base64 image string
        img = base64.b64encode(open(fig_file, "rb").read()).decode('utf-8')
        return "data:image/png;base64," + img

    # ------------------------------------------------------------------------------------------
    # Create three figures
    fig24_code = plot_single(P_IDF_24h, NG_IDF_24h, fig24_file, "24-hour")
    fig48_code = plot_single(P_IDF_48h, NG_IDF_48h, fig48_file, "48-hour")
    fig72_code = plot_single(P_IDF_72h, NG_IDF_72h, fig72_file, "72-hour")

    return fig24_code, fig48_code, fig72_code





