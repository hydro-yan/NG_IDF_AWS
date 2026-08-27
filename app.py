from flask import *
import pandas as pd
import numpy as np
import subprocess
import os
import os.path
from dhsvm_idf import generate_ng_idf
from extract_AM import extract_AM_data
import tempfile
import shutil
from gen_figures import (generate_fig, generate_figs_multiple, generate_am_timeseries_plots, 
                         generate_swe_timeseries_plot, generate_multi_scenario_idf_plots,
                         generate_multi_scenario_am_plots, generate_multi_scenario_swe_plot,
                         generate_cesm_ensemble_idf_plots, generate_cesm_ensemble_am_plots,
                         generate_cesm_ensemble_swe_plot)
from gen_spatial_figures import (generate_daymet_idf_figure, generate_daymet_alt_figure,
                                  generate_wrf_idf_figure, generate_wrf_alt_figure,
                                  generate_cesm_mid_century_idf_figure,
                                  generate_cesm_near_term_idf_figure,
                                  generate_cesm_mid_century_alt_figure,
                                  generate_cesm_near_term_alt_figure)






from concurrent.futures import ProcessPoolExecutor
import time


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
def structure_results(results):
    """
    Structure results dictionary with all available durations.
    
    This function takes the raw results from get_NG_IDF() and organizes them
    into a structured dictionary that's easy to use in HTML templates.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing:
        - 'durations': list of duration strings, e.g., ['1h', '3h', '6h', '12h', '24h', '48h', '72h']
        - 'idf_data': dict with keys like 'P_1h', 'NG_1h', etc. containing IDF curves (3x51 arrays)
        - 'fig_codes': dict with keys like 'fig_1h' containing base64-encoded figure strings
        - 'am_results': dict with keys like '1h', '24h' containing annual maximum data
    
    Returns:
    --------
    structured : dict
        Organized dictionary with:
        - All duration-specific data accessible via keys like 'P_1h', 'NG_3h', 'fig_12h'
        - Annual maximum data with REAL YEARS (no offsets)
        - Backward compatibility names for templates (e.g., 'fig24' points to 'fig_24h')
    
    Example structure for WRF/CESM (7 durations):
        {
            'durations': ['1h', '3h', '6h', '12h', '24h', '48h', '72h'],
            'P_1h': array(3x51),     # Precipitation IDF curve for 1-hour duration
            'NG_1h': array(3x51),    # Net Groundwater IDF curve for 1-hour duration
            'fig_1h': "data:image/png;base64,...",  # Base64 encoded figure
            'am_1h_P': array(Nx4),   # Annual max precip: [year, month, day, value]
            'am_1h_W': array(Nx4),   # Annual max net groundwater
            'am_1h_P_date': array(Nx3),  # Just the dates: [[2033, 10, 15], [2034, 11, 3], ...]
            ... (same for 3h, 6h, 12h, 24h, 48h, 72h)
        }
    """
    # Extract components from results dictionary
    idf_data = results['idf_data']        # IDF curves for all durations
    fig_codes = results['fig_codes']      # Base64-encoded figures
    am_results = results['am_results']    # Annual maximum data
    durations = results['durations']      # List of durations: ['1h', '3h', ...] or ['24h', '48h', '72h']
    
    # Helper function to extract just the date columns (Year, Month, Day) from AM data
    def process_dates(arr_in):
        """
        Extract and format date columns from annual maximum data.
        
        Input: array with shape (N, 4) where columns are [Year, Month, Day, Value]
        Output: array with shape (N, 3) where columns are [Year, Month, Day]
        
        Example:
            Input:  [[2033.5, 10.2, 15.8, 45.3], [2034.1, 11.0, 3.2, 52.1]]
            Output: [[2033, 10, 15], [2034, 11, 3]]
        
        NOTE: Uses REAL years from the data - no artificial offsets!
        """
        dates = np.round(arr_in[:, :3]).astype(int)  # Round and convert to integers
        return dates
    
    # Initialize structured dictionary with core data
    structured = {
        'durations': durations,      # List of all available durations
        'idf_data': idf_data,        # Raw IDF data dictionary
        'fig_codes': fig_codes       # Raw figure codes dictionary
    }
    
    # Loop through all durations and add data for each one
    # This makes data accessible via keys like 'P_1h', 'NG_3h', 'fig_12h', etc.
    for dur in durations:
        # IDF curve data (3 rows x 51 columns)
        # Row 0: Point estimates for 51 probabilities (0.50 to 0.99 plus 0.998)
        # Row 1: 5% confidence interval
        # Row 2: 95% confidence interval
        structured[f'P_{dur}'] = idf_data[f'P_{dur}']      # Precipitation IDF
        structured[f'NG_{dur}'] = idf_data[f'NG_{dur}']    # Net Groundwater IDF
        
        # Base64-encoded figure string (can be used directly in HTML <img src="...">)
        structured[f'fig_{dur}'] = fig_codes[f'fig_{dur}']
        
        # Annual maximum data
        # AM P: (N years x 4 columns: Year, Month, Day, Value)
        # AM W: (N years x 6 columns: Year, Month, Day, W_value, P_int, deltaSWE)
        # Example: [[2033, 10, 15, 45.3], [2034, 11, 3, 52.1], ...]
        structured[f'am_{dur}_P'] = np.round(am_results[dur]['P'], 2)        # Precipitation AM
        structured[f'am_{dur}_W'] = np.round(am_results[dur]['W_veg'], 2)    # Net Groundwater AM (6 columns)
        
        # Date-only arrays (N years x 3 columns: Year, Month, Day)
        # Example: [[2033, 10, 15], [2034, 11, 3], ...]
        structured[f'am_{dur}_P_date'] = process_dates(am_results[dur]['P'])
        structured[f'am_{dur}_W_date'] = process_dates(am_results[dur]['W_veg'])
        
        # Extract mechanism from W_veg array (column 7: 1=Rain, 2=Melt, 3=ROS)
        mechanism_values = am_results[dur]['W_veg'][:, 7].astype(int)
        mechanism_labels = []
        for mech in mechanism_values:
            if mech == 1:
                mechanism_labels.append("R")
            elif mech == 2:
                mechanism_labels.append("M")
            elif mech == 3:
                mechanism_labels.append("ROS")
            else:
                mechanism_labels.append("Unknown")
        structured[f'am_{dur}_W_mechanism'] = mechanism_labels
    
    # Add Snow Water Equivalent (SWE) data (only computed for 24h duration)
    structured['am_swe'] = np.round(am_results['swe'], 2)
    structured['am_swe_date'] = process_dates(am_results['swe'])
    
    # ========================================================================
    # BACKWARD COMPATIBILITY: Add alternative key names for existing templates
    # ========================================================================
    # Old templates use 'fig24', 'fig48', 'fig72' instead of 'fig_24h', etc.
    # We add both naming conventions so old and new templates both work
    
    # For ALL durations (including 1h, 3h, 6h, 12h for WRF/CESM)
    for dur in durations:
        # Extract the numeric part (e.g., '24' from '24h')
        dur_num = dur.replace('h', '')
        
        # Add alternative key names without underscore
        # e.g., 'fig24' points to same data as 'fig_24h'
        structured[f'fig{dur_num}'] = structured[f'fig_{dur}']
        structured[f'P_{dur_num}h'] = structured[f'P_{dur}']
        structured[f'NG_{dur_num}h'] = structured[f'NG_{dur}']
        structured[f'am_{dur_num}h_P'] = structured[f'am_{dur}_P']
        structured[f'am_{dur_num}h_W'] = structured[f'am_{dur}_W']
        structured[f'am_{dur_num}h_P_date'] = structured[f'am_{dur}_P_date']
        structured[f'am_{dur_num}h_W_date'] = structured[f'am_{dur}_W_date']
    
    return structured





# ----------------------------------------------------------------------------------------------------------------------------
app = Flask(__name__)

# Grid-cell vegetation data define the supported Interior Alaska analysis area.
# The source grid is approximately 1 km, so locations farther than 1.5 km from
# a grid-cell centre are outside the model domain rather than being assigned an
# arbitrary vegetation class.
LULC_GRID_PATH = os.path.join(app.root_path, "static", "gridcell_current_lulc.csv")
LULC_GRID = pd.read_csv(LULC_GRID_PATH)
LULC_GRID_COORDS = LULC_GRID[["latitude", "longitude"]].to_numpy(dtype=float)
SPATIAL_STUDY_BOUNDS = {
    "north": float(LULC_GRID["latitude"].max()),
    "south": float(LULC_GRID["latitude"].min()),
    "west": float(LULC_GRID["longitude"].min()),
    "east": float(LULC_GRID["longitude"].max()),
}
MAX_GRIDCELL_DISTANCE_KM = 1.5


@app.context_processor
def inject_spatial_study_bounds():
    """Make the spatial-map study extent available to the input page."""
    return {"spatial_study_bounds": SPATIAL_STUDY_BOUNDS}


LULC_TO_DHSVM_CODE = {
    "open": "1",
    "evergreen": "2",
    "deciduous": "3",
    "mixed": "4",
    "crop": "5",
    "grass": "6",
    "shrub": "7",
    "pasture": "8",
    "wetland": "9",
}


def lookup_gridcell_lulc(latitude, longitude):
    """Return the nearest supported grid-cell and its vegetation attributes.

    Distances use an equirectangular approximation, which is accurate at the
    sub-kilometre scale of the Interior Alaska grid and avoids a GIS dependency.
    """
    latitude = float(latitude)
    longitude = float(longitude)
    lat_scale_km = 111.32
    lon_scale_km = 111.32 * np.cos(np.deg2rad(latitude))
    delta_lat = (LULC_GRID_COORDS[:, 0] - latitude) * lat_scale_km
    delta_lon = (LULC_GRID_COORDS[:, 1] - longitude) * lon_scale_km
    nearest_index = int(np.argmin(delta_lat ** 2 + delta_lon ** 2))
    distance_km = float(np.hypot(delta_lat[nearest_index], delta_lon[nearest_index]))

    if distance_km > MAX_GRIDCELL_DISTANCE_KM:
        return None, distance_km
    return LULC_GRID.iloc[nearest_index], distance_km


@app.get("/api/gridcell-lulc")
def gridcell_lulc():
    """Validate a point against the Interior Alaska grid and return its defaults."""
    try:
        latitude = float(request.args["lat"])
        longitude = float(request.args["lon"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Valid latitude and longitude are required."), 400

    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return jsonify(error="Latitude or longitude is outside its valid range."), 400

    gridcell, distance_km = lookup_gridcell_lulc(latitude, longitude)
    if gridcell is None:
        return jsonify(
            inside_boundary=False,
            message="This location is outside the supported Interior Alaska boundary.",
            distance_to_grid_km=round(distance_km, 2),
        )

    land_cover = str(gridcell["dominant_LULC"]).strip().lower()
    lai = float(gridcell["max_LAI"])
    height = float(gridcell["mean_height_m"])

    # Missing vegetation data represent Open land cover, regardless of the
    # LULC label in the source CSV. Open always uses zero vegetation values.
    if not np.isfinite(lai) or not np.isfinite(height):
        land_cover = "open"
    if land_cover == "open":
        lai = 0.0
        height = 0.0

    return jsonify(
        inside_boundary=True,
        latitude=round(float(gridcell["latitude"]), 5),
        longitude=round(float(gridcell["longitude"]), 5),
        land_cover=land_cover,
        dhsvm_land_cover=LULC_TO_DHSVM_CODE[land_cover],
        lai=round(lai, 2),
        height=round(height, 2),
        distance_to_grid_km=round(distance_km, 2),
    )


@app.get("/api/spatial-study-boundary")
def spatial_study_boundary():
    """Return the bounding extent of the precomputed spatial-map grid."""
    return jsonify(SPATIAL_STUDY_BOUNDS)

# Access control token for external collaborators
SHARED_TOKEN = os.environ.get("SHARED_TOKEN", "pnnl_collab_secure")

# Add security headers to all responses
@app.after_request
def add_security_headers(response):
    """
    Add security headers to all HTTP responses for WAGER compliance.
    These headers help protect against common web vulnerabilities.
    """
    # Prevent MIME type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # Enable XSS protection in browsers
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Prevent clickjacking attacks
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    
    # Enforce HTTPS (if using HTTPS)
    # Uncomment the next line if your site is served over HTTPS
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    # Control referrer information
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    # Content Security Policy - adjust as needed for your application
    response.headers['Content-Security-Policy'] = (
        "default-src 'self' 'unsafe-inline' 'unsafe-eval' data:; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com; "
        "style-src 'self' 'unsafe-inline' https://unpkg.com; "
        "img-src 'self' data: https://tile.openstreetmap.org;"
    )
    
    # Permissions Policy (formerly Feature-Policy)
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    
    return response

@app.route("/NG_IDF", methods=["GET", "POST"])
def NG_IDF():
    # 1. Grab the user's real IP address
    user_ip = request.remote_addr
    
    # 2. Let PNNL staff bypass the token automatically
    is_pnnl_staff = user_ip.startswith('10.15.')
    
    # 3. Grab the token from the URL
    user_token = request.args.get('auth_token')
    
    # 4. Block anyone who isn't PNNL staff AND doesn't have the token
    if not is_pnnl_staff and user_token != SHARED_TOKEN:
        abort(403)

    if request.method == "POST":

        input_data = dict()

        # Determine which mode the form was submitted from: "point" or "spatial"
        tool_mode = request.form.get("mode", "point")

        if tool_mode == "spatial":
            # Spatial Map Mode inputs (predefined land cover, precomputed spatial maps)
            # land_cover: "open" | "evergreen" | "deciduous"
            land_cover = request.form.get("land_cover", "open")

            # The HTML form submits long-form scenario names; normalize them to
            # the short codes used by the backend/spatial map lookup:
            #   "historical"         -> "daymet"
            #   "future_wrf"         -> "wrf"
            #   "future_cesm_near"   -> "cesm_near"
            #   "future_cesm_mid"    -> "cesm_mid"
            spatial_scenario_raw = request.form.get("spatial_scenario", "future_cesm_near")
            scenario_map = {
                "historical": "daymet",
                "future_wrf": "wrf",
                "future_cesm_near": "cesm_near",
                "future_cesm_mid": "cesm_mid",
            }
            spatial_scenario = scenario_map.get(spatial_scenario_raw, spatial_scenario_raw)

            # duration: string, "1" to "72" (hours)
            duration = request.form.get("duration", "24")

            # ari: string, "2" to "500" (years)
            ari = request.form.get("ari", "25")

            print(f"DEBUG: Spatial Map Mode - land_cover={land_cover}, "
                  f"spatial_scenario={spatial_scenario} (raw={spatial_scenario_raw}), "
                  f"duration={duration}, ari={ari}", flush=True)

            # Human-readable label for the scenario, shown in the results page
            scenario_label_map = {
                "daymet": "Historical Weather (Daymet)",
                "wrf": "Historical Intensification (2034–2065)",
                "cesm_near": "Near-Term Design Lifespan (2034-2049)",
                "cesm_mid": "Mid-Century Design Lifespan (2034–2065)",
            }
            spatial_scenario_label = scenario_label_map.get(spatial_scenario, spatial_scenario)

            # Only the "daymet" (historical) spatial scenario is currently wired up
            # to a figure-generation function. Other scenarios will show a
            # "not yet available" notice on the results page.
            if spatial_scenario == "daymet":
                try:
                    # Generate the spatial comparison figure and save it to a
                    # temporary PNG file. generate_daymet_idf_figure() reads
                    # the PNG back and returns it as a base64 data URI, which
                    # we pass straight into the HTML template's <img> tag.
                    with tempfile.TemporaryDirectory() as td:
                        fig_file = os.path.join(
                            td, f"daymet_spatial_{land_cover}_{duration}h_{ari}yr.png"
                        )
                        fig_spatial = generate_daymet_idf_figure(
                            land_cover, spatial_scenario, duration, ari, fig_file
                        )

                        # Also generate the Active Layer Thickness (ALT) spatial
                        # map for the same land cover, saved to its own temp PNG
                        # and returned as a base64 data URI as well.
                        fig_alt_file = os.path.join(
                            td, f"daymet_alt_{land_cover}.png"
                        )
                        fig_alt = generate_daymet_alt_figure(
                            land_cover, spatial_scenario, duration, ari, fig_alt_file
                        )

                    return render_template(
                        "out_spatial.html",
                        land_cover=land_cover,
                        spatial_scenario=spatial_scenario,
                        spatial_scenario_label=spatial_scenario_label,
                        duration=duration,
                        ari=ari,
                        fig_spatial=fig_spatial,
                        fig_alt=fig_alt,
                    )
                except FileNotFoundError as exc:
                    print(f"ERROR: Spatial map data not found: {exc}", flush=True)
                    return render_template(
                        "out_spatial.html",
                        land_cover=land_cover,
                        spatial_scenario=spatial_scenario,
                        spatial_scenario_label=spatial_scenario_label,
                        duration=duration,
                        ari=ari,
                        error=(f"No precomputed spatial map data available for "
                               f"{land_cover} land cover at {duration}-hour / "
                               f"{ari}-year for this scenario."),
                    )
            elif spatial_scenario == "wrf":
                try:
                    # Generate the 3x3 WRF IDF comparison figure (historical
                    # magnitude + Medium/High future % change) and save it to
                    # a temporary PNG file. generate_wrf_idf_figure() reads
                    # the PNG back and returns it as a base64 data URI, which
                    # we pass straight into the HTML template's <img> tag.
                    with tempfile.TemporaryDirectory() as td:
                        fig_file = os.path.join(
                            td, f"wrf_spatial_{land_cover}_{duration}h_{ari}yr.png"
                        )
                        fig_spatial_wrf = generate_wrf_idf_figure(
                            land_cover, spatial_scenario, duration, ari, fig_file
                        )

                        # Also generate the WRF Active Layer Thickness (ALT)
                        # spatial map (historical mean + Medium/High future
                        # change) for the same land cover, saved to its own
                        # temp PNG and returned as a base64 data URI as well.
                        fig_alt_file = os.path.join(
                            td, f"wrf_alt_{land_cover}.png"
                        )
                        fig_alt_wrf = generate_wrf_alt_figure(
                            land_cover, spatial_scenario, duration, ari, fig_alt_file
                        )

                    return render_template(
                        "out_spatial_wrf.html",
                        land_cover=land_cover,
                        spatial_scenario=spatial_scenario,
                        spatial_scenario_label=spatial_scenario_label,
                        duration=duration,
                        ari=ari,
                        fig_spatial_wrf=fig_spatial_wrf,
                        fig_alt_wrf=fig_alt_wrf,
                    )

                except (FileNotFoundError, ValueError) as exc:
                    print(f"ERROR: WRF spatial map data not found: {exc}", flush=True)
                    return render_template(
                        "out_spatial_wrf.html",
                        land_cover=land_cover,
                        spatial_scenario=spatial_scenario,
                        spatial_scenario_label=spatial_scenario_label,
                        duration=duration,
                        ari=ari,
                        error=(f"No precomputed WRF spatial map data available for "
                               f"{land_cover} land cover at {duration}-hour / "
                               f"{ari}-year for this scenario. ({exc})"),
                    )
            elif spatial_scenario == "cesm_mid":
                try:
                    # Generate the 2x3 CESM mid-century ensemble-mean IDF
                    # figure (historical magnitude + ensemble-mean future %
                    # change) and save it to a temporary PNG file.
                    # generate_cesm_mid_century_idf_figure() reads the PNG
                    # back and returns it as a base64 data URI, which we pass
                    # straight into the HTML template's <img> tag.
                    with tempfile.TemporaryDirectory() as td:
                        fig_file = os.path.join(
                            td, f"cesm_spatial_{land_cover}_{duration}h_{ari}yr.png"
                        )
                        member_fig_file = os.path.join(
                            td, f"cesm_members_{land_cover}_{duration}h_{ari}yr.png"
                        )
                        fig_spatial_cesm, fig_spatial_cesm_members = generate_cesm_mid_century_idf_figure(
                            land_cover, spatial_scenario, duration, ari,
                            fig_file=fig_file, member_fig_file=member_fig_file
                        )

                        # Also generate the CESM Mid-Century Active Layer
                        # Thickness (ALT) spatial map (2x4 ensemble-mean +
                        # individual-member change figure) for the same
                        # land cover, saved to its own temp PNG and returned
                        # as a base64 data URI as well.
                        alt_fig_file = os.path.join(
                            td, f"cesm_alt_{land_cover}.png"
                        )
                        fig_spatial_cesm_alt = generate_cesm_mid_century_alt_figure(
                            land_cover, spatial_scenario, duration, ari,
                            fig_file=alt_fig_file
                        )

                    return render_template(
                        "out_spatial_cesm.html",
                        land_cover=land_cover,
                        spatial_scenario=spatial_scenario,
                        spatial_scenario_label=spatial_scenario_label,
                        duration=duration,
                        ari=ari,
                        fig_spatial_cesm=fig_spatial_cesm,
                        fig_spatial_cesm_members=fig_spatial_cesm_members,
                        fig_spatial_cesm_alt=fig_spatial_cesm_alt,
                    )



                except (FileNotFoundError, ValueError) as exc:
                    print(f"ERROR: CESM spatial map data not found: {exc}", flush=True)
                    return render_template(
                        "out_spatial_cesm.html",
                        land_cover=land_cover,
                        spatial_scenario=spatial_scenario,
                        spatial_scenario_label=spatial_scenario_label,
                        duration=duration,
                        ari=ari,
                        error=(f"No precomputed CESM mid-century spatial map data "
                               f"available for {land_cover} land cover at "
                               f"{duration}-hour / {ari}-year for this scenario. ({exc})"),
                    )
            elif spatial_scenario == "cesm_near":
                try:
                    # Generate the 2x3 CESM near-term ensemble-mean IDF
                    # figure (historical magnitude + ensemble-mean future %
                    # change) and save it to a temporary PNG file.
                    # generate_cesm_near_term_idf_figure() reads the PNG
                    # back and returns it as a base64 data URI, which we pass
                    # straight into the HTML template's <img> tag.
                    with tempfile.TemporaryDirectory() as td:
                        fig_file = os.path.join(
                            td, f"cesm_near_spatial_{land_cover}_{duration}h_{ari}yr.png"
                        )
                        member_fig_file = os.path.join(
                            td, f"cesm_near_members_{land_cover}_{duration}h_{ari}yr.png"
                        )
                        fig_spatial_cesm_near, fig_spatial_cesm_near_members = generate_cesm_near_term_idf_figure(
                            land_cover, spatial_scenario, duration, ari,
                            fig_file=fig_file, member_fig_file=member_fig_file
                        )

                        # Also generate the CESM Near-Term Active Layer
                        # Thickness (ALT) spatial map (2x4 ensemble-mean +
                        # individual-member change figure) for the same
                        # land cover, saved to its own temp PNG and returned
                        # as a base64 data URI as well.
                        alt_fig_file = os.path.join(
                            td, f"cesm_near_alt_{land_cover}.png"
                        )
                        fig_spatial_cesm_near_alt = generate_cesm_near_term_alt_figure(
                            land_cover, spatial_scenario, duration, ari,
                            fig_file=alt_fig_file
                        )

                    return render_template(
                        "out_spatial_cesm_near_term.html",
                        land_cover=land_cover,
                        spatial_scenario=spatial_scenario,
                        spatial_scenario_label=spatial_scenario_label,
                        duration=duration,
                        ari=ari,
                        fig_spatial_cesm_near=fig_spatial_cesm_near,
                        fig_spatial_cesm_near_members=fig_spatial_cesm_near_members,
                        fig_spatial_cesm_near_alt=fig_spatial_cesm_near_alt,
                    )


                except (FileNotFoundError, ValueError) as exc:
                    print(f"ERROR: CESM near-term spatial map data not found: {exc}", flush=True)
                    return render_template(
                        "out_spatial_cesm_near_term.html",
                        land_cover=land_cover,
                        spatial_scenario=spatial_scenario,
                        spatial_scenario_label=spatial_scenario_label,
                        duration=duration,
                        ari=ari,
                        error=(f"No precomputed CESM near-term spatial map data "
                               f"available for {land_cover} land cover at "
                               f"{duration}-hour / {ari}-year for this scenario. ({exc})"),
                    )
            else:
                # Unknown scenario code
                return render_template(
                    "out_spatial.html",
                    land_cover=land_cover,
                    spatial_scenario=spatial_scenario,
                    spatial_scenario_label=spatial_scenario_label,
                    duration=duration,
                    ari=ari,
                    error=("Spatial map figures for this scenario are not yet "
                           "available. Currently Historical Weather (Daymet), "
                           "Future Weather (WRF), Future Weather "
                           "(CESM Near-Term), and Future Weather "
                           "(CESM Mid-Century) are supported."),
                )



        # user input

        try:
            for i in range(1, 11):
                key = f"Value{i}"
                input_data[f"value{i}"] = float(request.form[key])
        except (KeyError, TypeError, ValueError):
            return render_template(
                "NG_IDF.html",
                location_error=(
                    "Complete all site input parameters with numeric values "
                    "before submitting."
                ),
            )

        lat = input_data['value1']
        lon = input_data['value2']
        scenario = request.form.get("scenario", "historical")

        # Also enforce the model boundary server-side so a manually crafted
        # form request cannot run an unsupported location.
        gridcell, _ = lookup_gridcell_lulc(lat, lon)
        if gridcell is None:
            return render_template(
                "NG_IDF.html",
                location_error="This location is outside the supported Interior Alaska boundary. Please select a location inside the displayed study area.",
            )


        # Debug: Print scenario value with quotes to see exact string
        print(f"DEBUG: scenario from form: '{scenario}' (type: {type(scenario)})", flush=True)
        print(f"DEBUG: scenario repr: {repr(scenario)}", flush=True)
        print(f"DEBUG: scenario bytes: {scenario.encode('utf-8')}", flush=True)
        
        # Strip any whitespace that might have been added
        scenario = scenario.strip()
        print(f"DEBUG: scenario after strip: '{scenario}'", flush=True)
        
        # ---------------------------------------------
        # BRANCH: HISTORICAL, WRF FUTURE, OR CESM FUTURE
        # ---------------------------------------------

        if scenario == "historical":

            # Run historical Daymet workflow
            results = get_NG_IDF(input_data, forcing_type="Daymet")

            data = structure_results(results)
            
            # Generate time series plots for AM P, W, and SWE
            with tempfile.TemporaryDirectory() as td:
                fig_am_p_file = os.path.join(td, 'am_timeseries_p.png')
                fig_am_w_file = os.path.join(td, 'am_timeseries_w.png')
                fig_swe_file = os.path.join(td, 'am_swe_timeseries.png')
                
                # Generate AM P and W time series plots
                durations = results['durations']  # ['24h', '48h', '72h'] for Daymet
                fig_am_p, fig_am_w = generate_am_timeseries_plots(
                    results['am_results'], 
                    durations, 
                    fig_am_p_file, 
                    fig_am_w_file
                )
                
                # Generate SWE time series plot
                fig_swe = generate_swe_timeseries_plot(
                    results['am_results']['swe'],
                    fig_swe_file
                )
                
                # Add time series figures to data
                data['fig_am_p'] = fig_am_p
                data['fig_am_w'] = fig_am_w
                data['fig_swe'] = fig_swe

            # Render historical results using existing out.html
            return render_template("out.html",
                                   lat=lat, lon=lon,
                                   **data) # Unpack all keys in data to template variables

        elif scenario == "future_wrf":

            # Run WRF Multi-Scenario Workflow IN PARALLEL (3 processors)
            print("Starting WRF parallel processing with 3 scenarios...", flush=True)
            start_time = time.time()
            
            # Define the 3 scenarios to run in parallel
            wrf_scenarios = ["WRF_historical", "WRF_medium", "WRF_high"]
            
            # Use ProcessPoolExecutor to run 3 scenarios in parallel
            with ProcessPoolExecutor(max_workers=3) as executor:
                # Submit all 3 jobs to the executor
                futures = {
                    executor.submit(get_NG_IDF, input_data, forcing_type): forcing_type
                    for forcing_type in wrf_scenarios
                }
                
                # Collect results as they complete
                results_dict = {}
                for future in futures:
                    forcing_type = futures[future]
                    try:
                        result = future.result()
                        results_dict[forcing_type] = result
                        print(f"Completed: {forcing_type}", flush=True)
                    except Exception as exc:
                        print(f"ERROR in {forcing_type}: {exc}", flush=True)
                        raise
            
            elapsed_time = time.time() - start_time
            print(f"WRF parallel processing completed in {elapsed_time:.2f} seconds", flush=True)
            
            # Extract and structure results
            res_hist = results_dict["WRF_historical"]
            hist_data = structure_results(res_hist)
            
            res_med = results_dict["WRF_medium"]
            med_data = structure_results(res_med)
            
            res_high = results_dict["WRF_high"]
            high_data = structure_results(res_high)

            # 4. Generate multi-scenario comparison plots
            durations = hist_data['durations']  # ['1h', '3h', '6h', '12h', '24h', '48h', '72h']
            
            # Generate plots directly without temp files (plots return base64 strings)
            fig_prec_comp, fig_ng_comp = generate_multi_scenario_idf_plots(
                hist_data, med_data, high_data, durations, None, None
            )
            
            fig_am_p_comp, fig_am_w_comp = generate_multi_scenario_am_plots(
                res_hist['am_results'], res_med['am_results'], res_high['am_results'],
                durations, None, None
            )
            
            fig_swe_comp = generate_multi_scenario_swe_plot(
                res_hist['am_results']['swe'],
                res_med['am_results']['swe'],
                res_high['am_results']['swe'],
                None
            )

            # 5. Build Comprehensive Summary Table
            # Indices: 2yr=0, 5yr=30, 10yr=40, 25yr=46, 50yr=48, 100yr=49, 500yr=50
            aris = [2, 5, 10, 25, 50, 100, 500]
            indices = [0, 30, 40, 46, 48, 49, 50]
            
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
                                   summary_rows=summary_rows,
                                   fig_prec_comp=fig_prec_comp,
                                   fig_ng_comp=fig_ng_comp,
                                   fig_am_p_comp=fig_am_p_comp,
                                   fig_am_w_comp=fig_am_w_comp,
                                   fig_swe_comp=fig_swe_comp)

        elif scenario == "future_cesm":

            # Run CESM Multi-Scenario Workflow IN PARALLEL (8 processors)
            # 4 historical + 4 future ensemble members
            print("Starting CESM parallel processing with 8 ensemble members...", flush=True)
            start_time = time.time()
            
            # Define all 8 CESM scenarios to run in parallel
            cesm_scenarios = [
                "CESM_hist_LE2", "CESM_futu_LE2",
                "CESM_hist_LE4", "CESM_futu_LE4",
                "CESM_hist_LE7", "CESM_futu_LE7",
                "CESM_hist_LE9", "CESM_futu_LE9"
            ]
            
            # Use ProcessPoolExecutor to run all 8 scenarios in parallel
            with ProcessPoolExecutor(max_workers=8) as executor:
                # Submit all 8 jobs to the executor
                futures = {
                    executor.submit(get_NG_IDF, input_data, forcing_type): forcing_type
                    for forcing_type in cesm_scenarios
                }
                
                # Collect results as they complete
                results_dict = {}
                for future in futures:
                    forcing_type = futures[future]
                    try:
                        result = future.result()
                        results_dict[forcing_type] = result
                        print(f"Completed: {forcing_type}", flush=True)
                    except Exception as exc:
                        print(f"ERROR in {forcing_type}: {exc}", flush=True)
                        raise
            
            elapsed_time = time.time() - start_time
            print(f"CESM parallel processing completed in {elapsed_time:.2f} seconds", flush=True)
            
            # Extract and structure results for each ensemble member
            res_hist_le2 = results_dict["CESM_hist_LE2"]
            hist_le2_data = structure_results(res_hist_le2)
            
            res_futu_le2 = results_dict["CESM_futu_LE2"]
            futu_le2_data = structure_results(res_futu_le2)
            
            res_hist_le4 = results_dict["CESM_hist_LE4"]
            hist_le4_data = structure_results(res_hist_le4)
            
            res_futu_le4 = results_dict["CESM_futu_LE4"]
            futu_le4_data = structure_results(res_futu_le4)
            
            res_hist_le7 = results_dict["CESM_hist_LE7"]
            hist_le7_data = structure_results(res_hist_le7)
            
            res_futu_le7 = results_dict["CESM_futu_LE7"]
            futu_le7_data = structure_results(res_futu_le7)
            
            res_hist_le9 = results_dict["CESM_hist_LE9"]
            hist_le9_data = structure_results(res_hist_le9)
            
            res_futu_le9 = results_dict["CESM_futu_LE9"]
            futu_le9_data = structure_results(res_futu_le9)

            # Get durations from CESM data (7 durations: 1h, 3h, 6h, 12h, 24h, 48h, 72h)
            durations = hist_le2_data['durations']
            
            # Generate CESM multi-ensemble comparison plots
            print("Generating CESM ensemble comparison plots...", flush=True)
            fig_prec_comp, fig_ng_comp = generate_cesm_ensemble_idf_plots(
                hist_le2_data, hist_le4_data, hist_le7_data, hist_le9_data,
                futu_le2_data, futu_le4_data, futu_le7_data, futu_le9_data,
                durations, None, None
            )
            
            fig_am_p_comp, fig_am_w_comp = generate_cesm_ensemble_am_plots(
                res_hist_le2['am_results'], res_hist_le4['am_results'], 
                res_hist_le7['am_results'], res_hist_le9['am_results'],
                res_futu_le2['am_results'], res_futu_le4['am_results'],
                res_futu_le7['am_results'], res_futu_le9['am_results'],
                durations, None, None
            )
            
            fig_swe_comp = generate_cesm_ensemble_swe_plot(
                res_hist_le2['am_results']['swe'], res_hist_le4['am_results']['swe'],
                res_hist_le7['am_results']['swe'], res_hist_le9['am_results']['swe'],
                res_futu_le2['am_results']['swe'], res_futu_le4['am_results']['swe'],
                res_futu_le7['am_results']['swe'], res_futu_le9['am_results']['swe'],
                None
            )
            
            # Build Comprehensive Summary Table for CESM
            aris = [2, 5, 10, 25, 50, 100, 500]
            indices = [0, 30, 40, 46, 48, 49, 50]
            
            summary_rows = []

            for dur in durations:
                key = f"NG_{dur}"
                
                # Get arrays for all ensemble members
                hist_le2_arr = hist_le2_data[key]
                futu_le2_arr = futu_le2_data[key]
                hist_le4_arr = hist_le4_data[key]
                futu_le4_arr = futu_le4_data[key]
                hist_le7_arr = hist_le7_data[key]
                futu_le7_arr = futu_le7_data[key]
                hist_le9_arr = hist_le9_data[key]
                futu_le9_arr = futu_le9_data[key]

                for ari, idx in zip(aris, indices):
                    # LE2
                    val_hist_le2 = hist_le2_arr[0, idx]
                    val_futu_le2 = futu_le2_arr[0, idx]
                    diff_le2 = val_futu_le2 - val_hist_le2
                    pct_le2 = (diff_le2 / val_hist_le2 * 100) if val_hist_le2 != 0 else 0.0

                    # LE4
                    val_hist_le4 = hist_le4_arr[0, idx]
                    val_futu_le4 = futu_le4_arr[0, idx]
                    diff_le4 = val_futu_le4 - val_hist_le4
                    pct_le4 = (diff_le4 / val_hist_le4 * 100) if val_hist_le4 != 0 else 0.0

                    # LE7
                    val_hist_le7 = hist_le7_arr[0, idx]
                    val_futu_le7 = futu_le7_arr[0, idx]
                    diff_le7 = val_futu_le7 - val_hist_le7
                    pct_le7 = (diff_le7 / val_hist_le7 * 100) if val_hist_le7 != 0 else 0.0

                    # LE9
                    val_hist_le9 = hist_le9_arr[0, idx]
                    val_futu_le9 = futu_le9_arr[0, idx]
                    diff_le9 = val_futu_le9 - val_hist_le9
                    pct_le9 = (diff_le9 / val_hist_le9 * 100) if val_hist_le9 != 0 else 0.0

                    summary_rows.append({
                        "duration": dur,
                        "ari": ari,
                        "hist_le2": val_hist_le2,
                        "futu_le2": val_futu_le2,
                        "diff_le2": diff_le2,
                        "pct_le2": pct_le2,
                        "hist_le4": val_hist_le4,
                        "futu_le4": val_futu_le4,
                        "diff_le4": diff_le4,
                        "pct_le4": pct_le4,
                        "hist_le7": val_hist_le7,
                        "futu_le7": val_futu_le7,
                        "diff_le7": diff_le7,
                        "pct_le7": pct_le7,
                        "hist_le9": val_hist_le9,
                        "futu_le9": val_futu_le9,
                        "diff_le9": diff_le9,
                        "pct_le9": pct_le9
                    })

            return render_template("out_cesm.html",
                                   lat=lat, lon=lon,
                                   hist_le2=hist_le2_data,
                                   futu_le2=futu_le2_data,
                                   hist_le4=hist_le4_data,
                                   futu_le4=futu_le4_data,
                                   hist_le7=hist_le7_data,
                                   futu_le7=futu_le7_data,
                                   hist_le9=hist_le9_data,
                                   futu_le9=futu_le9_data,
                                   summary_rows=summary_rows,
                                   fig_prec_comp=fig_prec_comp,
                                   fig_ng_comp=fig_ng_comp,
                                   fig_am_p_comp=fig_am_p_comp,
                                   fig_am_w_comp=fig_am_w_comp,
                                   fig_swe_comp=fig_swe_comp)

        elif scenario == "future_cesm_near":

            # Run CESM Near-Term (16-year) Multi-Scenario Workflow IN PARALLEL (8 processors)
            # Historical: 2006-2021 (16 years), Future: 2034-2049 (16 years)
            # 4 historical + 4 future ensemble members
            print("Starting CESM Near-Term parallel processing with 8 ensemble members...", flush=True)
            print("Historical period: 2006-2021 (16 years)", flush=True)
            print("Future period: 2034-2049 (16 years)", flush=True)
            start_time = time.time()
            
            # Define all 8 CESM scenarios to run in parallel
            cesm_scenarios = [
                "CESM_hist_LE2", "CESM_futu_LE2",
                "CESM_hist_LE4", "CESM_futu_LE4",
                "CESM_hist_LE7", "CESM_futu_LE7",
                "CESM_hist_LE9", "CESM_futu_LE9"
            ]
            
            # Use ProcessPoolExecutor to run all 8 scenarios in parallel
            # Pass year_range parameter to filter data
            with ProcessPoolExecutor(max_workers=8) as executor:
                # Submit all 8 jobs to the executor with year range filtering
                futures = {}
                for forcing_type in cesm_scenarios:
                    # Determine year range based on scenario type
                    # Note: We need to be careful with water year boundaries
                    # Water Year N starts Oct 1 of year N-1 and ends Sep 30 of year N
                    # To get WY 2006-2021 (16 WY): need data from Oct 2004 to Sep 2021
                    # To get WY 2034-2049 (16 WY): need data from Oct 2032 to Sep 2049
                    # After R removes first year: WY 2007-2021 (15 WY) and WY 2035-2049 (15 WY)
                    # To get 16 WY after R filtering, we need 17 WY before, which means:
                    # Historical: 2004-2021 → WY 2005-2021 (17 WY) → R removes 2005 → WY 2006-2021 (16 WY)
                    # Future: 2032-2048 → WY 2033-2049 (17 WY) → R removes 2033 → WY 2034-2049 (16 WY)
                    if "hist" in forcing_type:
                        year_range = ((2005, 1, 1), (2021, 9, 30))  # 2005/1/1 to 2021/9/30 → WY 2006-2021 (16 WY after R filtering)
                    else:  # "futu" in forcing_type
                        year_range = ((2033, 1, 1), (2049, 9, 30))  # 2033/1/1 to 2049/9/30 → WY 2034-2049 (16 WY after R filtering)
                    
                    futures[executor.submit(get_NG_IDF, input_data, forcing_type, year_range)] = forcing_type
                
                # Collect results as they complete
                results_dict = {}
                for future in futures:
                    forcing_type = futures[future]
                    try:
                        result = future.result()
                        results_dict[forcing_type] = result
                        print(f"Completed: {forcing_type}", flush=True)
                    except Exception as exc:
                        print(f"ERROR in {forcing_type}: {exc}", flush=True)
                        raise
            
            elapsed_time = time.time() - start_time
            print(f"CESM Near-Term parallel processing completed in {elapsed_time:.2f} seconds", flush=True)
            
            # Extract and structure results for each ensemble member
            res_hist_le2 = results_dict["CESM_hist_LE2"]
            hist_le2_data = structure_results(res_hist_le2)
            
            res_futu_le2 = results_dict["CESM_futu_LE2"]
            futu_le2_data = structure_results(res_futu_le2)
            
            res_hist_le4 = results_dict["CESM_hist_LE4"]
            hist_le4_data = structure_results(res_hist_le4)
            
            res_futu_le4 = results_dict["CESM_futu_LE4"]
            futu_le4_data = structure_results(res_futu_le4)
            
            res_hist_le7 = results_dict["CESM_hist_LE7"]
            hist_le7_data = structure_results(res_hist_le7)
            
            res_futu_le7 = results_dict["CESM_futu_LE7"]
            futu_le7_data = structure_results(res_futu_le7)
            
            res_hist_le9 = results_dict["CESM_hist_LE9"]
            hist_le9_data = structure_results(res_hist_le9)
            
            res_futu_le9 = results_dict["CESM_futu_LE9"]
            futu_le9_data = structure_results(res_futu_le9)

            # Get durations from CESM data (7 durations: 1h, 3h, 6h, 12h, 24h, 48h, 72h)
            durations = hist_le2_data['durations']
            
            # Generate CESM multi-ensemble comparison plots
            print("Generating CESM Near-Term ensemble comparison plots...", flush=True)
            fig_prec_comp, fig_ng_comp = generate_cesm_ensemble_idf_plots(
                hist_le2_data, hist_le4_data, hist_le7_data, hist_le9_data,
                futu_le2_data, futu_le4_data, futu_le7_data, futu_le9_data,
                durations, None, None
            )
            
            fig_am_p_comp, fig_am_w_comp = generate_cesm_ensemble_am_plots(
                res_hist_le2['am_results'], res_hist_le4['am_results'], 
                res_hist_le7['am_results'], res_hist_le9['am_results'],
                res_futu_le2['am_results'], res_futu_le4['am_results'],
                res_futu_le7['am_results'], res_futu_le9['am_results'],
                durations, None, None
            )
            
            fig_swe_comp = generate_cesm_ensemble_swe_plot(
                res_hist_le2['am_results']['swe'], res_hist_le4['am_results']['swe'],
                res_hist_le7['am_results']['swe'], res_hist_le9['am_results']['swe'],
                res_futu_le2['am_results']['swe'], res_futu_le4['am_results']['swe'],
                res_futu_le7['am_results']['swe'], res_futu_le9['am_results']['swe'],
                None
            )
            
            # Build Comprehensive Summary Table for CESM Near-Term
            aris = [2, 5, 10, 25, 50, 100, 500]
            indices = [0, 30, 40, 46, 48, 49, 50]
            
            summary_rows = []

            for dur in durations:
                key = f"NG_{dur}"
                
                # Get arrays for all ensemble members
                hist_le2_arr = hist_le2_data[key]
                futu_le2_arr = futu_le2_data[key]
                hist_le4_arr = hist_le4_data[key]
                futu_le4_arr = futu_le4_data[key]
                hist_le7_arr = hist_le7_data[key]
                futu_le7_arr = futu_le7_data[key]
                hist_le9_arr = hist_le9_data[key]
                futu_le9_arr = futu_le9_data[key]

                for ari, idx in zip(aris, indices):
                    # LE2
                    val_hist_le2 = hist_le2_arr[0, idx]
                    val_futu_le2 = futu_le2_arr[0, idx]
                    diff_le2 = val_futu_le2 - val_hist_le2
                    pct_le2 = (diff_le2 / val_hist_le2 * 100) if val_hist_le2 != 0 else 0.0

                    # LE4
                    val_hist_le4 = hist_le4_arr[0, idx]
                    val_futu_le4 = futu_le4_arr[0, idx]
                    diff_le4 = val_futu_le4 - val_hist_le4
                    pct_le4 = (diff_le4 / val_hist_le4 * 100) if val_hist_le4 != 0 else 0.0

                    # LE7
                    val_hist_le7 = hist_le7_arr[0, idx]
                    val_futu_le7 = futu_le7_arr[0, idx]
                    diff_le7 = val_futu_le7 - val_hist_le7
                    pct_le7 = (diff_le7 / val_hist_le7 * 100) if val_hist_le7 != 0 else 0.0

                    # LE9
                    val_hist_le9 = hist_le9_arr[0, idx]
                    val_futu_le9 = futu_le9_arr[0, idx]
                    diff_le9 = val_futu_le9 - val_hist_le9
                    pct_le9 = (diff_le9 / val_hist_le9 * 100) if val_hist_le9 != 0 else 0.0

                    summary_rows.append({
                        "duration": dur,
                        "ari": ari,
                        "hist_le2": val_hist_le2,
                        "futu_le2": val_futu_le2,
                        "diff_le2": diff_le2,
                        "pct_le2": pct_le2,
                        "hist_le4": val_hist_le4,
                        "futu_le4": val_futu_le4,
                        "diff_le4": diff_le4,
                        "pct_le4": pct_le4,
                        "hist_le7": val_hist_le7,
                        "futu_le7": val_futu_le7,
                        "diff_le7": diff_le7,
                        "pct_le7": pct_le7,
                        "hist_le9": val_hist_le9,
                        "futu_le9": val_futu_le9,
                        "diff_le9": diff_le9,
                        "pct_le9": pct_le9
                    })

            # Use the new near-term template
            return render_template("out_cesm_near.html",
                                   lat=lat, lon=lon,
                                   hist_le2=hist_le2_data,
                                   futu_le2=futu_le2_data,
                                   hist_le4=hist_le4_data,
                                   futu_le4=futu_le4_data,
                                   hist_le7=hist_le7_data,
                                   futu_le7=futu_le7_data,
                                   hist_le9=hist_le9_data,
                                   futu_le9=futu_le9_data,
                                   summary_rows=summary_rows,
                                   fig_prec_comp=fig_prec_comp,
                                   fig_ng_comp=fig_ng_comp,
                                   fig_am_p_comp=fig_am_p_comp,
                                   fig_am_w_comp=fig_am_w_comp,
                                   fig_swe_comp=fig_swe_comp)
        
        else:
            # Unknown scenario - return to main page with error message
            print(f"ERROR: Unknown scenario '{scenario}'", flush=True)
            return render_template("NG_IDF.html")

    return render_template("NG_IDF.html")





# ----------------------------------------------------------------------------------------------------------------------------
# read the values and start processing 
def get_NG_IDF(input_data, forcing_type="Daymet", year_range=None):
    """
    Generate NG-IDF curves for a given location and forcing type.
    
    Parameters:
    -----------
    input_data : dict
        Dictionary with keys 'value1' through 'value10' containing model parameters
    forcing_type : str
        Type of forcing data (e.g., "Daymet", "WRF_historical", "CESM_hist_LE2", etc.)
    year_range : tuple or None
        Optional (start_year, end_year) to filter data. If None, uses full range.
        Example: (2006, 2021) for 16-year historical period
    
    Returns:
    --------
    dict : Dictionary containing durations, idf_data, fig_codes, and am_results
    """

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
        "WRF_high": "./met/WRF_high/",
        "CESM_hist_LE2": "./met/CESM_hist_LE2/",
        "CESM_hist_LE4": "./met/CESM_hist_LE4/",
        "CESM_hist_LE7": "./met/CESM_hist_LE7/",
        "CESM_hist_LE9": "./met/CESM_hist_LE9/",
        "CESM_futu_LE2": "./met/CESM_futu_LE2/",
        "CESM_futu_LE4": "./met/CESM_futu_LE4/",
        "CESM_futu_LE7": "./met/CESM_futu_LE7/",
        "CESM_futu_LE9": "./met/CESM_futu_LE9/"
    }
    met_path = folder_map.get(forcing_type, "./met/Daymet/")




    with tempfile.TemporaryDirectory() as td:
        # --------------------------------------------------------------------------------------------------
        # Create file paths inside the temporary directory
        # These files will be created by DHSVM, extract_AM.py, and R scripts
        # --------------------------------------------------------------------------------------------------
        
        # DHSVM output file (created by DHSVM run)
        pixel_file = os.path.join(td, 'Pixel.CENTER')  # Main DHSVM output with timestep data
        
        # DHSVM auxiliary output files (created by DHSVM)
        tmp_fil1 = os.path.join(td, 'Mass.Final.Balance')
        tmp_fil2 = os.path.join(td, 'Mass.Balance')
        tmp_fil3 = os.path.join(td, 'Stream.Flow')
        tmp_fil4 = os.path.join(td, 'Streamflow.Only')
        tmp_fil5 = os.path.join(td, 'Aggregated.Values')
        
        # --------------------------------------------------------------------------------------------------
        # Determine which durations to process based on forcing type
        # Daymet: 3 durations (24h, 48h, 72h) - 3-hourly timestep
        # WRF/CESM: 7 durations (1h, 3h, 6h, 12h, 24h, 48h, 72h) - 1-hourly timestep
        # --------------------------------------------------------------------------------------------------
        if forcing_type == "Daymet":
            durations_to_process = ['24h', '48h', '72h']
        else:  # WRF or CESM
            durations_to_process = ['1h', '3h', '6h', '12h', '24h', '48h', '72h']
        
        # --------------------------------------------------------------------------------------------------
        # Create file path dictionaries for all durations
        # These files will be created by extract_AM.py and get_IDF.R
        # --------------------------------------------------------------------------------------------------
        
        # Annual Maximum (AM) files - created by extract_AM.py
        # Format: am_<duration>_<variable>
        # Example content: [[2033, 10, 15, 45.3], [2034, 11, 3, 52.1], ...]
        #                  (Year, Month, Day, Value for each water year)
        am_files = {}
        for dur in durations_to_process:
            am_files[f'am_{dur}_W_veg'] = os.path.join(td, f'am_{dur}_W_veg')  # Net groundwater AM
            am_files[f'am_{dur}_P'] = os.path.join(td, f'am_{dur}_P')          # Precipitation AM
        
        # IDF curve files - created by get_IDF.R
        # Format: IDF_<duration>_<variable>
        # Example content: 3 rows x 51 columns
        #   Row 0: Point estimates for 51 probabilities (0.50 to 0.998)
        #   Row 1: 5% confidence interval
        #   Row 2: 95% confidence interval
        idf_files = {}
        for dur in durations_to_process:
            idf_files[f'IDF_{dur}_P'] = os.path.join(td, f'IDF_{dur}_P')              # Precip IDF
            idf_files[f'IDF_{dur}_W_veg'] = os.path.join(td, f'IDF_{dur}_W_veg')      # Net GW IDF
        
        # Figure files - created by gen_figures.py
        # Format: fig_<duration>.png
        # These will be converted to base64 strings for HTML display
        fig_files = {}
        for dur in durations_to_process:
            fig_files[f'fig_{dur}'] = os.path.join(td, f'fig_{dur}.png')
        
        # --------------------------------------------------------------------------------------------------
        # Generate DHSVM configuration file and run DHSVM
        # --------------------------------------------------------------------------------------------------
        config_file = generate_ng_idf(bas_par, adv_par, td, met_path, forcing_type)
        print(f"Config file: {config_file}")
        print(f"Met path: {met_path}")
        print(f"Forcing type: {forcing_type}")
        print(f"Temp directory: {td}")
    
        # copy the config file to the local folder for debug
        debug_config_path = "./debug_config.txt"
        shutil.copy2(config_file, debug_config_path)
        print(f"DEBUG: Config file copied to {debug_config_path} for inspection")


        # Check if config file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not created: {config_file}")
        
        os.system('dos2unix ' + config_file)
        
        # Run DHSVM without streaming per-timestep output to the terminal
        print("Running DHSVM...")
        dhsvm_proc = subprocess.run(
            ['./dhsvm/no_sat_dump/DHSVM3.2', config_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # debug mode to run DHSVM with per-timestep output to the terminal
        # dhsvm_proc = subprocess.run(
        #     ['./dhsvm/no_sat_dump/DHSVM3.2', config_file]
        # )

        dhsvm_result = dhsvm_proc.returncode
        print(f"DHSVM exit code: {dhsvm_result}")
        
        # create r file in the temp folder and copy the lines into the new temp r file
        r_file = os.path.join(td, 'get_IDF.R')
        shutil.copy2('./get_IDF.R', r_file)

        # ----------------------------------------------------
        # pixel_file = './example_output/Pixel.CENTER'
        pixel_file = os.path.join(td, 'Pixel.CENTER')

        # Check if Pixel.CENTER was created
        if not os.path.exists(pixel_file):
            print(f"ERROR: Pixel.CENTER not found at {pixel_file}")
            print("DHSVM may have failed (model stdout/stderr are suppressed during the run).")
            raise FileNotFoundError(f"DHSVM did not create Pixel.CENTER file. Check DHSVM configuration and met data paths.")

        # Preview Pixel.CENTER (first and last 3 lines) for debugging
        with open(pixel_file, 'r', errors='replace') as pf:
            pix_lines = pf.readlines()
        n_pix = len(pix_lines)
        print(f"Pixel.CENTER ({n_pix} lines) — first 20 lines of {pixel_file}:")
        for row in pix_lines[:3]:
            print(row.rstrip('\n\r'))
        print(f"Pixel.CENTER — last 20 lines of {pixel_file}:")
        for row in pix_lines[-3:]:
            print(row.rstrip('\n\r'))




        # Determine durations based on forcing type
        if forcing_type == "Daymet":
            durations = ['24h', '48h', '72h']
        else:  # WRF or CESM
            durations = ['1h', '3h', '6h', '12h', '24h', '48h', '72h']
        
        # Create AM file dictionary
        am_files_dict = {}
        for dur in durations:
            am_files_dict[f'am_{dur}_W_veg'] = os.path.join(td, f'am_{dur}_W_veg')
            am_files_dict[f'am_{dur}_P'] = os.path.join(td, f'am_{dur}_P')
        
        # List files in temp directory
        print(f"Files in temp directory after DHSVM:")
        os.system('ls -lh ' + td)

        # Extract AM data with optional year range filtering
        am_results = extract_AM_data(pixel_file, am_files_dict, r_file, td, forcing_type, year_range)

        # Debug: inspect am_results in docker logs (docker logs -f <container>)
        # print("am_results keys:", list(am_results.keys()), flush=True)
        # for dur_key in sorted(k for k in am_results.keys() if k.endswith('h')):
        #     p = am_results[dur_key]['P']
        #     w = am_results[dur_key]['W_veg']
        #     print(f"  {dur_key}: P shape {p.shape}, W_veg shape {w.shape}", flush=True)
        #     print(f"    P first 2 rows:\n{np.array2string(p[:2], precision=4)}", flush=True)
        #     print(f"    P last 2 rows:\n{np.array2string(p[-2:], precision=4)}", flush=True)
        # if 'swe' in am_results:
        #     s = am_results['swe']
        #     print(f"  swe: shape {s.shape}, first row {s[0]}, last row {s[-1]}", flush=True)





        # load the P-IDF and NG-IDF data from R output
        # 1st row: estimated value; 2nd row: 5% quantile; 3rd row: 95% quantile
        # column: 51 probabilities (0.50 to 0.99 plus 0.998)
        num_pro = 51
        
        # Load IDF data for all durations
        idf_data = {}
        for dur in durations:
            IDF_P_file = os.path.join(td, f'IDF_{dur}_P')
            IDF_W_veg_file = os.path.join(td, f'IDF_{dur}_W_veg')
            
            P_IDF = npnan(3, num_pro)
            NG_IDF = npnan(3, num_pro)
            
            if os.path.exists(IDF_P_file):
                P_IDF = read_idf(IDF_P_file, P_IDF)
            if os.path.exists(IDF_W_veg_file):
                NG_IDF = read_idf(IDF_W_veg_file, NG_IDF)
            
            idf_data[f'P_{dur}'] = np.round(P_IDF, 2)
            idf_data[f'NG_{dur}'] = np.round(NG_IDF, 2)
        
        # Generate figures for all durations
        fig_files_dict = {}
        for dur in durations:
            fig_files_dict[f'fig_{dur}'] = os.path.join(td, f'fig_{dur}.png')
        
        fig_codes = generate_figs_multiple(idf_data, fig_files_dict, durations)
        
        # Return dictionary with all data
        return {
            'durations': durations,
            'idf_data': idf_data,
            'fig_codes': fig_codes,
            'am_results': am_results
        }



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

