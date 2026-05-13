# NG-IDF AWS Project

## Project Overview

This project is a Flask-based web application that generates **Nature-based Green Infrastructure Intensity-Duration-Frequency (NG-IDF)** curves for hydrological analysis. The application uses the DHSVM (Distributed Hydrology Soil Vegetation Model) to simulate hydrological processes and compute IDF curves for precipitation and vegetation-intercepted water across different climate scenarios.

## Code Architecture

### Core Components

#### 1. **app.py** (Main Application Entry Point)
The Flask web application that orchestrates the entire workflow.

**Key Functions:**
- `NG_IDF()`: Main route handler (`/NG_IDF`) that processes GET/POST requests
  - Handles user input (latitude, longitude, vegetation parameters)
  - Branches between historical (Daymet) and future (WRF) climate scenarios
  - Renders results using HTML templates
- `get_NG_IDF(input_data, forcing_type)`: Core processing function that:
  - Calls `generate_ng_idf()` from dhsvm_idf.py to create DHSVM configuration
  - Executes DHSVM simulation
  - Calls `extract_AM_data()` to process simulation output
  - Calls `generate_fig()` to create visualization plots
  - Returns IDF curves and annual maximum data
- `structure_results()`: Helper to organize results for template rendering
- `read_idf()`: Reads IDF curve data from files
- `npnan()`: Creates NaN-filled numpy arrays

**Dependencies:**
- Flask (web framework)
- numpy, pandas (data processing)
- matplotlib (plotting)
- subprocess, os (system operations)
- **dhsvm_idf.py** (configuration generation)
- **extract_AM.py** (data extraction)
- **gen_figures.py** (visualization)

---

#### 2. **dhsvm_idf.py** (DHSVM Configuration Generator)
Generates DHSVM model configuration files based on user inputs and climate forcing data.

**Key Functions:**
- `generate_ng_idf(base_par, adv_par, td, met_path, forcing_type)`: Main orchestrator
  - Determines nearest forcing data coordinates
  - Identifies snow and vegetation parameters
  - Calls appropriate configuration update function
  - Returns path to final configuration file
- `find_lat_lon(input_lat, input_lon, forcing_type)`: Finds nearest grid point in forcing data
- `find_cover(cover_idx)`: Maps cover type index to name
- `find_snow_param(grid_lat, grid_lon)`: Retrieves snow parameters from CSV
- `find_default_advan(cover, cluster)`: Retrieves vegetation parameters from CSV
- `update_config_file_basic()`: Updates config for overstory+understory vegetation
- `update_config_file_under_only()`: Updates config for understory-only vegetation
- `update_config_file_open()`: Updates config for open (no vegetation) areas
- `update_config_file_advanced()`: Applies advanced vegetation parameters
- `load_forcing_coords()`: Loads coordinate list from text files
- `find_nearest_coord()`: Finds nearest coordinate using Euclidean distance

**Called by:** `app.py::get_NG_IDF()`

**Dependencies:**
- numpy, pandas, re, os
- Configuration templates in `example_config/`
- Parameter files: `param_5cluster_ensembleMean_CONUS.csv`, `veg_param_type_cluster.csv`
- Coordinate files: `daymet_coordinates.txt`, `wrf_coordinates.txt`

---

#### 3. **extract_AM.py** (Annual Maximum Data Extraction)
Processes DHSVM output to extract annual maximum (AM) values and prepares data for IDF curve fitting.

**Key Functions:**
- `extract_AM_data(pixel_file, am_24h_W_veg_file, ...)`: Main processing function
  - Reads 3-hourly DHSVM output (`Pixel.CENTER`)
  - Aggregates to 24h, 48h, 72h durations
  - Filters data to complete water years (Oct 1 - Sep 30)
  - Extracts annual maximum values by water year
  - Writes AM data to temporary files
  - Dynamically updates and executes R script (`get_IDF.R`)
  - Returns AM arrays for precipitation, vegetation water, and SWE
- `finddate(year, month, day, var)`: Finds index of specific date in array
- `npnan(x, y)`: Creates NaN-filled numpy arrays

**Called by:** `app.py::get_NG_IDF()`

**Calls:** `get_IDF.R` (via subprocess)

**Dependencies:**
- numpy, pandas, scipy
- subprocess (to execute R script)
- **get_IDF.R** (statistical analysis)

---

#### 4. **gen_figures.py** (Visualization Generator)
Creates IDF curve plots comparing precipitation and nature-based green infrastructure curves.

**Key Functions:**
- `generate_fig(P_IDF_24h, P_IDF_48h, P_IDF_72h, NG_IDF_24h, NG_IDF_48h, NG_IDF_72h, ...)`: Main function
  - Generates three separate plots (24h, 48h, 72h)
  - Creates semi-log plots with return periods (2, 5, 10, 25, 50, 100, 500 years)
  - Converts plots to base64-encoded strings for HTML embedding
  - Returns base64 image codes
- `plot_single()`: Helper to create individual duration plot
- `fig_to_base64()`: Converts matplotlib figure to base64 string
- `read_idf()`: Reads IDF data from files
- `npnan()`: Creates NaN-filled numpy arrays

**Called by:** `app.py::get_NG_IDF()`

**Dependencies:**
- matplotlib, numpy, pandas
- io, base64 (for image encoding)

---

#### 5. **get_IDF.R** (Statistical IDF Curve Fitting)
R script that fits Gumbel distribution to annual maximum data and computes IDF curves with confidence intervals.

**Key Functions:**
- `esti_idf(data)`: Fits Gumbel distribution and computes quantiles for 51 probability levels (0.50 to 0.998)
- `esti_idf90(data)`: Computes 90% confidence intervals using Monte Carlo resampling (1000 iterations)
- Main loop: Processes all combinations of durations (24h, 48h, 72h) and variables (P, W_veg)

**Called by:** `extract_AM.py` (via subprocess)

**Dependencies:**
- R package: `lmom` (L-moments for distribution fitting)

---

### Dependency Flow Diagram

```
User Request (Web Browser)
         ↓
    [app.py]
         ↓
    NG_IDF() route handler
         ↓
    get_NG_IDF()
         ↓
    ┌────────────────────────────────────────┐
    │                                        │
    ↓                                        ↓
[dhsvm_idf.py]                         [DHSVM Binary]
generate_ng_idf()                      (C executable)
    ↓                                        ↓
Creates config file              Generates Pixel.CENTER
    │                                        │
    └────────────────┬───────────────────────┘
                     ↓
              [extract_AM.py]
              extract_AM_data()
                     ↓
              Writes AM files
                     ↓
               [get_IDF.R]
              (R subprocess)
                     ↓
              Writes IDF files
                     ↓
              [gen_figures.py]
              generate_fig()
                     ↓
              Returns results
                     ↓
                 [app.py]
              Renders template
                     ↓
              HTML Response
```

---

### Calling Relationships

#### app.py → dhsvm_idf.py
```python
# app.py line ~200
config_file = generate_ng_idf(bas_par, adv_par, td, met_path, forcing_type)
```
**Purpose:** Generate DHSVM configuration file with user parameters

#### app.py → extract_AM.py
```python
# app.py line ~210
am_24h_P, am_48h_P, am_72h_P, am_24h_W_veg, am_48h_W_veg, am_72h_W_veg, am_swe = \
    extract_AM_data(pixel_file, am_24h_W_veg_file, am_48h_W_veg_file, ...)
```
**Purpose:** Extract annual maximum data from DHSVM output

#### app.py → gen_figures.py
```python
# app.py line ~230
fig24_code, fig48_code, fig72_code = \
    generate_fig(P_IDF_24h, P_IDF_48h, P_IDF_72h, NG_IDF_24h, NG_IDF_48h, NG_IDF_72h, ...)
```
**Purpose:** Generate visualization plots

#### extract_AM.py → get_IDF.R
```python
# extract_AM.py line ~200
cmd = 'Rscript %s' % (r_file,)
p = subprocess.call(cmd, stdout=subprocess.PIPE, shell=True)
```
**Purpose:** Fit statistical distributions and compute IDF curves

---

### Data Flow

1. **User Input** → app.py receives:
   - Latitude, Longitude
   - Vegetation parameters (LAI, height, fractional cover, cover type)
   - Advanced parameters (LAI multipliers, snow interception)
   - Climate scenario (historical/future)

2. **Configuration Generation** → dhsvm_idf.py:
   - Finds nearest forcing data coordinates
   - Retrieves calibrated snow/vegetation parameters
   - Generates DHSVM configuration file in temp directory

3. **Hydrological Simulation** → DHSVM binary:
   - Reads configuration and forcing data
   - Simulates 32+ years of hydrology (1989-2021)
   - Outputs 3-hourly `Pixel.CENTER` file

4. **Data Processing** → extract_AM.py:
   - Aggregates 3-hourly to daily, 48h, 72h
   - Filters to complete water years (WY1990-WY2021)
   - Extracts annual maximum for each water year
   - Writes AM data files

5. **Statistical Analysis** → get_IDF.R:
   - Fits Gumbel distribution using L-moments
   - Computes quantiles for 51 probability levels
   - Generates 90% confidence intervals via Monte Carlo
   - Writes IDF curve files

6. **Visualization** → gen_figures.py:
   - Creates semi-log plots for 24h, 48h, 72h
   - Encodes as base64 for HTML embedding

7. **Response** → app.py:
   - Structures results for template
   - Renders HTML with tables, plots, and data
   - Returns to user's browser

---

### File Structure

```
NG_IDF_AWS/
├── app.py                    # Flask application (main entry)
├── dhsvm_idf.py             # DHSVM configuration generator
├── extract_AM.py            # Annual maximum data extraction
├── gen_figures.py           # Visualization generator
├── get_IDF.R                # Statistical IDF curve fitting
├── Dockerfile               # Container configuration
├── dhsvm/no_sat_dump/       # DHSVM C source code and binary
│   └── DHSVM3.2             # Compiled DHSVM executable
├── example_config/          # Configuration templates and parameters
│   ├── Input.Snotel.T4      # DHSVM config template
│   ├── param_5cluster_ensembleMean_CONUS.csv
│   └── veg_param_type_cluster.csv
├── input/                   # DHSVM input files (DEM, soil, veg, etc.)
├── met/                     # Meteorological forcing data
│   ├── Daymet/              # Historical observations
│   ├── WRF_historical/      # WRF historical baseline
│   ├── WRF_medium/          # WRF future medium scenario
│   └── WRF_high/            # WRF future high scenario
├── static/                  # Web assets and coordinate files
│   ├── daymet_coordinates.txt
│   ├── wrf_coordinates.txt
│   └── *.png                # Logos
└── templates/               # HTML templates
    ├── NG_IDF.html          # Input form
    ├── out.html             # Historical results
    └── out_future.html      # Future scenario comparison
```

---

### Climate Scenarios

The application supports four forcing datasets:

1. **Daymet** (Historical): Observational data for historical analysis
2. **WRF_historical**: WRF model historical baseline (1989-2021)
3. **WRF_medium**: Future medium emissions scenario (2033-2065, +44 year offset)
4. **WRF_high**: Future high emissions scenario (2033-2065, +44 year offset)

Future scenarios enable climate change impact assessment by comparing baseline vs. projected IDF curves.

---

### Key Technologies

- **Backend**: Python 3 (Flask, NumPy, Pandas, Matplotlib)
- **Statistical Analysis**: R (lmom package)
- **Hydrological Model**: DHSVM 3.2 (C, compiled binary)
- **Deployment**: Docker container
- **Frontend**: HTML templates with Jinja2

---

### Execution Flow Summary

1. User submits form → `app.py::NG_IDF()`
2. Generate config → `dhsvm_idf.py::generate_ng_idf()`
3. Run DHSVM → `os.system('./dhsvm/no_sat_dump/DHSVM3.2 ...')`
4. Extract AM data → `extract_AM.py::extract_AM_data()`
5. Fit IDF curves → `get_IDF.R` (subprocess)
6. Generate plots → `gen_figures.py::generate_fig()`
7. Render results → Flask template rendering
8. Return HTML response to user

---

## Installation & Deployment

### Docker Deployment
```bash
docker build -t ng-idf-app .
docker run -p 5000:5000 ng-idf-app
```

### Local Development
```bash
# Install Python dependencies
pip install flask pandas numpy matplotlib scipy

# Install R and lmom package
R -e "install.packages('lmom', dependencies=TRUE)"

# Compile DHSVM
cd dhsvm/no_sat_dump/
make -f makefile_for_binary

# Run application
python app.py
```

Access the application at `http://localhost:5000/NG_IDF`

---

## Authors & Acknowledgments

This project integrates hydrological modeling (DHSVM), statistical analysis (R), and web technologies to provide nature-based infrastructure design tools for climate adaptation.
