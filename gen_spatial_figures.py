import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from scipy.interpolate import griddata
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



def generate_daymet_idf_figure(land_cover, spatial_scenario, duration, ari, fig_file=None):
    """
    Generate the 2x2 Daymet spatial IDF comparison figure (context map, PREC-IDF,
    NG-IDF, and percent difference) for a given land cover / duration / ARI.

    Parameters
    ----------
    land_cover : str
        "open" | "evergreen" | "deciduous"
    spatial_scenario : str
        Scenario code (currently only "daymet" is supported by this function).
    duration : str or int
        Storm duration in hours, e.g. "24"
    ari : str or int
        Average recurrence interval in years, e.g. "25"
    fig_file : str or None
        If provided, the figure is saved to this file path (e.g. a path inside a
        temporary directory). If None, defaults to "./ng_idf_spatial_comparison.png"
        in the current working directory (legacy behavior).

    Returns
    -------
    str
        Base64-encoded PNG string prefixed with "data:image/png;base64," suitable
        for direct use in an HTML <img src="..."> tag.
    """

    idf_path = f"./spatial_map/{land_cover}/Daymet/ng_idf_return_{duration}h_{ari}yr.csv"
    


    # ---- Load data ----
    df = pd.read_csv(idf_path)
    MM_TO_IN = 1.0 / 25.4
    df["prec_idf_hist_in"] = df["prec_idf_hist"] * MM_TO_IN
    df["ng_idf_hist_in"]   = df["ng_idf_hist"]   * MM_TO_IN
    df["diff_pct"] = (df["ng_idf_hist"] - df["prec_idf_hist"]) / df["prec_idf_hist"] * 100.0

    shared_vmin = np.nanpercentile(
        np.concatenate([df["prec_idf_hist_in"].values, df["ng_idf_hist_in"].values]), 2)
    shared_vmax = np.nanpercentile(
        np.concatenate([df["prec_idf_hist_in"].values, df["ng_idf_hist_in"].values]), 98)
    dlim = np.nanpercentile(np.abs(df["diff_pct"].values), 98)
    mag_label = f"{duration}-h {ari}-yr Magnitude (in)"
    panels = [
        ("prec_idf_hist_in", "(b) PREC-IDF", "viridis", shared_vmin, shared_vmax, mag_label),
        ("ng_idf_hist_in",   "(c) NG-IDF",   "viridis", shared_vmin, shared_vmax, mag_label),
        ("diff_pct", "(d) Difference (NG - PREC)", "RdBu_r", -dlim, dlim, "Difference (%)"),
    ]

    data_crs = ccrs.PlateCarree()

    # ---- interpolate onto regular grid ----
    pad = 0.05
    nx, ny = 400, 400
    grid_lon = np.linspace(df["lon"].min()+pad, df["lon"].max()-pad, nx)
    grid_lat = np.linspace(df["lat"].min()+pad, df["lat"].max()-pad, ny)
    GX, GY = np.meshgrid(grid_lon, grid_lat)
    pts = df[["lon", "lat"]].values
    fields = {c: griddata(pts, df[c].values, (GX, GY), method="linear") for c, *_ in panels}

    # ---- basemap ----
    # Keep OSM for the data panels. Panel (a) uses Esri's topographic tiles,
    # which include terrain relief (hillshade) and natural land-cover context.
    tiler = cimgt.OSM()
    context_tiler = cimgt.GoogleTiles(
        desired_tile_form="RGB",
        url=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
        ),
    )
    proj = tiler.crs
    ZOOM = 9
    ALPHA = 0.55          # more transparent than before

    extent = [df["lon"].min(), df["lon"].max(), df["lat"].min(), df["lat"].max()]
    ext_pad = [extent[0]-0.1, extent[1]+0.1, extent[2]-0.1, extent[3]+0.1]

    sites = [
        ("Fort Wainwright", -147.6389, 64.8283),
        ("North Pole",      -147.3494, 64.7511),
        ("Delta Junction",  -145.7336, 64.0378),
    ]

    # Panel (a) uses the locations requested for the study-area context map.
    # Keep the original locations in the IDF data panels below.
    context_sites = [
        ("Fort Wainwright", -147.6389, 64.8283),
        ("Fort Greely",     -145.7350, 63.9710),
    ]
    # Each entry has the training-area center and the end of its short callout
    # leader. Yukon is called out above its area; Tanana and Donnelly below.
    # training_area_labels = [
    #     ("Yukon\nTraining Area", -146.85, 64.78, -146.85, 64.96),
    #     ("Tanana Flats\nTraining Area", -147.10, 64.54, -147.10, 64.34),
    #     ("Donnelly\nTraining Area", -145.75, 63.88, -145.75, 63.68),
    # ]

    training_area_labels = [
        # Yukon: Line is perfectly HORIZONTAL pointing to the RIGHT (dy = 0.0)
        ("Yukon\nTraining\nArea",        -146.35, 64.70,  0.45,  0.00, "left",   "center"),  
        
        # Tanana: Line is perfectly VERTICAL pointing DOWNWARD (dx = 0.0)
        ("Tanana\nTraining\nArea",       -147.7, 64.5,  0.00, -0.25, "center", "top"),  
        
        # Donnelly: Line is perfectly VERTICAL pointing DOWNWARD (dx = 0.0)
        ("Donnelly\nTraining\nArea",     -146.5, 63.85,  0.00, -0.22, "center", "top"),  
    ]


    def add_sites(ax, fontsize=7, star=10, locations=None):
        for name, clon, clat in (sites if locations is None else locations):
            if extent[0] <= clon <= extent[1] and extent[2] <= clat <= extent[3]:
                ax.plot(clon, clat, marker="*", color="red", markersize=star,markeredgecolor="white", markeredgewidth=0.9,
                        transform=data_crs, zorder=6)
                text_dy = -0.09 if name == "North Pole" else 0.05
                ax.text(clon+0.05, clat+text_dy, name, fontsize=fontsize, color="black",
                        weight="bold", transform=data_crs, zorder=7)

    # def add_training_area_labels(ax):
    #     """Add short leader callouts for the online-basemap training areas."""
    #     for name, clon, clat, label_lon, label_lat in training_area_labels:
    #         ax.plot([clon, label_lon], [clat, label_lat], color="black",
    #                 linewidth=0.8, transform=data_crs, zorder=8)
    #         ax.text(label_lon, label_lat, name, fontsize=7, color="black",
    #                 weight="bold", ha="center",
    #                 va="bottom" if label_lat > clat else "top",
    #                 transform=data_crs, zorder=8)
    def add_training_area_labels(ax):
        """Add clean text labels and angled leader lines without markers or background boxes."""
        for name, clon, clat, dx, dy, ha_align, va_align in training_area_labels:
            # Calculate label position
            label_lon = clon + dx
            label_lat = clat + dy
            
            # Draw annotation with callout line and clean text
            ax.annotate(
                name,
                xy=(clon, clat),                      # Map location (target point)
                xytext=(label_lon, label_lat),        # Label text position
                xycoords=data_crs._as_mpl_transform(ax),
                textcoords=data_crs._as_mpl_transform(ax),
                fontsize=7,
                color="black",
                ha=ha_align,                          # Horizontal alignment (left/right)
                va=va_align,                          # Vertical alignment
                zorder=9,
                arrowprops=dict(
                    arrowstyle="-",                   # Clean leader line
                    color="black", 
                    lw=0.8
                )
            )
    def add_grid(ax):
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                        alpha=0.5, linestyle="--", zorder=4)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(-149, -144, 1))
        gl.ylocator = mticker.FixedLocator(np.arange(63, 66, 0.5))
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}

    # ---- 2x2 layout ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6),
                            subplot_kw={"projection": proj},
                            constrained_layout=True)
    axf = axes.flatten()

    # Tighten the spacing between panels (smaller gap between (a) and (b), etc.)
    fig.set_constrained_layout_pads(w_pad=0.0, h_pad=0.02, wspace=0.0, hspace=0.02)


    # ===== Panel (a): context map =====
    ax = axf[0]
    ax.set_extent(ext_pad, crs=data_crs)
    ax.add_image(context_tiler, ZOOM)

    # ---- training-area boundaries (uncomment when shapefile available) ----
    # import cartopy.io.shapereader as shpreader
    # from cartopy.feature import ShapelyFeature
    # shp = shpreader.Reader("training_areas.shp")
    # ax.add_feature(ShapelyFeature(shp.geometries(), data_crs,
    #                facecolor="none", edgecolor="darkorange", linewidth=2.0), zorder=5)

    ax.plot([extent[0], extent[1], extent[1], extent[0], extent[0]],
            [extent[2], extent[2], extent[3], extent[3], extent[2]],
            color="blue", linewidth=1.8, transform=data_crs, zorder=5,
            label="Study domain")

    add_sites(ax, fontsize=7, star=10, locations=context_sites)
    add_training_area_labels(ax)
    add_grid(ax)
    ax.set_title("(a) Study Area: Interior Alaska", fontsize=11)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)

    # ===== Panels (b)(c)(d) =====
    mesh_ref = None
    for ax, (col, title, cmap, vmin, vmax, clabel) in zip(axf[1:], panels):
        ax.set_extent(ext_pad, crs=data_crs)
        ax.add_image(tiler, ZOOM)

        mesh = ax.pcolormesh(GX, GY, fields[col], cmap=cmap,
                            vmin=vmin, vmax=vmax, transform=data_crs,
                            alpha=ALPHA, shading="auto", zorder=2)
        mesh_ref = mesh

        add_sites(ax)
        add_grid(ax)
        ax.set_title(title, fontsize=11)

        cb = fig.colorbar(mesh, ax=ax, orientation="vertical",
                        fraction=0.046, pad=0.01)
        cb.set_label(clabel, fontsize=9)
        cb.ax.tick_params(labelsize=8)
        cb.solids.set_alpha(1.0)

    # ---- invisible colorbar on (a) so it aligns with (c) ----
    cb_a = fig.colorbar(mesh_ref, ax=axf[0], orientation="vertical",
                        fraction=0.046, pad=0.01)
    cb_a.outline.set_visible(False)
    cb_a.ax.set_facecolor("none")
    cb_a.ax.tick_params(size=0, labelsize=0, colors="none")
    for s in cb_a.ax.spines.values():
        s.set_visible(False)
    cb_a.solids.set_alpha(0.0)

    # fig.suptitle(f"{duration}h {ari}-yr Return Level: PREC-IDF vs NG-IDF, Interior Alaska "
    #              f"({land_cover.capitalize()} Land Cover)",
    #             fontsize=16)


    if fig_file is None:
        fig_file = "./ng_idf_spatial_comparison.png"

    plt.savefig(fig_file, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Encode the saved PNG as a base64 data URI so it can be embedded directly
    # in an HTML <img> tag without needing to serve the file separately.
    with open(fig_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return "data:image/png;base64," + encoded


def generate_daymet_alt_figure(land_cover, spatial_scenario, duration, ari, fig_file=None):
    """
    Generate the 1x2 Daymet Active Layer Thickness (ALT) map (study domain context
    map + mean ALT map) for a given land cover.

    Parameters
    ----------
    land_cover : str
        "open" | "evergreen" | "deciduous"
    spatial_scenario : str
        Scenario code (currently only "daymet" is supported by this function).
    duration : str or int
        Storm duration in hours, e.g. "24" (kept for signature consistency with
        generate_daymet_idf_figure; not used to filter ALT data).
    ari : str or int
        Average recurrence interval in years, e.g. "25" (kept for signature
        consistency with generate_daymet_idf_figure; not used to filter ALT data).
    fig_file : str or None
        If provided, the figure is saved to this file path (e.g. a path inside a
        temporary directory). If None, defaults to "./alt_spatial.png" in the
        current working directory (legacy behavior).

    Returns
    -------
    str
        Base64-encoded PNG string prefixed with "data:image/png;base64," suitable
        for direct use in an HTML <img src="..."> tag.
    """

    # ---- Load data ----

    ald_path = f"./spatial_map/{land_cover}/Daymet/ng_idf_ald_results.csv"

    M_TO_FT = 3.28084
    df = pd.read_csv(ald_path)
    df = df.dropna(subset=["lat", "lon", "mean_ALD_hist"])
    df["mean_ALD_hist_ft"] = df["mean_ALD_hist"] * M_TO_FT
    VAR = "mean_ALD_hist_ft"
    vmin = np.nanpercentile(df[VAR].values, 2)
    vmax = np.nanpercentile(df[VAR].values, 98)

    data_crs = ccrs.PlateCarree()

    # ---- interpolate onto regular grid ----
    pad = 0.05
    nx, ny = 400, 400
    grid_lon = np.linspace(df["lon"].min() + pad, df["lon"].max() - pad, nx)
    grid_lat = np.linspace(df["lat"].min() + pad, df["lat"].max() - pad, ny)
    GX, GY = np.meshgrid(grid_lon, grid_lat)
    pts = df[["lon", "lat"]].values
    ALT = griddata(pts, df[VAR].values, (GX, GY), method="linear")

    # ---- basemap ----
    tiler = cimgt.OSM()
    proj = tiler.crs
    ZOOM = 9
    ALPHA = 0.55

    extent = [df["lon"].min(), df["lon"].max(), df["lat"].min(), df["lat"].max()]
    ext_pad = [extent[0] - 0.1, extent[1] + 0.1, extent[2] - 0.1, extent[3] + 0.1]

    sites = [
        ("Fort Wainwright", -147.6389, 64.8283),
        ("North Pole",      -147.3494, 64.7511),
        ("Delta Junction",  -145.7336, 64.0378),
    ]


    def add_sites(ax, fontsize=7, star=10):
        for name, clon, clat in sites:
            if extent[0] <= clon <= extent[1] and extent[2] <= clat <= extent[3]:
                ax.plot(clon, clat, marker="*", color="red", markersize=star,
                        markeredgecolor="white", markeredgewidth=0.9,
                        transform=data_crs, zorder=6)
                # Move the "North Pole" label lower so it doesn't overlap
                # with the nearby "Fort Wainwright" label.
                text_dy = -0.09 if name == "North Pole" else 0.05
                ax.text(clon + 0.05, clat + text_dy, name, fontsize=fontsize,
                        color="black", weight="bold", transform=data_crs, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                ec="none", alpha=0.8))


    def add_grid(ax):
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                        alpha=0.5, linestyle="--", zorder=4)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(-149, -144, 1))
        gl.ylocator = mticker.FixedLocator(np.arange(63, 66, 0.5))
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}


    # ---- 1x2 layout ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6),
                            subplot_kw={"projection": proj},
                            constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.0, h_pad=0.02, wspace=0.0, hspace=0.02)

    # ===== Panel (a): study domain =====
    ax = axes[0]
    ax.set_extent(ext_pad, crs=data_crs)
    ax.add_image(tiler, ZOOM)

    # ---- training-area boundaries (uncomment when shapefile available) ----
    # import cartopy.io.shapereader as shpreader
    # from cartopy.feature import ShapelyFeature
    # shp = shpreader.Reader("training_areas.shp")
    # ax.add_feature(ShapelyFeature(shp.geometries(), data_crs,
    #                facecolor="none", edgecolor="darkorange", linewidth=2.0), zorder=5)

    ax.plot([extent[0], extent[1], extent[1], extent[0], extent[0]],
            [extent[2], extent[2], extent[3], extent[3], extent[2]],
            color="blue", linewidth=1.8, transform=data_crs, zorder=5,
            label="Study domain")

    add_sites(ax, fontsize=8, star=12)
    add_grid(ax)
    ax.set_title("(a) Study Area: Interior Alaska", fontsize=11)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)

    # ===== Panel (b): ALT =====
    ax = axes[1]
    ax.set_extent(ext_pad, crs=data_crs)
    ax.add_image(tiler, ZOOM)

    mesh = ax.pcolormesh(GX, GY, ALT, cmap="YlOrRd", vmin=vmin, vmax=vmax,
                        transform=data_crs, alpha=ALPHA, shading="auto", zorder=2)

    add_sites(ax)
    add_grid(ax)
    ax.set_title("(b) Mean Active Layer Thickness (ALT)", fontsize=11)

    cb = fig.colorbar(mesh, ax=ax, orientation="vertical",
                    fraction=0.046, pad=0.01)
    cb.set_label("ALT (ft)", fontsize=9)
    cb.ax.tick_params(labelsize=8)     
    cb.solids.set_alpha(1.0)

    # ---- invisible colorbar on (a) so panels align ----
    cb_a = fig.colorbar(mesh, ax=axes[0], orientation="vertical",
                        fraction=0.046, pad=0.01)
    cb_a.outline.set_visible(False)
    cb_a.ax.set_facecolor("none")
    cb_a.ax.tick_params(size=0, labelsize=0, colors="none")
    for s in cb_a.ax.spines.values():
        s.set_visible(False)
    cb_a.solids.set_alpha(0.0)

    #fig.suptitle("Active Layer Thickness, Interior Alaska", fontsize=16)

    if fig_file is None:
        fig_file = "./alt_spatial.png"

    plt.savefig(fig_file, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Encode the saved PNG as a base64 data URI so it can be embedded directly
    # in an HTML <img> tag without needing to serve the file separately.
    with open(fig_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return "data:image/png;base64," + encoded




def generate_wrf_idf_figure(land_cover, spatial_scenario, duration, ari,
                             fig_file=None, lulc_file=None):
    """
    Generate the 3x3 WRF IDF figure combining PREC-IDF and NG-IDF, showing
    historical magnitudes and percent change under Medium/High future scenarios.

    Row 1: (a) land cover / study domain | (b) PREC-IDF hist (in)  | (c) NG-IDF hist (in)
    Row 2: [scenario text]               | (d) PREC medium chg (%) | (e) NG medium chg (%)
    Row 3: [scenario text]               | (f) PREC high chg (%)   | (g) NG high chg (%)

    (b)(c) share one magnitude colorbar (inches); (d)-(g) share one change colorbar (%).

    Parameters
    ----------
    land_cover : str
        "open" | "evergreen" | "deciduous"
    spatial_scenario : str
        Scenario code (currently only "wrf" is supported by this function).
    duration : str or int
        Storm duration in hours, e.g. "24"
    ari : str or int
        Average recurrence interval in years, e.g. "25"
    fig_file : str or None
        If provided, the figure is saved to this file path. If None, defaults
        to "./wrf_idf_3x3_{land_cover}_{duration}h_{ari}yr.png".
    lulc_file : str or None
        Optional land cover raster path for panel (a); if None, panel (a)
        only shows the study domain outline and site markers.

    Returns
    -------
    str
        Base64-encoded PNG string prefixed with "data:image/png;base64," suitable
        for direct use in an HTML <img src="..."> tag.
    """

    MM_TO_IN = 1.0 / 25.4


    # ---- Load data: duration selects file, ari selects rows ----
    idf_path = f"./spatial_map/{land_cover}/WRF/ng_idf_return_{duration}h_ALL.csv"

    if not os.path.exists(idf_path):
        raise FileNotFoundError(f"CSV not found: {os.path.abspath(idf_path)}")
    df_all = pd.read_csv(idf_path)

    ari = int(ari)
    avail = sorted(df_all["return_period_yr"].unique())
    if ari not in avail:
        raise ValueError(f"ARI {ari} not in {idf_path}; available: {avail}")
    df = df_all[df_all["return_period_yr"] == ari].copy()
    if df.empty:
        raise ValueError(f"No rows for ARI={ari} in {idf_path}")

    # ---- unit conversion: mm -> inches for magnitude columns ----
    df["prec_idf_hist_in"] = df["prec_idf_hist"] * MM_TO_IN
    df["ng_idf_hist_in"] = df["ng_idf_hist"] * MM_TO_IN

    # ---- derived change fields (ratio is unit-independent, use raw mm) ----
    for pre in ("prec_idf", "ng_idf"):
        h = df[f"{pre}_hist"]
        df[f"{pre}_chg_med"] = (df[f"{pre}_future_medium"] - h) / h * 100.0
        df[f"{pre}_chg_high"] = (df[f"{pre}_future_high"] - h) / h * 100.0

    # ---- shared color limits ----
    mag_vals = np.concatenate([df["prec_idf_hist_in"].values, df["ng_idf_hist_in"].values])
    vmax = np.nanpercentile(mag_vals, 98)

    chg_vals = np.concatenate([df["prec_idf_chg_med"].values, df["ng_idf_chg_med"].values,
                               df["prec_idf_chg_high"].values, df["ng_idf_chg_high"].values])
    dlim = np.nanpercentile(np.abs(chg_vals), 98)

    # (axes index in 3x3 grid, data column, panel label)
    mag_panels = [
        (1, "prec_idf_hist_in", "(b) PREC-IDF Baseline Simulation"),
        (2, "ng_idf_hist_in",   "(c) NG-IDF Baseline Simulation"),
    ]
    chg_panels = [
        (4, "prec_idf_chg_med",  "(d) PREC-IDF Change: Medium"),
        (5, "ng_idf_chg_med",    "(e) NG-IDF Change: Medium"),
        (7, "prec_idf_chg_high", "(f) PREC-IDF Change: High"),
        (8, "ng_idf_chg_high",   "(g) NG-IDF Change: High"),
    ]
    all_cols = [c for _, c, _ in mag_panels + chg_panels]

    data_crs = ccrs.PlateCarree()

    # ---- interpolate onto regular grid ----
    pad = 0.05
    nx, ny = 400, 400
    grid_lon = np.linspace(df["lon"].min() + pad, df["lon"].max() - pad, nx)
    grid_lat = np.linspace(df["lat"].min() + pad, df["lat"].max() - pad, ny)
    GX, GY = np.meshgrid(grid_lon, grid_lat)
    pts = df[["lon", "lat"]].values
    fields = {c: griddata(pts, df[c].values, (GX, GY), method="linear")
              for c in all_cols}

    # ---- basemap ----
    tiler = cimgt.OSM()
    proj = tiler.crs
    ZOOM = 9   # match Daymet
    ALPHA = 0.55

    extent = [df["lon"].min(), df["lon"].max(), df["lat"].min(), df["lat"].max()]
    ext_pad = [extent[0] - 0.1, extent[1] + 0.1, extent[2] - 0.1, extent[3] + 0.1]

    sites = [
        ("Fort Wainwright", -147.6389, 64.8283),
        ("North Pole",      -147.3494, 64.7511),
        ("Delta Junction",  -145.7336, 64.0378),
    ]

    def add_sites(ax, fontsize=5, star=10):
        for name, clon, clat in sites:
            if extent[0] <= clon <= extent[1] and extent[2] <= clat <= extent[3]:
                ax.plot(clon, clat, marker="*", color="red", markersize=star,
                        markeredgecolor="white", markeredgewidth=0.9,
                        transform=data_crs, zorder=6)
                # Move the "North Pole" label lower so it doesn't overlap
                # with the nearby "Fort Wainwright" label.
                text_dy = -0.09 if name == "North Pole" else 0.05
                ax.text(clon + 0.05, clat + text_dy, name, fontsize=fontsize,
                        color="black", weight="bold", transform=data_crs, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                  ec="none", alpha=0.8))

    def add_grid(ax):
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                          alpha=0.5, linestyle="--", zorder=4)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(-149, -144, 1))
        gl.ylocator = mticker.FixedLocator(np.arange(63, 66, 0.5))
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}

    # ---- 3x3 layout ----
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.0),
                             subplot_kw={"projection": proj},
                             layout="compressed")
    axf = axes.flatten()
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)

    # blank out (3,3,4) and (3,3,7) — used for scenario text instead
    for i in (3, 6):
        axf[i].set_visible(False)

    # ===== Panel (a): land cover =====
    ax = axf[0]
    ax.set_extent(ext_pad, crs=data_crs)
    ax.add_image(tiler, ZOOM)

    lulc_mesh = None
    if lulc_file is not None:
        import rioxarray as rxr
        from matplotlib.colors import ListedColormap, BoundaryNorm
        lc = rxr.open_rasterio(lulc_file, masked=True).squeeze()
        lc = lc.rio.reproject("EPSG:4326")
        classes = [1, 2, 3]                       # edit to your class codes
        names = ["Open", "Evergreen", "Deciduous"]
        colors = ["#D9C89E", "#1B4D3E", "#7FBF3F"]
        cmap_lc = ListedColormap(colors)
        norm_lc = BoundaryNorm(np.array(classes + [classes[-1] + 1]) - 0.5, cmap_lc.N)
        lulc_mesh = ax.pcolormesh(lc.x.values, lc.y.values, lc.values,
                                  cmap=cmap_lc, norm=norm_lc, transform=data_crs,
                                  alpha=0.75, shading="auto", zorder=2)
        handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=n)
                   for c, n in zip(colors, names)]
        ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.85)

    ax.plot([extent[0], extent[1], extent[1], extent[0], extent[0]],
            [extent[2], extent[2], extent[3], extent[3], extent[2]],
            color="blue", linewidth=1.8, transform=data_crs, zorder=5,
            label="Study domain")

    add_sites(ax, fontsize=5, star=12)
    add_grid(ax)
    ax.set_title("(a) Land Cover / Study Domain", fontsize=9)
    if lulc_mesh is None:
        ax.legend(loc="lower left", fontsize=9, framealpha=0.85)

    # ===== magnitude / change panels =====
    def draw(ax, col, title, cmap, vmin_, vmax_):
        ax.set_extent(ext_pad, crs=data_crs)
        ax.add_image(tiler, ZOOM)
        mesh = ax.pcolormesh(GX, GY, fields[col], cmap=cmap,
                             vmin=vmin_, vmax=vmax_, transform=data_crs,
                             alpha=ALPHA, shading="auto", zorder=3)
        add_sites(ax)
        add_grid(ax)
        ax.set_title(title, fontsize=9)
        return mesh

    mag_mesh = None
    for idx, col, title in mag_panels:
        mag_mesh = draw(axf[idx], col, title, "viridis", 0.0, vmax)

    chg_mesh = None
    for idx, col, title in chg_panels:
        chg_mesh = draw(axf[idx], col, title, "RdBu_r", -dlim, dlim)

    # ===== shared colorbars =====
    cb1 = fig.colorbar(mag_mesh, ax=[axf[1], axf[2]], orientation="vertical",
                       fraction=0.046, pad=0.01, aspect=20)
    cb1.set_label(f"{duration}h {ari}-yr Magnitude (in)", fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    cb1.solids.set_alpha(1.0)

    # invisible colorbar on (a) so column 0 matches columns 1-2 in width
    cb_a = fig.colorbar(mag_mesh, ax=axf[0], orientation="vertical",
                        fraction=0.046, pad=0.01, aspect=20)
    cb_a.outline.set_visible(False)
    cb_a.ax.set_facecolor("none")
    cb_a.ax.tick_params(size=0, labelsize=0, colors="none")
    for s in cb_a.ax.spines.values():
        s.set_visible(False)
    cb_a.solids.set_alpha(0.0)

    cb2 = fig.colorbar(chg_mesh, ax=[axf[4], axf[5], axf[7], axf[8]],
                       orientation="vertical", fraction=0.046, pad=0.01, aspect=40)
    cb2.set_label("Change (%)", fontsize=9)
    cb2.ax.tick_params(labelsize=8)
    cb2.solids.set_alpha(1.0)

    # ===== scenario explainer text in the blank left cells (rows 2-3) =====
    def cell_center(ax):
        bbox = ax.get_position()
        return bbox.x0 + bbox.width / 2, bbox.y0 + bbox.height / 2

    xm, ym = cell_center(axf[3])
    fig.text(xm, ym, "Medium Scenario (2033-2065):\nModerately Hotter and Drier",
             ha="center", va="center", fontsize=8, weight="bold",
             color="black", wrap=True)

    xh, yh = cell_center(axf[6])
    fig.text(xh, yh, "High Scenario (2033-2065):\nSeverely Hotter and Drier",
             ha="center", va="center", fontsize=8, weight="bold",
             color="black", wrap=True)

    # ---- save ----
    if fig_file is None:
        fig_file = f"./wrf_idf_3x3_{land_cover}_{duration}h_{ari}yr.png"
    fig_file = os.path.abspath(fig_file)
    d = os.path.dirname(fig_file)
    if d:
        os.makedirs(d, exist_ok=True)

    plt.savefig(fig_file, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved:", fig_file, os.path.exists(fig_file))

    with open(fig_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return "data:image/png;base64," + encoded




def generate_wrf_alt_figure(land_cover, spatial_scenario, duration, ari,
                            fig_file=None, lulc_file=None):
    """
    2x2 WRF Active Layer Thickness (ALT) figure.

    Row 1: (a) land cover / study domain | (b) historical mean ALT (ft)
    Row 2: (c) ALT change: Medium (ft)   | (d) ALT change: High (ft)

    Basemap setup is identical to generate_wrf_idf_figure: same tiler, same
    ZOOM, same ALPHA, same ext_pad construction, same add_image call order.
    """

    M_TO_FT = 3.28084

    # ---- Load data ----
    alt_path = f"./spatial_map/{land_cover}/WRF/ng_idf_ald_results.csv"

    if not os.path.exists(alt_path):
        raise FileNotFoundError(f"CSV not found: {os.path.abspath(alt_path)}")
    df = pd.read_csv(alt_path)
    df = df.dropna(subset=["lat", "lon", "mean_ALD_hist"])

    # ALD == 0 means no thaw / masked cell; exclude so it does not bias scales.
    # NOTE: applied only to the ALD columns, never to lat/lon, so the domain
    # extent stays identical to the IDF figure.
    for c in ("mean_ALD_hist", "mean_ALD_future_medium", "mean_ALD_future_high"):
        df.loc[df[c] <= 0, c] = np.nan

    # ---- unit conversion: m -> ft, and absolute change (ft) ----
    df["alt_hist_ft"] = df["mean_ALD_hist"] * M_TO_FT
    df["alt_chg_med"] = (df["mean_ALD_future_medium"] - df["mean_ALD_hist"]) * M_TO_FT
    df["alt_chg_high"] = (df["mean_ALD_future_high"] - df["mean_ALD_hist"]) * M_TO_FT

    # ---- shared color limits ----
    vmin_mag = np.nanpercentile(df["alt_hist_ft"].values, 2)
    vmax_mag = np.nanpercentile(df["alt_hist_ft"].values, 98)

    chg_vals = np.concatenate([df["alt_chg_med"].values, df["alt_chg_high"].values])
    chg_vals = chg_vals[np.isfinite(chg_vals)]
    if np.nanmin(chg_vals) < 0:
        dlim = np.nanpercentile(np.abs(chg_vals), 98)
        chg_cmap, chg_vmin, chg_vmax = "RdBu_r", -dlim, dlim
    else:
        chg_cmap, chg_vmin = "YlOrRd", 0.0
        chg_vmax = np.nanpercentile(chg_vals, 98)

    mag_panels = [
        (1, "alt_hist_ft", "(b) Baseline Simulated Mean ALT", None),
    ]
    chg_panels = [
        (2, "alt_chg_med",  "(c) ALT Change: Medium", "medium"),
        (3, "alt_chg_high", "(d) ALT Change: High",   "high"),
    ]
    all_cols = [c for _, c, _, _ in mag_panels + chg_panels]

    scen_text = {
        "medium": "Medium Scenario (2033-2065):\nModerately Hotter and Drier",
        "high":   "High Scenario (2033-2065):\nSeverely Hotter and Drier",
    }

    data_crs = ccrs.PlateCarree()

    # ---- interpolate onto regular grid ----
    pad = 0.05
    nx, ny = 400, 400
    grid_lon = np.linspace(df["lon"].min() + pad, df["lon"].max() - pad, nx)
    grid_lat = np.linspace(df["lat"].min() + pad, df["lat"].max() - pad, ny)
    GX, GY = np.meshgrid(grid_lon, grid_lat)
    fields = {}
    for c in all_cols:
        ok = np.isfinite(df[c].values)
        fields[c] = griddata(df.loc[ok, ["lon", "lat"]].values,
                             df.loc[ok, c].values, (GX, GY), method="linear")

    # ================= BASEMAP: identical to generate_wrf_idf_figure =========
    tiler = cimgt.OSM()
    proj = tiler.crs
    ZOOM = 9   # match Daymet
    ALPHA = 0.55

    extent = [df["lon"].min(), df["lon"].max(), df["lat"].min(), df["lat"].max()]
    ext_pad = [extent[0] - 0.1, extent[1] + 0.1, extent[2] - 0.1, extent[3] + 0.1]

    # Diagnostic: compare this against the IDF figure's extent. If the two
    # differ noticeably, the ALD CSV covers a different domain and that -- not
    # the plotting code -- is why the tiles come back empty.
    print("ALT extent :", [round(v, 4) for v in extent])
    _dlon = ext_pad[1] - ext_pad[0]
    _dlat = ext_pad[3] - ext_pad[2]
    _ntiles = (_dlon / 360.0 * 2 ** ZOOM) * (_dlat / 180.0 * 2 ** ZOOM) * 2
    print(f"ALT domain : {_dlon:.2f} lon x {_dlat:.2f} lat, "
          f"~{_ntiles:.0f} tiles at ZOOM={ZOOM}")
    if _ntiles > 400:
        print("WARNING: tile request is very large; OSM will likely refuse it. "
              "Lower ZOOM or check the ALD CSV domain.")
    # ========================================================================

    sites = [
        ("Fort Wainwright", -147.6389, 64.8283),
        ("North Pole",      -147.3494, 64.7511),
        ("Delta Junction",  -145.7336, 64.0378),
    ]

    def add_sites(ax, fontsize=5, star=10):
        for name, clon, clat in sites:
            if extent[0] <= clon <= extent[1] and extent[2] <= clat <= extent[3]:
                ax.plot(clon, clat, marker="*", color="red", markersize=star,
                        markeredgecolor="white", markeredgewidth=0.9,
                        transform=data_crs, zorder=6)
                text_dy = -0.09 if name == "North Pole" else 0.05
                ax.text(clon + 0.05, clat + text_dy, name, fontsize=fontsize,
                        color="black", weight="bold", transform=data_crs, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                  ec="none", alpha=0.8))

    def add_grid(ax):
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                          alpha=0.5, linestyle="--", zorder=4)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(-149, -144, 1))
        gl.ylocator = mticker.FixedLocator(np.arange(63, 66, 0.5))
        gl.xlabel_style = {"size": 6}
        gl.ylabel_style = {"size": 6}

    def add_scenario_text(ax, key):
        ax.text(0.02, 0.02, scen_text[key], transform=ax.transAxes,
                ha="left", va="bottom", fontsize=6, weight="bold",
                color="black", zorder=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="0.4", lw=0.6, alpha=0.85))

    # ---- 2x2 layout (smaller canvas) ----
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2),
                             subplot_kw={"projection": proj},
                             layout="compressed")
    axf = axes.flatten()
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)

    # ===== Panel (a): land cover =====
    ax = axf[0]
    ax.set_extent(ext_pad, crs=data_crs)
    ax.add_image(tiler, ZOOM)

    lulc_mesh = None
    if lulc_file is not None:
        import rioxarray as rxr
        from matplotlib.colors import ListedColormap, BoundaryNorm
        lc = rxr.open_rasterio(lulc_file, masked=True).squeeze()
        lc = lc.rio.reproject("EPSG:4326")
        classes = [1, 2, 3]
        names = ["Open", "Evergreen", "Deciduous"]
        colors = ["#D9C89E", "#1B4D3E", "#7FBF3F"]
        cmap_lc = ListedColormap(colors)
        norm_lc = BoundaryNorm(np.array(classes + [classes[-1] + 1]) - 0.5, cmap_lc.N)
        lulc_mesh = ax.pcolormesh(lc.x.values, lc.y.values, lc.values,
                                  cmap=cmap_lc, norm=norm_lc, transform=data_crs,
                                  alpha=0.75, shading="auto", zorder=2)
        handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=n)
                   for c, n in zip(colors, names)]
        ax.legend(handles=handles, loc="lower left", fontsize=6, framealpha=0.85)

    ax.plot([extent[0], extent[1], extent[1], extent[0], extent[0]],
            [extent[2], extent[2], extent[3], extent[3], extent[2]],
            color="blue", linewidth=1.8, transform=data_crs, zorder=5,
            label="Study domain")

    add_sites(ax, fontsize=5, star=10)
    add_grid(ax)
    ax.set_title("(a) Land Cover / Study Domain", fontsize=7)
    if lulc_mesh is None:
        ax.legend(loc="lower left", fontsize=6, framealpha=0.85)

    # ===== magnitude / change panels (same draw() as the IDF figure) =====
    def draw(ax, col, title, cmap, vmin_, vmax_):
        ax.set_extent(ext_pad, crs=data_crs)
        ax.add_image(tiler, ZOOM)
        mesh = ax.pcolormesh(GX, GY, fields[col], cmap=cmap,
                             vmin=vmin_, vmax=vmax_, transform=data_crs,
                             alpha=ALPHA, shading="auto", zorder=3)
        add_sites(ax)
        add_grid(ax)
        ax.set_title(title, fontsize=7)
        return mesh

    mag_mesh = None
    for idx, col, title, _ in mag_panels:
        mag_mesh = draw(axf[idx], col, title, "YlGnBu", vmin_mag, vmax_mag)

    chg_mesh = None
    for idx, col, title, scen in chg_panels:
        chg_mesh = draw(axf[idx], col, title, chg_cmap, chg_vmin, chg_vmax)
        add_scenario_text(axf[idx], scen)

    # ===== colorbars =====
    cb1 = fig.colorbar(mag_mesh, ax=axf[1], orientation="vertical",
                       fraction=0.046, pad=0.01, aspect=20)
    cb1.set_label("Historical Mean ALT (ft)", fontsize=7)
    cb1.ax.tick_params(labelsize=6)
    cb1.solids.set_alpha(1.0)

    cb2 = fig.colorbar(chg_mesh, ax=[axf[2], axf[3]], orientation="vertical",
                       fraction=0.046, pad=0.01, aspect=20)
    cb2.set_label("ALT Change (ft)", fontsize=7)
    cb2.ax.tick_params(labelsize=6)
    cb2.solids.set_alpha(1.0)

    # invisible colorbar on (a) so column 0 matches column 1 in width
    cb_a = fig.colorbar(mag_mesh, ax=axf[0], orientation="vertical",
                        fraction=0.046, pad=0.01, aspect=20)
    cb_a.outline.set_visible(False)
    cb_a.ax.set_facecolor("none")
    cb_a.ax.tick_params(size=0, labelsize=0, colors="none")
    for s in cb_a.ax.spines.values():
        s.set_visible(False)
    cb_a.solids.set_alpha(0.0)

    # ---- save ----
    if fig_file is None:
        fig_file = f"./wrf_alt_2x2_{land_cover}.png"
    fig_file = os.path.abspath(fig_file)
    d = os.path.dirname(fig_file)
    if d:
        os.makedirs(d, exist_ok=True)

    plt.savefig(fig_file, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved:", fig_file, os.path.exists(fig_file))

    with open(fig_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return "data:image/png;base64," + encoded



def generate_cesm_mid_century_idf_figure(land_cover, spatial_scenario, duration, ari,
                                         fig_file=None, lulc_file=None,
                                         members=("LE2", "LE4", "LE7", "LE9"),
                                         vmax_fixed=None, dlim_fixed=None,
                                         member_fig_file=None,
                                         dpi=150, save_pdf=False):
    """
    Generate the 2x3 CESM mid-century ensemble-mean IDF figure.

    Row 1: (a) land cover / study domain | (b) PREC-IDF hist  | (c) NG-IDF hist
    Row 2: [ensemble text]               | (d) PREC-IDF chg % | (e) NG-IDF chg %

    (b)(c) share one magnitude colorbar (inches); (d)(e) share one change
    colorbar (%). Members are inner-merged on (lat, lon) so every member
    contributes the same cell population. The ensemble change is derived from
    the ensemble-mean hist and ensemble-mean future -- NOT the average of the
    per-member changes (those differ whenever members have different baselines).

    Signature matches generate_wrf_idf_figure / generate_daymet_idf_figure:
    the data path is built from land_cover, duration, and ari.

        ./spatial_map/{land_cover}/CESM_{member}_Mid_Century/
            ng_idf_return_{duration}h_ALL.csv

    Parameters
    ----------
    land_cover : str
        "open" | "evergreen" | "deciduous"
    spatial_scenario : str
        Scenario code (currently only "cesm_mid_century" is supported here;
        kept for signature consistency with the other figure functions).
    duration : str or int
        Storm duration in hours, e.g. "24"
    ari : str or int
        Average recurrence interval in years, e.g. "25"
    fig_file : str or None
        Output path for the ensemble-mean figure. If None, defaults to
        "./cesm_idf_ensmean_{land_cover}_{duration}h_{ari}yr.png".
    lulc_file : str or None
        Optional land cover raster for panel (a).
    members : sequence of str
        CESM ensemble member codes.
    vmax_fixed : float or None
        Upper limit of the magnitude colorbar, in inches. None autoscales to the
        98th percentile. IDF magnitude scales strongly with duration, so a value
        pinned for 24h will not suit 1h.
    dlim_fixed : float or None
        Symmetric limit of the change colorbar, in percent. None autoscales to
        the 98th percentile of |change|. Percent change is dimensionless, so one
        value can safely be shared across durations and ARIs.
    member_fig_file : str or None
        If given, also write the companion 2xN individual-member change figure
        to this path, forced onto the same color scale. The return value is
        always the ensemble-mean figure.
    dpi : int
        Raster resolution; 150 matches the WRF/Daymet figures, 300 for print.
    save_pdf : bool
        Also write a vector PDF next to each PNG.

    Returns
    -------
    tuple(str, str)
        Two base64-encoded PNG strings, each prefixed with
        "data:image/png;base64,": (1) the 2x3 ensemble-mean figure, and
        (2) the companion 2xN figure showing each individual ensemble
        member's own-baseline percent change (PREC-IDF row / NG-IDF row).
    """

    MM_TO_IN = 1.0 / 25.4
    LATLON_DEC = 5          # rounding used to build the merge key
    EPS = 1e-6              # guard against divide-by-zero on the baseline


    members = list(members)
    ari = int(ari)
    base_dir = f"./spatial_map/{land_cover}/"

    # ---- Load data: one CSV per member, duration selects file, ari selects rows ----
    frames = []
    for m in members:
        path = os.path.join(base_dir, "CESM_%s_Mid_Century" % m,
                            "ng_idf_return_%sh_ALL.csv" % duration)
        if not os.path.exists(path):
            raise FileNotFoundError("CSV not found: %s" % os.path.abspath(path))

        d = pd.read_csv(path)

        need = ["lat", "lon", "return_period_yr",
                "prec_idf_hist", "prec_idf_future",
                "ng_idf_hist", "ng_idf_future"]
        missing = [c for c in need if c not in d.columns]
        if missing:
            raise ValueError("%s is missing columns %s; found: %s"
                             % (path, missing, list(d.columns)))

        avail = sorted(d["return_period_yr"].unique())
        if ari not in avail:
            raise ValueError("ARI %d not in %s; available: %s" % (ari, path, avail))

        d = d[d["return_period_yr"] == ari].copy()
        if d.empty:
            raise ValueError("No rows for ARI=%d in %s" % (ari, path))

        d["lat"] = d["lat"].round(LATLON_DEC)
        d["lon"] = d["lon"].round(LATLON_DEC)
        d = d.drop_duplicates(subset=["lat", "lon"])

        d = d.rename(columns={
            "prec_idf_hist":   "prec_hist_%s" % m,
            "prec_idf_future": "prec_fut_%s" % m,
            "ng_idf_hist":     "ng_hist_%s" % m,
            "ng_idf_future":   "ng_fut_%s" % m,
        })
        frames.append(d[["lat", "lon",
                         "prec_hist_%s" % m, "prec_fut_%s" % m,
                         "ng_hist_%s" % m, "ng_fut_%s" % m]])

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on=["lat", "lon"], how="inner")
    if df.empty:
        raise ValueError("Inner merge across members produced zero cells. "
                         "Check that all members share the same grid.")

    # ---- per-member change, each against ITS OWN hist ----
    prec_chg_cols, ng_chg_cols = [], []
    for m in members:
        base = df["prec_hist_%s" % m].where(df["prec_hist_%s" % m].abs() > EPS)
        df["prec_chg_%s" % m] = (df["prec_fut_%s" % m] - base) / base * 100.0
        base = df["ng_hist_%s" % m].where(df["ng_hist_%s" % m].abs() > EPS)
        df["ng_chg_%s" % m] = (df["ng_fut_%s" % m] - base) / base * 100.0
        prec_chg_cols.append("prec_chg_%s" % m)
        ng_chg_cols.append("ng_chg_%s" % m)

    # ---- ensemble mean of the fields (raw mm), then the ensemble change ----
    prec_hist_mm = df[["prec_hist_%s" % m for m in members]].mean(axis=1)
    prec_fut_mm = df[["prec_fut_%s" % m for m in members]].mean(axis=1)
    ng_hist_mm = df[["ng_hist_%s" % m for m in members]].mean(axis=1)
    ng_fut_mm = df[["ng_fut_%s" % m for m in members]].mean(axis=1)

    prec_base = prec_hist_mm.where(prec_hist_mm.abs() > EPS)
    ng_base = ng_hist_mm.where(ng_hist_mm.abs() > EPS)
    df["prec_chg"] = (prec_fut_mm - prec_base) / prec_base * 100.0
    df["ng_chg"] = (ng_fut_mm - ng_base) / ng_base * 100.0

    # magnitudes for display, in inches
    df["prec_hist_mean_in"] = prec_hist_mm * MM_TO_IN
    df["ng_hist_mean_in"] = ng_hist_mm * MM_TO_IN

    print("ensemble cells after inner merge: %d" % len(df))
    print("domain-mean change (ensemble-mean based): PREC-IDF %+.2f %%, "
          "NG-IDF %+.2f %%" % (np.nanmean(df["prec_chg"].values),
                               np.nanmean(df["ng_chg"].values)))
    for m in members:
        print("  member %s own-baseline change: PREC-IDF %+.2f %%, NG-IDF %+.2f %%"
              % (m, np.nanmean(df["prec_chg_%s" % m].values),
                 np.nanmean(df["ng_chg_%s" % m].values)))

    # ---- shared color limits ----
    mag_vals = np.concatenate([df["prec_hist_mean_in"].values,
                               df["ng_hist_mean_in"].values])
    vmax = float(vmax_fixed) if vmax_fixed is not None \
        else float(np.nanpercentile(mag_vals, 98))

    chg_vals = np.concatenate([df["prec_chg"].values, df["ng_chg"].values])
    dlim = float(dlim_fixed) if dlim_fixed is not None \
        else float(np.nanpercentile(np.abs(chg_vals), 98))

    data_crs = ccrs.PlateCarree()

    # ---- interpolate onto regular grid ----
    pad = 0.05
    nx, ny = 400, 400
    grid_lon = np.linspace(df["lon"].min() + pad, df["lon"].max() - pad, nx)
    grid_lat = np.linspace(df["lat"].min() + pad, df["lat"].max() - pad, ny)
    GX, GY = np.meshgrid(grid_lon, grid_lat)
    pts = df[["lon", "lat"]].values

    map_cols = ["prec_hist_mean_in", "ng_hist_mean_in", "prec_chg", "ng_chg"]
    fields = {c: griddata(pts, df[c].values, (GX, GY), method="linear")
              for c in map_cols}

    # ---- basemap (same tiler / ZOOM / ALPHA as the WRF and Daymet figures) ----
    tiler = cimgt.OSM()
    proj = tiler.crs
    ZOOM = 9
    ALPHA = 0.55

    extent = [df["lon"].min(), df["lon"].max(), df["lat"].min(), df["lat"].max()]
    ext_pad = [extent[0] - 0.1, extent[1] + 0.1, extent[2] - 0.1, extent[3] + 0.1]

    sites = [
        ("Fort Wainwright", -147.6389, 64.8283),
        ("North Pole",      -147.3494, 64.7511),
        ("Delta Junction",  -145.7336, 64.0378),
    ]

    def add_sites(ax, fontsize=5, star=10):
        for name, clon, clat in sites:
            if extent[0] <= clon <= extent[1] and extent[2] <= clat <= extent[3]:
                ax.plot(clon, clat, marker="*", color="red", markersize=star,
                        markeredgecolor="white", markeredgewidth=0.9,
                        transform=data_crs, zorder=6)
                # Move the "North Pole" label lower so it does not overlap
                # with the nearby "Fort Wainwright" label.
                text_dy = -0.09 if name == "North Pole" else 0.05
                ax.text(clon + 0.05, clat + text_dy, name, fontsize=fontsize,
                        color="black", weight="bold", transform=data_crs, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                  ec="none", alpha=0.8))

    def add_grid(ax, labelsize=8):
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                          alpha=0.5, linestyle="--", zorder=4)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(-149, -144, 1))
        gl.ylocator = mticker.FixedLocator(np.arange(63, 66, 0.5))
        gl.xlabel_style = {"size": labelsize}
        gl.ylabel_style = {"size": labelsize}

    def add_lulc(ax, legend_fontsize=8):
        """Panel (a): basemap + optional land cover raster + domain outline."""
        ax.set_extent(ext_pad, crs=data_crs)
        ax.add_image(tiler, ZOOM)

        lulc_mesh = None
        if lulc_file is not None:
            import rioxarray as rxr
            from matplotlib.colors import ListedColormap, BoundaryNorm
            lc = rxr.open_rasterio(lulc_file, masked=True).squeeze()
            lc = lc.rio.reproject("EPSG:4326")
            classes = [1, 2, 3]
            names = ["Open", "Evergreen", "Deciduous"]
            colors = ["#D9C89E", "#1B4D3E", "#7FBF3F"]
            cmap_lc = ListedColormap(colors)
            norm_lc = BoundaryNorm(np.array(classes + [classes[-1] + 1]) - 0.5,
                                   cmap_lc.N)
            lulc_mesh = ax.pcolormesh(lc.x.values, lc.y.values, lc.values,
                                      cmap=cmap_lc, norm=norm_lc,
                                      transform=data_crs, alpha=0.75,
                                      shading="auto", zorder=2)
            handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=n)
                       for c, n in zip(colors, names)]
            ax.legend(handles=handles, loc="lower left",
                      fontsize=legend_fontsize, framealpha=0.85)

        ax.plot([extent[0], extent[1], extent[1], extent[0], extent[0]],
                [extent[2], extent[2], extent[3], extent[3], extent[2]],
                color="blue", linewidth=1.8, transform=data_crs, zorder=5,
                label="Study domain")
        add_sites(ax, fontsize=5, star=12)
        add_grid(ax)
        ax.set_title("(a) Land Cover / Study Domain", fontsize=9)
        if lulc_mesh is None:
            ax.legend(loc="lower left", fontsize=legend_fontsize, framealpha=0.85)

    def blank_colorbar(fig, mesh, ax):
        """Invisible colorbar so a text/context column matches the map columns."""
        cbx = fig.colorbar(mesh, ax=ax, orientation="vertical",
                           fraction=0.046, pad=0.01, aspect=20)
        cbx.outline.set_visible(False)
        cbx.ax.set_facecolor("none")
        cbx.ax.tick_params(size=0, labelsize=0, colors="none")
        for s in cbx.ax.spines.values():
            s.set_visible(False)
        cbx.solids.set_alpha(0.0)
        return cbx

    def save_and_encode(fig, path, default_name):
        if path is None:
            path = default_name
        path = os.path.abspath(path)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        if save_pdf:
            fig.savefig(os.path.splitext(path)[0] + ".pdf",
                        bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print("saved:", path, os.path.exists(path))
        with open(path, "rb") as fh:
            return "data:image/png;base64," + base64.b64encode(fh.read()).decode("utf-8")

    # ================= 2x3 ensemble-mean figure =================
    fig = plt.figure(figsize=(13.5, 8.4), layout="compressed")
    gs = fig.add_gridspec(2, 3)
    ax_lulc = fig.add_subplot(gs[0, 0], projection=proj)
    ax_ph = fig.add_subplot(gs[0, 1], projection=proj)
    ax_nh = fig.add_subplot(gs[0, 2], projection=proj)
    ax_txt = fig.add_subplot(gs[1, 0])
    ax_pc = fig.add_subplot(gs[1, 1], projection=proj)
    ax_nc = fig.add_subplot(gs[1, 2], projection=proj)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)

    add_lulc(ax_lulc)

    def draw(ax, col, title, cmap, vmin_, vmax_):
        ax.set_extent(ext_pad, crs=data_crs)
        ax.add_image(tiler, ZOOM)
        mesh = ax.pcolormesh(GX, GY, fields[col], cmap=cmap,
                             vmin=vmin_, vmax=vmax_, transform=data_crs,
                             alpha=ALPHA, shading="auto", zorder=3,
                             rasterized=True)
        add_sites(ax)
        add_grid(ax)
        ax.set_title(title, fontsize=9)
        return mesh

    mag_mesh = draw(ax_ph, "prec_hist_mean_in",
                    "(b) PREC-IDF Baseline (ensemble mean)", "viridis", 0.0, vmax)
    draw(ax_nh, "ng_hist_mean_in",
         "(c) NG-IDF Baseline (ensemble mean)", "viridis", 0.0, vmax)

    chg_mesh = draw(ax_pc, "prec_chg",
                    "(d) PREC-IDF Change (ensemble mean)", "RdBu_r", -dlim, dlim)
    draw(ax_nc, "ng_chg",
         "(e) NG-IDF Change (ensemble mean)", "RdBu_r", -dlim, dlim)

    # ===== blank cell: ensemble description =====
    ax_txt.set_axis_off()
    ax_txt.text(0.5, 0.66, "Mid-Century Ensemble Mean",
                ha="center", va="center", fontsize=9, weight="bold",
                transform=ax_txt.transAxes)
    ax_txt.text(0.5, 0.52, "Average of %d model simulations" % len(members),
                ha="center", va="center", fontsize=9, transform=ax_txt.transAxes)
    ax_txt.text(0.5, 0.40,
                "Members 1-%d: %s" % (len(members),
                                      ", ".join("CESM-%s" % m for m in members)),
                ha="center", va="center", fontsize=9, transform=ax_txt.transAxes)

    # ===== shared colorbars =====
    cb1 = fig.colorbar(mag_mesh, ax=[ax_ph, ax_nh], orientation="vertical",
                       fraction=0.046, pad=0.01, aspect=20)
    cb1.set_label("%s-h %s-yr Magnitude (in)" % (duration, ari), fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    cb1.solids.set_alpha(1.0)

    cb2 = fig.colorbar(chg_mesh, ax=[ax_pc, ax_nc], orientation="vertical",
                       fraction=0.046, pad=0.01, aspect=20)
    cb2.set_label("Change (%)", fontsize=9)
    cb2.ax.tick_params(labelsize=8)
    cb2.solids.set_alpha(1.0)

    for ax_blank in (ax_lulc, ax_txt):
        blank_colorbar(fig, mag_mesh, ax_blank)

    encoded = save_and_encode(
        fig, fig_file,
        "./cesm_idf_ensmean_%s_%sh_%syr.png" % (land_cover, duration, ari))

    # ================= 2xN individual-member change figure =================
    # Always generated (not gated on member_fig_file being passed) so callers
    # get both figures back from a single call.
    if True:
        mx, my = 300, 300

        mgrid_lon = np.linspace(df["lon"].min() + pad, df["lon"].max() - pad, mx)
        mgrid_lat = np.linspace(df["lat"].min() + pad, df["lat"].max() - pad, my)
        MGX, MGY = np.meshgrid(mgrid_lon, mgrid_lat)
        mfields = {c: griddata(pts, df[c].values, (MGX, MGY), method="linear")
                   for c in prec_chg_cols + ng_chg_cols}

        ncols = len(members)
        mfig = plt.figure(figsize=(4.5 * ncols, 10.0), layout="compressed")
        mgs = mfig.add_gridspec(2, ncols)
        maxes = [[mfig.add_subplot(mgs[i, j], projection=proj)
                  for j in range(ncols)] for i in range(2)]
        mfig.get_layout_engine().set(w_pad=0.02, h_pad=0.02,
                                     wspace=0.02, hspace=0.02)

        rows = (("PREC-IDF: %s-h %s-yr" %(duration, ari), prec_chg_cols), ("NG-IDF: %s-h %s-yr" %(duration, ari), ng_chg_cols))
        letters = "abcdefghijklmnop"
        mesh = None
        for j, m in enumerate(members):
            for i, (var, cols) in enumerate(rows):
                ax = maxes[i][j]
                ax.set_extent(ext_pad, crs=data_crs)
                ax.add_image(tiler, ZOOM)
                mesh = ax.pcolormesh(MGX, MGY, mfields[cols[j]], cmap="RdBu_r",
                                     vmin=-dlim, vmax=dlim, transform=data_crs,
                                     alpha=ALPHA, shading="auto", zorder=3,
                                     rasterized=True)
                add_sites(ax)
                add_grid(ax)
                ax.text(0.97, 0.04, "(%s)" % letters[i * ncols + j],
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=8, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.7))
                if i == 0:
                    ax.set_title("Member %d (CESM-%s)" % (j + 1, m), fontsize=10)
                if j == 0:
                    ax.annotate(var, xy=(0.0, 0.5), xycoords="axes fraction",
                                xytext=(-38, 0), textcoords="offset points",
                                fontsize=10, ha="center", va="center", rotation=90)

        cbm = mfig.colorbar(mesh, ax=[a for row in maxes for a in row],
                            orientation="horizontal", location="bottom",
                            fraction=0.035, pad=0.04, aspect=40, shrink=0.6)
        cbm.set_label("Change (%)", fontsize=9)
        cbm.ax.tick_params(labelsize=8)
        cbm.solids.set_alpha(1.0)

        #mfig.suptitle("%s-h, %s-yr return level: individual member change"
        #              % (duration, ari), fontsize=11)

        member_encoded = save_and_encode(
            mfig, member_fig_file,
            "./cesm_idf_members_%s_%sh_%syr.png" % (land_cover, duration, ari))

    return encoded, member_encoded





def generate_cesm_near_term_idf_figure(land_cover, spatial_scenario, duration, ari,
                                         fig_file=None, lulc_file=None,
                                         members=("LE2", "LE4", "LE7", "LE9"),
                                         vmax_fixed=None, dlim_fixed=None,
                                         member_fig_file=None,
                                         dpi=150, save_pdf=False):
    """
    Generate the 2x3 CESM mid-century ensemble-mean IDF figure.

    Row 1: (a) land cover / study domain | (b) PREC-IDF hist  | (c) NG-IDF hist
    Row 2: [ensemble text]               | (d) PREC-IDF chg % | (e) NG-IDF chg %

    (b)(c) share one magnitude colorbar (inches); (d)(e) share one change
    colorbar (%). Members are inner-merged on (lat, lon) so every member
    contributes the same cell population. The ensemble change is derived from
    the ensemble-mean hist and ensemble-mean future -- NOT the average of the
    per-member changes (those differ whenever members have different baselines).

    Signature matches generate_wrf_idf_figure / generate_daymet_idf_figure:
    the data path is built from land_cover, duration, and ari.

        ./spatial_map/{land_cover}/CESM_{member}_Near_Term/
            ng_idf_return_{duration}h_ALL.csv

    Parameters
    ----------
    land_cover : str
        "open" | "evergreen" | "deciduous"
    spatial_scenario : str
        Scenario code (currently only "cesm_near_term" is supported here;
        kept for signature consistency with the other figure functions).
    duration : str or int
        Storm duration in hours, e.g. "24"
    ari : str or int
        Average recurrence interval in years, e.g. "25"
    fig_file : str or None
        Output path for the ensemble-mean figure. If None, defaults to
        "./cesm_idf_ensmean_{land_cover}_{duration}h_{ari}yr.png".
    lulc_file : str or None
        Optional land cover raster for panel (a).
    members : sequence of str
        CESM ensemble member codes.
    vmax_fixed : float or None
        Upper limit of the magnitude colorbar, in inches. None autoscales to the
        98th percentile. IDF magnitude scales strongly with duration, so a value
        pinned for 24h will not suit 1h.
    dlim_fixed : float or None
        Symmetric limit of the change colorbar, in percent. None autoscales to
        the 98th percentile of |change|. Percent change is dimensionless, so one
        value can safely be shared across durations and ARIs.
    member_fig_file : str or None
        If given, also write the companion 2xN individual-member change figure
        to this path, forced onto the same color scale. The return value is
        always the ensemble-mean figure.
    dpi : int
        Raster resolution; 150 matches the WRF/Daymet figures, 300 for print.
    save_pdf : bool
        Also write a vector PDF next to each PNG.

    Returns
    -------
    tuple(str, str)
        Two base64-encoded PNG strings, each prefixed with
        "data:image/png;base64,": (1) the 2x3 ensemble-mean figure, and
        (2) the companion 2xN figure showing each individual ensemble
        member's own-baseline percent change (PREC-IDF row / NG-IDF row).
    """

    MM_TO_IN = 1.0 / 25.4
    LATLON_DEC = 5          # rounding used to build the merge key
    EPS = 1e-6              # guard against divide-by-zero on the baseline


    members = list(members)
    ari = int(ari)
    base_dir = f"./spatial_map/{land_cover}/"

    # ---- Load data: one CSV per member, duration selects file, ari selects rows ----
    frames = []
    for m in members:
        path = os.path.join(base_dir, "CESM_%s_Near_Term" % m,
                            "ng_idf_return_%sh_ALL.csv" % duration)
        if not os.path.exists(path):
            raise FileNotFoundError("CSV not found: %s" % os.path.abspath(path))

        d = pd.read_csv(path)

        need = ["lat", "lon", "return_period_yr",
                "prec_idf_hist", "prec_idf_future",
                "ng_idf_hist", "ng_idf_future"]
        missing = [c for c in need if c not in d.columns]
        if missing:
            raise ValueError("%s is missing columns %s; found: %s"
                             % (path, missing, list(d.columns)))

        avail = sorted(d["return_period_yr"].unique())
        if ari not in avail:
            raise ValueError("ARI %d not in %s; available: %s" % (ari, path, avail))

        d = d[d["return_period_yr"] == ari].copy()
        if d.empty:
            raise ValueError("No rows for ARI=%d in %s" % (ari, path))

        d["lat"] = d["lat"].round(LATLON_DEC)
        d["lon"] = d["lon"].round(LATLON_DEC)
        d = d.drop_duplicates(subset=["lat", "lon"])

        d = d.rename(columns={
            "prec_idf_hist":   "prec_hist_%s" % m,
            "prec_idf_future": "prec_fut_%s" % m,
            "ng_idf_hist":     "ng_hist_%s" % m,
            "ng_idf_future":   "ng_fut_%s" % m,
        })
        frames.append(d[["lat", "lon",
                         "prec_hist_%s" % m, "prec_fut_%s" % m,
                         "ng_hist_%s" % m, "ng_fut_%s" % m]])

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on=["lat", "lon"], how="inner")
    if df.empty:
        raise ValueError("Inner merge across members produced zero cells. "
                         "Check that all members share the same grid.")

    # ---- per-member change, each against ITS OWN hist ----
    prec_chg_cols, ng_chg_cols = [], []
    for m in members:
        base = df["prec_hist_%s" % m].where(df["prec_hist_%s" % m].abs() > EPS)
        df["prec_chg_%s" % m] = (df["prec_fut_%s" % m] - base) / base * 100.0
        base = df["ng_hist_%s" % m].where(df["ng_hist_%s" % m].abs() > EPS)
        df["ng_chg_%s" % m] = (df["ng_fut_%s" % m] - base) / base * 100.0
        prec_chg_cols.append("prec_chg_%s" % m)
        ng_chg_cols.append("ng_chg_%s" % m)

    # ---- ensemble mean of the fields (raw mm), then the ensemble change ----
    prec_hist_mm = df[["prec_hist_%s" % m for m in members]].mean(axis=1)
    prec_fut_mm = df[["prec_fut_%s" % m for m in members]].mean(axis=1)
    ng_hist_mm = df[["ng_hist_%s" % m for m in members]].mean(axis=1)
    ng_fut_mm = df[["ng_fut_%s" % m for m in members]].mean(axis=1)

    prec_base = prec_hist_mm.where(prec_hist_mm.abs() > EPS)
    ng_base = ng_hist_mm.where(ng_hist_mm.abs() > EPS)
    df["prec_chg"] = (prec_fut_mm - prec_base) / prec_base * 100.0
    df["ng_chg"] = (ng_fut_mm - ng_base) / ng_base * 100.0

    # magnitudes for display, in inches
    df["prec_hist_mean_in"] = prec_hist_mm * MM_TO_IN
    df["ng_hist_mean_in"] = ng_hist_mm * MM_TO_IN

    print("ensemble cells after inner merge: %d" % len(df))
    print("domain-mean change (ensemble-mean based): PREC-IDF %+.2f %%, "
          "NG-IDF %+.2f %%" % (np.nanmean(df["prec_chg"].values),
                               np.nanmean(df["ng_chg"].values)))
    for m in members:
        print("  member %s own-baseline change: PREC-IDF %+.2f %%, NG-IDF %+.2f %%"
              % (m, np.nanmean(df["prec_chg_%s" % m].values),
                 np.nanmean(df["ng_chg_%s" % m].values)))

    # ---- shared color limits ----
    mag_vals = np.concatenate([df["prec_hist_mean_in"].values,
                               df["ng_hist_mean_in"].values])
    vmax = float(vmax_fixed) if vmax_fixed is not None \
        else float(np.nanpercentile(mag_vals, 98))

    chg_vals = np.concatenate([df["prec_chg"].values, df["ng_chg"].values])
    dlim = float(dlim_fixed) if dlim_fixed is not None \
        else float(np.nanpercentile(np.abs(chg_vals), 98))

    data_crs = ccrs.PlateCarree()

    # ---- interpolate onto regular grid ----
    pad = 0.05
    nx, ny = 400, 400
    grid_lon = np.linspace(df["lon"].min() + pad, df["lon"].max() - pad, nx)
    grid_lat = np.linspace(df["lat"].min() + pad, df["lat"].max() - pad, ny)
    GX, GY = np.meshgrid(grid_lon, grid_lat)
    pts = df[["lon", "lat"]].values

    map_cols = ["prec_hist_mean_in", "ng_hist_mean_in", "prec_chg", "ng_chg"]
    fields = {c: griddata(pts, df[c].values, (GX, GY), method="linear")
              for c in map_cols}

    # ---- basemap (same tiler / ZOOM / ALPHA as the WRF and Daymet figures) ----
    tiler = cimgt.OSM()
    proj = tiler.crs
    ZOOM = 9
    ALPHA = 0.55

    extent = [df["lon"].min(), df["lon"].max(), df["lat"].min(), df["lat"].max()]
    ext_pad = [extent[0] - 0.1, extent[1] + 0.1, extent[2] - 0.1, extent[3] + 0.1]

    sites = [
        ("Fort Wainwright", -147.6389, 64.8283),
        ("North Pole",      -147.3494, 64.7511),
        ("Delta Junction",  -145.7336, 64.0378),
    ]

    def add_sites(ax, fontsize=5, star=10):
        for name, clon, clat in sites:
            if extent[0] <= clon <= extent[1] and extent[2] <= clat <= extent[3]:
                ax.plot(clon, clat, marker="*", color="red", markersize=star,
                        markeredgecolor="white", markeredgewidth=0.9,
                        transform=data_crs, zorder=6)
                # Move the "North Pole" label lower so it does not overlap
                # with the nearby "Fort Wainwright" label.
                text_dy = -0.09 if name == "North Pole" else 0.05
                ax.text(clon + 0.05, clat + text_dy, name, fontsize=fontsize,
                        color="black", weight="bold", transform=data_crs, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                  ec="none", alpha=0.8))

    def add_grid(ax, labelsize=8):
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                          alpha=0.5, linestyle="--", zorder=4)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(-149, -144, 1))
        gl.ylocator = mticker.FixedLocator(np.arange(63, 66, 0.5))
        gl.xlabel_style = {"size": labelsize}
        gl.ylabel_style = {"size": labelsize}

    def add_lulc(ax, legend_fontsize=8):
        """Panel (a): basemap + optional land cover raster + domain outline."""
        ax.set_extent(ext_pad, crs=data_crs)
        ax.add_image(tiler, ZOOM)

        lulc_mesh = None
        if lulc_file is not None:
            import rioxarray as rxr
            from matplotlib.colors import ListedColormap, BoundaryNorm
            lc = rxr.open_rasterio(lulc_file, masked=True).squeeze()
            lc = lc.rio.reproject("EPSG:4326")
            classes = [1, 2, 3]
            names = ["Open", "Evergreen", "Deciduous"]
            colors = ["#D9C89E", "#1B4D3E", "#7FBF3F"]
            cmap_lc = ListedColormap(colors)
            norm_lc = BoundaryNorm(np.array(classes + [classes[-1] + 1]) - 0.5,
                                   cmap_lc.N)
            lulc_mesh = ax.pcolormesh(lc.x.values, lc.y.values, lc.values,
                                      cmap=cmap_lc, norm=norm_lc,
                                      transform=data_crs, alpha=0.75,
                                      shading="auto", zorder=2)
            handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=n)
                       for c, n in zip(colors, names)]
            ax.legend(handles=handles, loc="lower left",
                      fontsize=legend_fontsize, framealpha=0.85)

        ax.plot([extent[0], extent[1], extent[1], extent[0], extent[0]],
                [extent[2], extent[2], extent[3], extent[3], extent[2]],
                color="blue", linewidth=1.8, transform=data_crs, zorder=5,
                label="Study domain")
        add_sites(ax, fontsize=5, star=12)
        add_grid(ax)
        ax.set_title("(a) Land Cover / Study Domain", fontsize=9)
        if lulc_mesh is None:
            ax.legend(loc="lower left", fontsize=legend_fontsize, framealpha=0.85)

    def blank_colorbar(fig, mesh, ax):
        """Invisible colorbar so a text/context column matches the map columns."""
        cbx = fig.colorbar(mesh, ax=ax, orientation="vertical",
                           fraction=0.046, pad=0.01, aspect=20)
        cbx.outline.set_visible(False)
        cbx.ax.set_facecolor("none")
        cbx.ax.tick_params(size=0, labelsize=0, colors="none")
        for s in cbx.ax.spines.values():
            s.set_visible(False)
        cbx.solids.set_alpha(0.0)
        return cbx

    def save_and_encode(fig, path, default_name):
        if path is None:
            path = default_name
        path = os.path.abspath(path)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        if save_pdf:
            fig.savefig(os.path.splitext(path)[0] + ".pdf",
                        bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print("saved:", path, os.path.exists(path))
        with open(path, "rb") as fh:
            return "data:image/png;base64," + base64.b64encode(fh.read()).decode("utf-8")

    # ================= 2x3 ensemble-mean figure =================
    fig = plt.figure(figsize=(13.5, 8.4), layout="compressed")
    gs = fig.add_gridspec(2, 3)
    ax_lulc = fig.add_subplot(gs[0, 0], projection=proj)
    ax_ph = fig.add_subplot(gs[0, 1], projection=proj)
    ax_nh = fig.add_subplot(gs[0, 2], projection=proj)
    ax_txt = fig.add_subplot(gs[1, 0])
    ax_pc = fig.add_subplot(gs[1, 1], projection=proj)
    ax_nc = fig.add_subplot(gs[1, 2], projection=proj)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)

    add_lulc(ax_lulc)

    def draw(ax, col, title, cmap, vmin_, vmax_):
        ax.set_extent(ext_pad, crs=data_crs)
        ax.add_image(tiler, ZOOM)
        mesh = ax.pcolormesh(GX, GY, fields[col], cmap=cmap,
                             vmin=vmin_, vmax=vmax_, transform=data_crs,
                             alpha=ALPHA, shading="auto", zorder=3,
                             rasterized=True)
        add_sites(ax)
        add_grid(ax)
        ax.set_title(title, fontsize=9)
        return mesh

    mag_mesh = draw(ax_ph, "prec_hist_mean_in",
                    "(b) PREC-IDF Baseline (ensemble mean)", "viridis", 0.0, vmax)
    draw(ax_nh, "ng_hist_mean_in",
         "(c) NG-IDF Baseline (ensemble mean)", "viridis", 0.0, vmax)

    chg_mesh = draw(ax_pc, "prec_chg",
                    "(d) PREC-IDF Change (ensemble mean)", "RdBu_r", -dlim, dlim)
    draw(ax_nc, "ng_chg",
         "(e) NG-IDF Change (ensemble mean)", "RdBu_r", -dlim, dlim)

    # ===== blank cell: ensemble description =====
    ax_txt.set_axis_off()
    ax_txt.text(0.5, 0.66, "Near-Term Ensemble Mean",
                ha="center", va="center", fontsize=9, weight="bold",
                transform=ax_txt.transAxes)
    ax_txt.text(0.5, 0.52, "Average of %d model simulations" % len(members),
                ha="center", va="center", fontsize=9, transform=ax_txt.transAxes)
    ax_txt.text(0.5, 0.40,
                "Members 1-%d: %s" % (len(members),
                                      ", ".join("CESM-%s" % m for m in members)),
                ha="center", va="center", fontsize=9, transform=ax_txt.transAxes)

    # ===== shared colorbars =====
    cb1 = fig.colorbar(mag_mesh, ax=[ax_ph, ax_nh], orientation="vertical",
                       fraction=0.046, pad=0.01, aspect=20)
    cb1.set_label("%s-h %s-yr Magnitude (in)" % (duration, ari), fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    cb1.solids.set_alpha(1.0)

    cb2 = fig.colorbar(chg_mesh, ax=[ax_pc, ax_nc], orientation="vertical",
                       fraction=0.046, pad=0.01, aspect=20)
    cb2.set_label("Change (%)", fontsize=9)
    cb2.ax.tick_params(labelsize=8)
    cb2.solids.set_alpha(1.0)

    for ax_blank in (ax_lulc, ax_txt):
        blank_colorbar(fig, mag_mesh, ax_blank)

    encoded = save_and_encode(
        fig, fig_file,
        "./cesm_idf_ensmean_%s_%sh_%syr.png" % (land_cover, duration, ari))

    # ================= 2xN individual-member change figure =================
    # Always generated (not gated on member_fig_file being passed) so callers
    # get both figures back from a single call.
    if True:
        mx, my = 300, 300

        mgrid_lon = np.linspace(df["lon"].min() + pad, df["lon"].max() - pad, mx)
        mgrid_lat = np.linspace(df["lat"].min() + pad, df["lat"].max() - pad, my)
        MGX, MGY = np.meshgrid(mgrid_lon, mgrid_lat)
        mfields = {c: griddata(pts, df[c].values, (MGX, MGY), method="linear")
                   for c in prec_chg_cols + ng_chg_cols}

        ncols = len(members)
        mfig = plt.figure(figsize=(4.5 * ncols, 10.0), layout="compressed")
        mgs = mfig.add_gridspec(2, ncols)
        maxes = [[mfig.add_subplot(mgs[i, j], projection=proj)
                  for j in range(ncols)] for i in range(2)]
        mfig.get_layout_engine().set(w_pad=0.02, h_pad=0.02,
                                     wspace=0.02, hspace=0.02)

        rows = (("PREC-IDF: %s-h %s-yr" %(duration, ari), prec_chg_cols), ("NG-IDF: %s-h %s-yr" %(duration, ari), ng_chg_cols))
        letters = "abcdefghijklmnop"
        mesh = None
        for j, m in enumerate(members):
            for i, (var, cols) in enumerate(rows):
                ax = maxes[i][j]
                ax.set_extent(ext_pad, crs=data_crs)
                ax.add_image(tiler, ZOOM)
                mesh = ax.pcolormesh(MGX, MGY, mfields[cols[j]], cmap="RdBu_r",
                                     vmin=-dlim, vmax=dlim, transform=data_crs,
                                     alpha=ALPHA, shading="auto", zorder=3,
                                     rasterized=True)
                add_sites(ax)
                add_grid(ax)
                ax.text(0.97, 0.04, "(%s)" % letters[i * ncols + j],
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=8, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.7))
                if i == 0:
                    ax.set_title("Member %d (CESM-%s)" % (j + 1, m), fontsize=10)
                if j == 0:
                    ax.annotate(var, xy=(0.0, 0.5), xycoords="axes fraction",
                                xytext=(-38, 0), textcoords="offset points",
                                fontsize=10, ha="center", va="center", rotation=90)

        cbm = mfig.colorbar(mesh, ax=[a for row in maxes for a in row],
                            orientation="horizontal", location="bottom",
                            fraction=0.035, pad=0.04, aspect=40, shrink=0.6)
        cbm.set_label("Change (%)", fontsize=9)
        cbm.ax.tick_params(labelsize=8)
        cbm.solids.set_alpha(1.0)

        #mfig.suptitle("%s-h, %s-yr return level: individual member change"
        #              % (duration, ari), fontsize=11)

        member_encoded = save_and_encode(
            mfig, member_fig_file,
            "./cesm_idf_members_%s_%sh_%syr.png" % (land_cover, duration, ari))

    return encoded, member_encoded




# ============================================================================
# CESM Active Layer Thickness (ALT) figures -- website format.
#
# Two independent, self-contained functions. Uses the imports already at the
# top of test.py (os, pandas as pd, numpy as np, matplotlib.pyplot as plt,
# matplotlib.ticker as mticker, cartopy.crs as ccrs, cartopy.io.img_tiles as
# cimgt, scipy.interpolate.griddata, base64).
# ============================================================================


def generate_cesm_near_term_alt_figure(land_cover, spatial_scenario, duration, ari,
                                       fig_file=None, lulc_file=None,
                                       members=("LE2", "LE4", "LE7", "LE9"),
                                       vmax_fixed=None, dlim_fixed=None,
                                       member_fig_file=None,
                                       dpi=150, save_pdf=False):
    """
    Generate the 2x4 CESM near-term Active Layer Thickness (ALT) figure.

    Row 1: (a) land cover / study domain | (b) historical ALT (ensemble mean)
           | (c) near-term ALT (ensemble mean) | (d) change (ensemble mean)
    Row 2: (e)-(h) individual member change, each against its OWN baseline

    (b)(c) share one magnitude colorbar (ft); (d) and all of row 2 share one
    change colorbar (ft), so the ensemble panel and the member panels are
    directly comparable. Members are inner-merged on (lat, lon) so every member
    contributes the same cell population. The ensemble change is derived from
    the ensemble-mean hist and ensemble-mean future -- NOT the average of the
    per-member changes (those differ whenever members have different baselines).

    Signature matches generate_cesm_near_term_idf_figure: the data path is
    built from land_cover.

        ./spatial_map/{land_cover}/CESM_{member}_Near_Term/
            ng_idf_ald_results.csv

    Parameters
    ----------
    land_cover : str
        "open" | "evergreen" | "deciduous"
    spatial_scenario : str
        Scenario code (currently only "cesm_near_term" is supported here;
        kept for signature consistency with the other figure functions).
    duration : str or int
        Ignored. Kept for signature consistency with the IDF functions; ALT is
        not a return-period quantity.
    ari : str or int
        Ignored. Kept for signature consistency with the IDF functions.
    fig_file : str or None
        Output path for the 2x4 figure. If None, defaults to
        "./cesm_alt_ensmean_{land_cover}_near_term.png".
    lulc_file : str or None
        Optional land cover raster for panel (a).
    members : sequence of str
        CESM ensemble member codes. Four members give a clean 2x4; more members
        widen the second row and leave row 1 padded on the right.
    vmax_fixed : float or None
        Upper limit of the magnitude colorbar, in ft. None autoscales to the
        98th percentile; the lower limit always tracks the 2nd percentile. Pin
        this when near-term and mid-century are shown side by side.
    dlim_fixed : float or None
        Symmetric limit of the change colorbar, in ft. None autoscales to the
        98th percentile of |change|, falling back to a sequential YlOrRd scale
        from 0 when no cell thins.
    member_fig_file : str or None
        If given, also write a standalone 1xN member-change figure to this
        path, forced onto the same color scale. The member panels are already
        row 2 of the main figure, so this is optional. The return value is
        always the main figure.
    dpi : int
        Raster resolution; 150 matches the WRF/Daymet figures, 300 for print.
    save_pdf : bool
        Also write a vector PDF next to each PNG.

    Returns
    -------
    str
        Base64-encoded PNG string prefixed with "data:image/png;base64,"
        suitable for direct use in an HTML <img src="..."> tag.
    """

    M_TO_FT = 3.28084
    LATLON_DEC = 5          # rounding used to build the merge key

    members = list(members)
    base_dir = f"./spatial_map/{land_cover}/"

    # ---- Load data: one ALD CSV per member ----
    frames = []
    for m in members:
        path = os.path.join(base_dir, "CESM_%s_Near_Term" % m,
                            "ng_idf_ald_results.csv")
        if not os.path.exists(path):
            raise FileNotFoundError("CSV not found: %s" % os.path.abspath(path))

        d = pd.read_csv(path)

        need = ["lat", "lon", "mean_ALD_hist", "mean_ALD_future"]
        missing = [c for c in need if c not in d.columns]
        if missing:
            raise ValueError("%s is missing columns %s; found: %s"
                             % (path, missing, list(d.columns)))

        d = d.dropna(subset=["lat", "lon"])
        d["lat"] = d["lat"].round(LATLON_DEC)
        d["lon"] = d["lon"].round(LATLON_DEC)
        d = d.drop_duplicates(subset=["lat", "lon"])

        d = d.rename(columns={"mean_ALD_hist": "alt_hist_%s" % m,
                              "mean_ALD_future": "alt_fut_%s" % m})

        # ALD == 0 means no thaw / masked cell; exclude so it does not bias the
        # color scales. Applied only to the ALD columns, never to lat/lon, so
        # the domain extent stays identical to the IDF figure.
        for c in ("alt_hist_%s" % m, "alt_fut_%s" % m):
            d.loc[d[c] <= 0, c] = np.nan
            d[c] = d[c] * M_TO_FT

        frames.append(d[["lat", "lon", "alt_hist_%s" % m, "alt_fut_%s" % m]])

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on=["lat", "lon"], how="inner")
    if df.empty:
        raise ValueError("Inner merge across members produced zero cells. "
                         "Check that all members share the same grid.")

    # ---- per-member change, each against ITS OWN hist ----
    chg_cols = []
    for m in members:
        col = "alt_chg_%s" % m
        df[col] = df["alt_fut_%s" % m] - df["alt_hist_%s" % m]
        chg_cols.append(col)

    # ---- ensemble means (ft), then the ensemble change ----
    # skipna=False: a cell must be valid in every member to enter the mean, so
    # the mean does not shift where members drop out.
    df["alt_hist_mean"] = df[["alt_hist_%s" % m for m in members]].mean(
        axis=1, skipna=False)
    df["alt_fut_mean"] = df[["alt_fut_%s" % m for m in members]].mean(
        axis=1, skipna=False)
    df["alt_chg_mean"] = df["alt_fut_mean"] - df["alt_hist_mean"]

    print("ensemble cells after inner merge: %d" % len(df))
    print("valid ensemble-mean cells: %d"
          % int(np.isfinite(df["alt_chg_mean"]).sum()))
    print("domain-mean ALT: hist %.2f ft, future %.2f ft, change %+.2f ft"
          % (np.nanmean(df["alt_hist_mean"].values),
             np.nanmean(df["alt_fut_mean"].values),
             np.nanmean(df["alt_chg_mean"].values)))
    for m in members:
        print("  member %s: hist %.2f ft, future %.2f ft, change %+.2f ft"
              % (m, np.nanmean(df["alt_hist_%s" % m].values),
                 np.nanmean(df["alt_fut_%s" % m].values),
                 np.nanmean(df["alt_chg_%s" % m].values)))

    # ---- shared color limits ----
    mag_vals = np.concatenate([df["alt_hist_mean"].values,
                               df["alt_fut_mean"].values])
    mag_vals = mag_vals[np.isfinite(mag_vals)]
    vmin_mag = float(np.nanpercentile(mag_vals, 2))
    vmax_mag = float(vmax_fixed) if vmax_fixed is not None \
        else float(np.nanpercentile(mag_vals, 98))

    chg_vals = np.concatenate([df["alt_chg_mean"].values] +
                              [df[c].values for c in chg_cols])
    chg_vals = chg_vals[np.isfinite(chg_vals)]
    if dlim_fixed is not None:
        dlim = float(dlim_fixed)
        chg_cmap, chg_vmin, chg_vmax = "RdBu_r", -dlim, dlim
    elif np.nanmin(chg_vals) < 0:
        dlim = float(np.nanpercentile(np.abs(chg_vals), 98))
        chg_cmap, chg_vmin, chg_vmax = "RdBu_r", -dlim, dlim
    else:
        chg_cmap = "YlOrRd"
        chg_vmin = 0.0
        chg_vmax = float(np.nanpercentile(chg_vals, 98))

    data_crs = ccrs.PlateCarree()

    # ---- interpolate onto regular grid ----
    pad = 0.05
    nx, ny = 400, 400
    grid_lon = np.linspace(df["lon"].min() + pad, df["lon"].max() - pad, nx)
    grid_lat = np.linspace(df["lat"].min() + pad, df["lat"].max() - pad, ny)
    GX, GY = np.meshgrid(grid_lon, grid_lat)

    map_cols = ["alt_hist_mean", "alt_fut_mean", "alt_chg_mean"] + chg_cols
    fields = {}
    for c in map_cols:
        ok = np.isfinite(df[c].values)
        if ok.sum() < 4:
            raise ValueError("Column %s has only %d valid cells; cannot "
                             "interpolate." % (c, int(ok.sum())))
        fields[c] = griddata(df.loc[ok, ["lon", "lat"]].values,
                             df.loc[ok, c].values, (GX, GY), method="linear")

    # ---- basemap (same tiler / ZOOM / ALPHA as the WRF and Daymet figures) ----
    tiler = cimgt.OSM()
    proj = tiler.crs
    ZOOM = 9
    ALPHA = 0.55

    extent = [df["lon"].min(), df["lon"].max(), df["lat"].min(), df["lat"].max()]
    ext_pad = [extent[0] - 0.1, extent[1] + 0.1, extent[2] - 0.1, extent[3] + 0.1]
    print("ALT extent :", [round(v, 4) for v in extent])

    sites = [
        ("Fort Wainwright", -147.6389, 64.8283),
        ("North Pole",      -147.3494, 64.7511),
        ("Delta Junction",  -145.7336, 64.0378),
    ]

    def add_sites(ax, fontsize=5, star=10):
        for name, clon, clat in sites:
            if extent[0] <= clon <= extent[1] and extent[2] <= clat <= extent[3]:
                ax.plot(clon, clat, marker="*", color="red", markersize=star,
                        markeredgecolor="white", markeredgewidth=0.9,
                        transform=data_crs, zorder=6)
                # Move the "North Pole" label lower so it does not overlap
                # with the nearby "Fort Wainwright" label.
                text_dy = -0.09 if name == "North Pole" else 0.05
                ax.text(clon + 0.05, clat + text_dy, name, fontsize=fontsize,
                        color="black", weight="bold", transform=data_crs, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                  ec="none", alpha=0.8))

    def add_grid(ax, labelsize=7):
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                          alpha=0.5, linestyle="--", zorder=4)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(-149, -144, 1))
        gl.ylocator = mticker.FixedLocator(np.arange(63, 66, 0.5))
        gl.xlabel_style = {"size": labelsize}
        gl.ylabel_style = {"size": labelsize}

    def add_lulc(ax, legend_fontsize=7):
        """Panel (a): basemap + optional land cover raster + domain outline."""
        ax.set_extent(ext_pad, crs=data_crs)
        ax.add_image(tiler, ZOOM)

        lulc_mesh = None
        if lulc_file is not None:
            import rioxarray as rxr
            from matplotlib.colors import ListedColormap, BoundaryNorm
            lc = rxr.open_rasterio(lulc_file, masked=True).squeeze()
            lc = lc.rio.reproject("EPSG:4326")
            classes = [1, 2, 3]
            names = ["Open", "Evergreen", "Deciduous"]
            colors = ["#D9C89E", "#1B4D3E", "#7FBF3F"]
            cmap_lc = ListedColormap(colors)
            norm_lc = BoundaryNorm(np.array(classes + [classes[-1] + 1]) - 0.5,
                                   cmap_lc.N)
            lulc_mesh = ax.pcolormesh(lc.x.values, lc.y.values, lc.values,
                                      cmap=cmap_lc, norm=norm_lc,
                                      transform=data_crs, alpha=0.75,
                                      shading="auto", zorder=2)
            handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=n)
                       for c, n in zip(colors, names)]
            ax.legend(handles=handles, loc="lower left",
                      fontsize=legend_fontsize, framealpha=0.85)

        ax.plot([extent[0], extent[1], extent[1], extent[0], extent[0]],
                [extent[2], extent[2], extent[3], extent[3], extent[2]],
                color="blue", linewidth=1.8, transform=data_crs, zorder=5,
                label="Study domain")
        add_sites(ax, fontsize=5, star=12)
        add_grid(ax)
        ax.set_title("(a) Land Cover / Study Domain", fontsize=8)
        if lulc_mesh is None:
            ax.legend(loc="lower left", fontsize=legend_fontsize, framealpha=0.85)

    def blank_colorbar(fig, mesh, ax):
        """Invisible colorbar so a context column matches the map columns."""
        cbx = fig.colorbar(mesh, ax=ax, orientation="vertical",
                           fraction=0.046, pad=0.01, aspect=20)
        cbx.outline.set_visible(False)
        cbx.ax.set_facecolor("none")
        cbx.ax.tick_params(size=0, labelsize=0, colors="none")
        for s in cbx.ax.spines.values():
            s.set_visible(False)
        cbx.solids.set_alpha(0.0)
        return cbx

    def save_and_encode(fig, path, default_name):
        if path is None:
            path = default_name
        path = os.path.abspath(path)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        if save_pdf:
            fig.savefig(os.path.splitext(path)[0] + ".pdf",
                        bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print("saved:", path, os.path.exists(path))
        with open(path, "rb") as fh:
            return "data:image/png;base64," + \
                base64.b64encode(fh.read()).decode("utf-8")

    # ================= 2x4 figure =================
    ncols = max(4, len(members))
    fig = plt.figure(figsize=(4.0 * ncols, 8.0), layout="compressed")
    gs = fig.add_gridspec(2, ncols)
    ax_lulc = fig.add_subplot(gs[0, 0], projection=proj)
    ax_h = fig.add_subplot(gs[0, 1], projection=proj)
    ax_f = fig.add_subplot(gs[0, 2], projection=proj)
    ax_c = fig.add_subplot(gs[0, 3], projection=proj)
    maxes = [fig.add_subplot(gs[1, j], projection=proj)
             for j in range(len(members))]
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)

    def draw(ax, col, title, cmap, vmin_, vmax_):
        ax.set_extent(ext_pad, crs=data_crs)
        ax.add_image(tiler, ZOOM)
        mesh = ax.pcolormesh(GX, GY, fields[col], cmap=cmap,
                             vmin=vmin_, vmax=vmax_, transform=data_crs,
                             alpha=ALPHA, shading="auto", zorder=3,
                             rasterized=True)
        add_sites(ax)
        add_grid(ax)
        ax.set_title(title, fontsize=8)
        return mesh

    # ===== Row 1 =====
    add_lulc(ax_lulc)
    mag_mesh = draw(ax_h, "alt_hist_mean",
                    "(b) Baseline ALT (ensemble mean)",
                    "YlGnBu", vmin_mag, vmax_mag)
    draw(ax_f, "alt_fut_mean",
         "(c) Near-Term ALT (ensemble mean)",
         "YlGnBu", vmin_mag, vmax_mag)
    chg_mesh = draw(ax_c, "alt_chg_mean",
                    "(d) ALT Change (ensemble mean)",
                    chg_cmap, chg_vmin, chg_vmax)

    # ===== Row 2: individual member change =====
    letters = "abcdefghijklmnop"
    for j, m in enumerate(members):
        ax = maxes[j]
        draw(ax, "alt_chg_%s" % m,
             "(%s) Member %d (CESM-%s) Change" % (letters[4 + j], j + 1, m),
             chg_cmap, chg_vmin, chg_vmax)
        ax.text(0.97, 0.04, "own baseline", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6, zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.7))

    # ===== colorbars =====
    cb1 = fig.colorbar(mag_mesh, ax=[ax_h, ax_f], orientation="vertical",
                       fraction=0.046, pad=0.01, aspect=20)
    cb1.set_label("Mean ALT (ft)", fontsize=8)
    cb1.ax.tick_params(labelsize=7)
    cb1.solids.set_alpha(1.0)

    # shrink=0.5 over the two-row span gives the same length as cb1's one-row
    # span; aspect=20 then yields the same width.
    cb2 = fig.colorbar(chg_mesh, ax=[ax_c] + maxes, orientation="vertical",
                       fraction=0.046, pad=0.01, aspect=20, shrink=0.5)
    cb2.set_label("ALT Change (ft)", fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    cb2.solids.set_alpha(1.0)

    # invisible colorbar on (a) so column 0 matches the map columns in width
    blank_colorbar(fig, mag_mesh, ax_lulc)

    encoded = save_and_encode(
        fig, fig_file,
        "./cesm_alt_ensmean_%s_near_term.png" % land_cover)

    # ================= optional standalone member figure =================
    # The member panels are already row 2 above, so this is only built when a
    # caller explicitly asks for them on their own.
    if member_fig_file is not None:
        mfig = plt.figure(figsize=(4.0 * len(members), 4.4), layout="compressed")
        mgs = mfig.add_gridspec(1, len(members))
        mm_axes = [mfig.add_subplot(mgs[0, j], projection=proj)
                   for j in range(len(members))]
        mfig.get_layout_engine().set(w_pad=0.02, h_pad=0.02,
                                     wspace=0.02, hspace=0.02)

        mmesh = None
        for j, m in enumerate(members):
            ax = mm_axes[j]
            ax.set_extent(ext_pad, crs=data_crs)
            ax.add_image(tiler, ZOOM)
            mmesh = ax.pcolormesh(GX, GY, fields["alt_chg_%s" % m],
                                  cmap=chg_cmap, vmin=chg_vmin, vmax=chg_vmax,
                                  transform=data_crs, alpha=ALPHA,
                                  shading="auto", zorder=3, rasterized=True)
            add_sites(ax)
            add_grid(ax)
            ax.set_title("(%s) Member %d (CESM-%s) Change"
                         % (letters[j], j + 1, m), fontsize=8)

        cbm = mfig.colorbar(mmesh, ax=mm_axes, orientation="horizontal",
                            location="bottom", fraction=0.035, pad=0.04,
                            aspect=40, shrink=0.6)
        cbm.set_label("ALT Change (ft)", fontsize=8)
        cbm.ax.tick_params(labelsize=7)
        cbm.solids.set_alpha(1.0)

        save_and_encode(mfig, member_fig_file,
                        "./cesm_alt_members_%s_near_term.png" % land_cover)

    return encoded


def generate_cesm_mid_century_alt_figure(land_cover, spatial_scenario, duration, ari,
                                       fig_file=None, lulc_file=None,
                                       members=("LE2", "LE4", "LE7", "LE9"),
                                       vmax_fixed=None, dlim_fixed=None,
                                       member_fig_file=None,
                                       dpi=150, save_pdf=False):
    """
    Generate the 2x4 CESM mid-century Active Layer Thickness (ALT) figure.

    Row 1: (a) land cover / study domain | (b) historical ALT (ensemble mean)
           | (c) mid-century ALT (ensemble mean) | (d) change (ensemble mean)
    Row 2: (e)-(h) individual member change, each against its OWN baseline

    (b)(c) share one magnitude colorbar (ft); (d) and all of row 2 share one
    change colorbar (ft), so the ensemble panel and the member panels are
    directly comparable. Members are inner-merged on (lat, lon) so every member
    contributes the same cell population. The ensemble change is derived from
    the ensemble-mean hist and ensemble-mean future -- NOT the average of the
    per-member changes (those differ whenever members have different baselines).

    Signature matches generate_cesm_mid_century_idf_figure: the data path is
    built from land_cover.

        ./spatial_map/{land_cover}/CESM_{member}_Mid_Century/
            ng_idf_ald_results.csv

    Parameters
    ----------
    land_cover : str
        "open" | "evergreen" | "deciduous"
    spatial_scenario : str
        Scenario code (currently only "cesm_mid_century" is supported here;
        kept for signature consistency with the other figure functions).
    duration : str or int
        Ignored. Kept for signature consistency with the IDF functions; ALT is
        not a return-period quantity.
    ari : str or int
        Ignored. Kept for signature consistency with the IDF functions.
    fig_file : str or None
        Output path for the 2x4 figure. If None, defaults to
        "./cesm_alt_ensmean_{land_cover}_mid_century.png".
    lulc_file : str or None
        Optional land cover raster for panel (a).
    members : sequence of str
        CESM ensemble member codes. Four members give a clean 2x4; more members
        widen the second row and leave row 1 padded on the right.
    vmax_fixed : float or None
        Upper limit of the magnitude colorbar, in ft. None autoscales to the
        98th percentile; the lower limit always tracks the 2nd percentile. Pin
        this when mid-century and near-term are shown side by side.
    dlim_fixed : float or None
        Symmetric limit of the change colorbar, in ft. None autoscales to the
        98th percentile of |change|, falling back to a sequential YlOrRd scale
        from 0 when no cell thins.
    member_fig_file : str or None
        If given, also write a standalone 1xN member-change figure to this
        path, forced onto the same color scale. The member panels are already
        row 2 of the main figure, so this is optional. The return value is
        always the main figure.
    dpi : int
        Raster resolution; 150 matches the WRF/Daymet figures, 300 for print.
    save_pdf : bool
        Also write a vector PDF next to each PNG.

    Returns
    -------
    str
        Base64-encoded PNG string prefixed with "data:image/png;base64,"
        suitable for direct use in an HTML <img src="..."> tag.
    """

    M_TO_FT = 3.28084
    LATLON_DEC = 5          # rounding used to build the merge key

    members = list(members)
    base_dir = f"./spatial_map/{land_cover}/"

    # ---- Load data: one ALD CSV per member ----
    frames = []
    for m in members:
        path = os.path.join(base_dir, "CESM_%s_Mid_Century" % m,
                            "ng_idf_ald_results.csv")
        if not os.path.exists(path):
            raise FileNotFoundError("CSV not found: %s" % os.path.abspath(path))

        d = pd.read_csv(path)

        need = ["lat", "lon", "mean_ALD_hist", "mean_ALD_future"]
        missing = [c for c in need if c not in d.columns]
        if missing:
            raise ValueError("%s is missing columns %s; found: %s"
                             % (path, missing, list(d.columns)))

        d = d.dropna(subset=["lat", "lon"])
        d["lat"] = d["lat"].round(LATLON_DEC)
        d["lon"] = d["lon"].round(LATLON_DEC)
        d = d.drop_duplicates(subset=["lat", "lon"])

        d = d.rename(columns={"mean_ALD_hist": "alt_hist_%s" % m,
                              "mean_ALD_future": "alt_fut_%s" % m})

        # ALD == 0 means no thaw / masked cell; exclude so it does not bias the
        # color scales. Applied only to the ALD columns, never to lat/lon, so
        # the domain extent stays identical to the IDF figure.
        for c in ("alt_hist_%s" % m, "alt_fut_%s" % m):
            d.loc[d[c] <= 0, c] = np.nan
            d[c] = d[c] * M_TO_FT

        frames.append(d[["lat", "lon", "alt_hist_%s" % m, "alt_fut_%s" % m]])

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on=["lat", "lon"], how="inner")
    if df.empty:
        raise ValueError("Inner merge across members produced zero cells. "
                         "Check that all members share the same grid.")

    # ---- per-member change, each against ITS OWN hist ----
    chg_cols = []
    for m in members:
        col = "alt_chg_%s" % m
        df[col] = df["alt_fut_%s" % m] - df["alt_hist_%s" % m]
        chg_cols.append(col)

    # ---- ensemble means (ft), then the ensemble change ----
    # skipna=False: a cell must be valid in every member to enter the mean, so
    # the mean does not shift where members drop out.
    df["alt_hist_mean"] = df[["alt_hist_%s" % m for m in members]].mean(
        axis=1, skipna=False)
    df["alt_fut_mean"] = df[["alt_fut_%s" % m for m in members]].mean(
        axis=1, skipna=False)
    df["alt_chg_mean"] = df["alt_fut_mean"] - df["alt_hist_mean"]

    print("ensemble cells after inner merge: %d" % len(df))
    print("valid ensemble-mean cells: %d"
          % int(np.isfinite(df["alt_chg_mean"]).sum()))
    print("domain-mean ALT: hist %.2f ft, future %.2f ft, change %+.2f ft"
          % (np.nanmean(df["alt_hist_mean"].values),
             np.nanmean(df["alt_fut_mean"].values),
             np.nanmean(df["alt_chg_mean"].values)))
    for m in members:
        print("  member %s: hist %.2f ft, future %.2f ft, change %+.2f ft"
              % (m, np.nanmean(df["alt_hist_%s" % m].values),
                 np.nanmean(df["alt_fut_%s" % m].values),
                 np.nanmean(df["alt_chg_%s" % m].values)))

    # ---- shared color limits ----
    mag_vals = np.concatenate([df["alt_hist_mean"].values,
                               df["alt_fut_mean"].values])
    mag_vals = mag_vals[np.isfinite(mag_vals)]
    vmin_mag = float(np.nanpercentile(mag_vals, 2))
    vmax_mag = float(vmax_fixed) if vmax_fixed is not None \
        else float(np.nanpercentile(mag_vals, 98))

    chg_vals = np.concatenate([df["alt_chg_mean"].values] +
                              [df[c].values for c in chg_cols])
    chg_vals = chg_vals[np.isfinite(chg_vals)]
    if dlim_fixed is not None:
        dlim = float(dlim_fixed)
        chg_cmap, chg_vmin, chg_vmax = "RdBu_r", -dlim, dlim
    elif np.nanmin(chg_vals) < 0:
        dlim = float(np.nanpercentile(np.abs(chg_vals), 98))
        chg_cmap, chg_vmin, chg_vmax = "RdBu_r", -dlim, dlim
    else:
        chg_cmap = "YlOrRd"
        chg_vmin = 0.0
        chg_vmax = float(np.nanpercentile(chg_vals, 98))

    data_crs = ccrs.PlateCarree()

    # ---- interpolate onto regular grid ----
    pad = 0.05
    nx, ny = 400, 400
    grid_lon = np.linspace(df["lon"].min() + pad, df["lon"].max() - pad, nx)
    grid_lat = np.linspace(df["lat"].min() + pad, df["lat"].max() - pad, ny)
    GX, GY = np.meshgrid(grid_lon, grid_lat)

    map_cols = ["alt_hist_mean", "alt_fut_mean", "alt_chg_mean"] + chg_cols
    fields = {}
    for c in map_cols:
        ok = np.isfinite(df[c].values)
        if ok.sum() < 4:
            raise ValueError("Column %s has only %d valid cells; cannot "
                             "interpolate." % (c, int(ok.sum())))
        fields[c] = griddata(df.loc[ok, ["lon", "lat"]].values,
                             df.loc[ok, c].values, (GX, GY), method="linear")

    # ---- basemap (same tiler / ZOOM / ALPHA as the WRF and Daymet figures) ----
    tiler = cimgt.OSM()
    proj = tiler.crs
    ZOOM = 9
    ALPHA = 0.55

    extent = [df["lon"].min(), df["lon"].max(), df["lat"].min(), df["lat"].max()]
    ext_pad = [extent[0] - 0.1, extent[1] + 0.1, extent[2] - 0.1, extent[3] + 0.1]
    print("ALT extent :", [round(v, 4) for v in extent])

    sites = [
        ("Fort Wainwright", -147.6389, 64.8283),
        ("North Pole",      -147.3494, 64.7511),
        ("Delta Junction",  -145.7336, 64.0378),
    ]

    def add_sites(ax, fontsize=5, star=10):
        for name, clon, clat in sites:
            if extent[0] <= clon <= extent[1] and extent[2] <= clat <= extent[3]:
                ax.plot(clon, clat, marker="*", color="red", markersize=star,
                        markeredgecolor="white", markeredgewidth=0.9,
                        transform=data_crs, zorder=6)
                # Move the "North Pole" label lower so it does not overlap
                # with the nearby "Fort Wainwright" label.
                text_dy = -0.09 if name == "North Pole" else 0.05
                ax.text(clon + 0.05, clat + text_dy, name, fontsize=fontsize,
                        color="black", weight="bold", transform=data_crs, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                  ec="none", alpha=0.8))

    def add_grid(ax, labelsize=7):
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                          alpha=0.5, linestyle="--", zorder=4)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(-149, -144, 1))
        gl.ylocator = mticker.FixedLocator(np.arange(63, 66, 0.5))
        gl.xlabel_style = {"size": labelsize}
        gl.ylabel_style = {"size": labelsize}

    def add_lulc(ax, legend_fontsize=7):
        """Panel (a): basemap + optional land cover raster + domain outline."""
        ax.set_extent(ext_pad, crs=data_crs)
        ax.add_image(tiler, ZOOM)

        lulc_mesh = None
        if lulc_file is not None:
            import rioxarray as rxr
            from matplotlib.colors import ListedColormap, BoundaryNorm
            lc = rxr.open_rasterio(lulc_file, masked=True).squeeze()
            lc = lc.rio.reproject("EPSG:4326")
            classes = [1, 2, 3]
            names = ["Open", "Evergreen", "Deciduous"]
            colors = ["#D9C89E", "#1B4D3E", "#7FBF3F"]
            cmap_lc = ListedColormap(colors)
            norm_lc = BoundaryNorm(np.array(classes + [classes[-1] + 1]) - 0.5,
                                   cmap_lc.N)
            lulc_mesh = ax.pcolormesh(lc.x.values, lc.y.values, lc.values,
                                      cmap=cmap_lc, norm=norm_lc,
                                      transform=data_crs, alpha=0.75,
                                      shading="auto", zorder=2)
            handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=n)
                       for c, n in zip(colors, names)]
            ax.legend(handles=handles, loc="lower left",
                      fontsize=legend_fontsize, framealpha=0.85)

        ax.plot([extent[0], extent[1], extent[1], extent[0], extent[0]],
                [extent[2], extent[2], extent[3], extent[3], extent[2]],
                color="blue", linewidth=1.8, transform=data_crs, zorder=5,
                label="Study domain")
        add_sites(ax, fontsize=5, star=12)
        add_grid(ax)
        ax.set_title("(a) Land Cover / Study Domain", fontsize=8)
        if lulc_mesh is None:
            ax.legend(loc="lower left", fontsize=legend_fontsize, framealpha=0.85)

    def blank_colorbar(fig, mesh, ax):
        """Invisible colorbar so a context column matches the map columns."""
        cbx = fig.colorbar(mesh, ax=ax, orientation="vertical",
                           fraction=0.046, pad=0.01, aspect=20)
        cbx.outline.set_visible(False)
        cbx.ax.set_facecolor("none")
        cbx.ax.tick_params(size=0, labelsize=0, colors="none")
        for s in cbx.ax.spines.values():
            s.set_visible(False)
        cbx.solids.set_alpha(0.0)
        return cbx

    def save_and_encode(fig, path, default_name):
        if path is None:
            path = default_name
        path = os.path.abspath(path)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        if save_pdf:
            fig.savefig(os.path.splitext(path)[0] + ".pdf",
                        bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print("saved:", path, os.path.exists(path))
        with open(path, "rb") as fh:
            return "data:image/png;base64," + \
                base64.b64encode(fh.read()).decode("utf-8")

    # ================= 2x4 figure =================
    ncols = max(4, len(members))
    fig = plt.figure(figsize=(4.0 * ncols, 8.0), layout="compressed")
    gs = fig.add_gridspec(2, ncols)
    ax_lulc = fig.add_subplot(gs[0, 0], projection=proj)
    ax_h = fig.add_subplot(gs[0, 1], projection=proj)
    ax_f = fig.add_subplot(gs[0, 2], projection=proj)
    ax_c = fig.add_subplot(gs[0, 3], projection=proj)
    maxes = [fig.add_subplot(gs[1, j], projection=proj)
             for j in range(len(members))]
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)

    def draw(ax, col, title, cmap, vmin_, vmax_):
        ax.set_extent(ext_pad, crs=data_crs)
        ax.add_image(tiler, ZOOM)
        mesh = ax.pcolormesh(GX, GY, fields[col], cmap=cmap,
                             vmin=vmin_, vmax=vmax_, transform=data_crs,
                             alpha=ALPHA, shading="auto", zorder=3,
                             rasterized=True)
        add_sites(ax)
        add_grid(ax)
        ax.set_title(title, fontsize=8)
        return mesh

    # ===== Row 1 =====
    add_lulc(ax_lulc)
    mag_mesh = draw(ax_h, "alt_hist_mean",
                    "(b) Baseline ALT (ensemble mean)",
                    "YlGnBu", vmin_mag, vmax_mag)
    draw(ax_f, "alt_fut_mean",
         "(c) Mid-Century ALT (ensemble mean)",
         "YlGnBu", vmin_mag, vmax_mag)
    chg_mesh = draw(ax_c, "alt_chg_mean",
                    "(d) ALT Change (ensemble mean)",
                    chg_cmap, chg_vmin, chg_vmax)

    # ===== Row 2: individual member change =====
    letters = "abcdefghijklmnop"
    for j, m in enumerate(members):
        ax = maxes[j]
        draw(ax, "alt_chg_%s" % m,
             "(%s) Member %d (CESM-%s) Change" % (letters[4 + j], j + 1, m),
             chg_cmap, chg_vmin, chg_vmax)
        ax.text(0.97, 0.04, "own baseline", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6, zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.7))

    # ===== colorbars =====
    cb1 = fig.colorbar(mag_mesh, ax=[ax_h, ax_f], orientation="vertical",
                       fraction=0.046, pad=0.01, aspect=20)
    cb1.set_label("Mean ALT (ft)", fontsize=8)
    cb1.ax.tick_params(labelsize=7)
    cb1.solids.set_alpha(1.0)

    # shrink=0.5 over the two-row span gives the same length as cb1's one-row
    # span; aspect=20 then yields the same width.
    cb2 = fig.colorbar(chg_mesh, ax=[ax_c] + maxes, orientation="vertical",
                       fraction=0.046, pad=0.01, aspect=20, shrink=0.5)
    cb2.set_label("ALT Change (ft)", fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    cb2.solids.set_alpha(1.0)

    # invisible colorbar on (a) so column 0 matches the map columns in width
    blank_colorbar(fig, mag_mesh, ax_lulc)

    encoded = save_and_encode(
        fig, fig_file,
        "./cesm_alt_ensmean_%s_mid_century.png" % land_cover)

    # ================= optional standalone member figure =================
    # The member panels are already row 2 above, so this is only built when a
    # caller explicitly asks for them on their own.
    if member_fig_file is not None:
        mfig = plt.figure(figsize=(4.0 * len(members), 4.4), layout="compressed")
        mgs = mfig.add_gridspec(1, len(members))
        mm_axes = [mfig.add_subplot(mgs[0, j], projection=proj)
                   for j in range(len(members))]
        mfig.get_layout_engine().set(w_pad=0.02, h_pad=0.02,
                                     wspace=0.02, hspace=0.02)

        mmesh = None
        for j, m in enumerate(members):
            ax = mm_axes[j]
            ax.set_extent(ext_pad, crs=data_crs)
            ax.add_image(tiler, ZOOM)
            mmesh = ax.pcolormesh(GX, GY, fields["alt_chg_%s" % m],
                                  cmap=chg_cmap, vmin=chg_vmin, vmax=chg_vmax,
                                  transform=data_crs, alpha=ALPHA,
                                  shading="auto", zorder=3, rasterized=True)
            add_sites(ax)
            add_grid(ax)
            ax.set_title("(%s) Member %d (CESM-%s) Change"
                         % (letters[j], j + 1, m), fontsize=8)

        cbm = mfig.colorbar(mmesh, ax=mm_axes, orientation="horizontal",
                            location="bottom", fraction=0.035, pad=0.04,
                            aspect=40, shrink=0.6)
        cbm.set_label("ALT Change (ft)", fontsize=8)
        cbm.ax.tick_params(labelsize=7)
        cbm.solids.set_alpha(1.0)

        save_and_encode(mfig, member_fig_file,
                        "./cesm_alt_members_%s_mid_century.png" % land_cover)

    return encoded