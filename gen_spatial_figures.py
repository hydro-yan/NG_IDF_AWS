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
    tiler = cimgt.OSM()
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

    def add_sites(ax, fontsize=7, star=10):
        for name, clon, clat in sites:
            if extent[0] <= clon <= extent[1] and extent[2] <= clat <= extent[3]:
                ax.plot(clon, clat, marker="*", color="red", markersize=star,
                        markeredgecolor="white", markeredgewidth=0.9,
                        transform=data_crs, zorder=6)
                # Move the "North Pole" label lower so it doesn't overlap
                # with the nearby "Fort Wainwright" label.
                text_dy = -0.09 if name == "North Pole" else 0.05
                ax.text(clon+0.05, clat+text_dy, name, fontsize=fontsize, color="black",
                        weight="bold", transform=data_crs, zorder=7,
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
        (1, "prec_idf_hist_in", "(b) PREC-IDF Historical"),
        (2, "ng_idf_hist_in",   "(c) NG-IDF Historical"),
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
        (1, "alt_hist_ft", "(b) Historical Mean ALT", None),
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