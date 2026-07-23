import os
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from scipy.interpolate import griddata

MM_TO_IN = 1.0 / 25.4


def plot_wrf_idf_3x3(land_cover="open",
                     duration="1",
                     ari=25,
                     lulc_file=None,
                     fig_file=None):
    """
    3x3 WRF IDF figure combining PREC-IDF and NG-IDF.

    Row 1: (a) land cover  | (b) PREC-IDF hist (in)  | (c) NG-IDF hist (in)
    Row 2: [scenario text] | (d) PREC medium chg %   | (e) NG medium chg %
    Row 3: [scenario text] | (f) PREC high chg %     | (g) NG high chg %

    (b)(c) share one magnitude colorbar (inches); (d)-(g) share one change colorbar (%).
    """

    # ---- Load data: duration selects file, ari selects rows ----
    idf_path = f"./spatial_map/{land_cover}/WRF/ng_idf_return_{duration}h_ALL.csv"
    idf_path = 'ng_idf_return_1h_ALL.csv'
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

    def add_sites(ax, fontsize=7, star=10):
        for name, clon, clat in sites:
            if extent[0] <= clon <= extent[1] and extent[2] <= clat <= extent[3]:
                ax.plot(clon, clat, marker="*", color="red", markersize=star,
                        markeredgecolor="white", markeredgewidth=0.9,
                        transform=data_crs, zorder=6)
                ax.text(clon + 0.05, clat + 0.05, name, fontsize=fontsize,
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

    add_sites(ax, fontsize=8, star=12)
    add_grid(ax)
    ax.set_title("(a) Land Cover / Study Domain", fontsize=11)
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
        ax.set_title(title, fontsize=11)
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
    fig.text(xm, ym, "Medium Scenario:\nModerately Hotter\nand Drier",
             ha="center", va="center", fontsize=11, weight="bold",
             color="black", wrap=True)

    xh, yh = cell_center(axf[6])
    fig.text(xh, yh, "High Scenario:\nSeverely Hotter\nand Drier",
             ha="center", va="center", fontsize=11, weight="bold",
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


if __name__ == "__main__":
    uri = plot_wrf_idf_3x3(land_cover="open", duration="1", ari=25)
    print(len(uri))