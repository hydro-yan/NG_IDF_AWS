"""
CESM 4-member ensemble IDF figure generator.

Companion to generate_wrf_idf_figure() in gen_spatial_figures.py.
Pure ASCII, no unicode, NERSC-safe.

Layout (2 x 3):
    (a) land cover / domain | (b) PREC-IDF hist mean | (c) NG-IDF hist mean
    [member name panel]     | (d) PREC-IDF change    | (e) NG-IDF change

Ensemble aggregation
--------------------
The ensemble mean is taken on the FIELDS first, then the change is derived
from those means:

    hist_mean = mean_over_members(hist)
    fut_mean  = mean_over_members(future)
    change    = (fut_mean - hist_mean) / hist_mean * 100

Members: CESM_LE2 / LE4 / LE7 / LE9, each folder holding
ng_idf_return_{duration}h_ALL.csv with columns
    return_period_yr, lat, lon, prec_idf_hist, prec_idf_future,
    ng_idf_hist, ng_idf_future
"""

import os
import base64
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from scipy.interpolate import griddata


MEMBERS = ["LE2", "LE4", "LE7", "LE9"]
MM_TO_IN = 1.0 / 25.4
LATLON_DEC = 5          # rounding used to build the merge key
EPS = 1e-6              # guard against divide-by-zero on the mean baseline


# ---------------------------------------------------------------------------
# shared basemap / site-marker / gridline setup
#
# Both figures MUST use this same projection (PROJ, from the OSM tiler).
# It is a Web Mercator projection, which stretches latitude relative to
# longitude - it is NOT the same as plotting raw lon/lat degrees with
# ax.set_aspect("equal"). Using plain axes for one figure and GeoAxes for
# the other produces panels with two different width:height ratios even
# though both cover the identical lon/lat domain. Defining these once here
# and importing them into both figure functions keeps that from happening
# again.
# ---------------------------------------------------------------------------
TILER = cimgt.OSM()
PROJ = TILER.crs
ZOOM = 9
BASEMAP_ALPHA = 0.55

SITES = [
    ("Fort Wainwright", -147.6389, 64.8283),
    ("North Pole",      -147.3494, 64.7511),
    ("Delta Junction",  -145.7336, 64.0378),
]


def add_sites(ax, extent, data_crs, fontsize=5, star=10):
    """Plot the site markers and labels that fall inside extent."""
    for name, clon, clat in SITES:
        if extent[0] <= clon <= extent[1] and extent[2] <= clat <= extent[3]:
            ax.plot(clon, clat, marker="*", color="red", markersize=star,
                    markeredgecolor="white", markeredgewidth=0.9,
                    transform=data_crs, zorder=6)
            text_dy = -0.09 if name == "North Pole" else 0.05
            ax.text(clon + 0.05, clat + text_dy, name, fontsize=fontsize,
                    color="black", weight="bold", transform=data_crs,
                    zorder=7,
                    bbox=dict(boxstyle="round,pad=0.16", fc="white",
                              ec="none", alpha=0.8))


def add_grid(ax):
    """Draw dashed lat/lon gridlines with labels on the left and bottom."""
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                      alpha=0.5, linestyle="--", zorder=4)
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(-149, -144, 1))
    gl.ylocator = mticker.FixedLocator(np.arange(63, 66, 0.5))
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------
def load_cesm_ensemble(base_dir, duration, ari, members=MEMBERS):
    """
    Read all member CSVs for one duration, keep the ARI rows, inner-merge on
    (lat, lon) so every member contributes the same cell population, average
    the historical and future fields across members, and derive the change
    from those ensemble means.

    Returns
    -------
    df : DataFrame with columns
         lat, lon,
         prec_hist_mean_in, prec_fut_mean_in    (inches, ensemble mean)
         ng_hist_mean_in,   ng_fut_mean_in      (inches, ensemble mean)
         prec_chg, ng_chg                       (percent, ensemble-mean-based)
         prec_chg_{member}, ng_chg_{member}     (percent, one per member,
                                                  each against ITS OWN hist -
                                                  not the ensemble-mean hist)
    """
    ari = int(ari)
    frames = []

    for m in members:
        path = os.path.join(base_dir, "CESM_%s_Mid_Century" % m,
                            "ng_idf_return_%sh_ALL.csv" % duration)
        if not os.path.exists(path):
            raise FileNotFoundError("CSV not found: %s" % os.path.abspath(path))

        d = pd.read_csv(path)
        avail = sorted(d["return_period_yr"].unique())
        if ari not in avail:
            raise ValueError("ARI %d not in %s; available: %s"
                             % (ari, path, avail))

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
        keep = ["lat", "lon",
                "prec_hist_%s" % m, "prec_fut_%s" % m,
                "ng_hist_%s" % m, "ng_fut_%s" % m]
        frames.append(d[keep])

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on=["lat", "lon"], how="inner")

    if df.empty:
        raise ValueError("Inner merge across members produced zero cells. "
                         "Check that all members share the same grid.")

    # ---- per-member change, each against ITS OWN hist ----
    # This is what individual-member panels must use: LE2's change comes from
    # LE2's own hist and future, never from the ensemble-mean hist.
    chg_cols = []
    for m in members:
        base = df["prec_hist_%s" % m].where(df["prec_hist_%s" % m].abs() > EPS)
        df["prec_chg_%s" % m] = (df["prec_fut_%s" % m] - base) / base * 100.0
        base = df["ng_hist_%s" % m].where(df["ng_hist_%s" % m].abs() > EPS)
        df["ng_chg_%s" % m] = (df["ng_fut_%s" % m] - base) / base * 100.0
        chg_cols += ["prec_chg_%s" % m, "ng_chg_%s" % m]

    # ---- ensemble mean of the fields (raw mm), then the ensemble change ----
    # For the mean figure: average hist and future first, then derive one
    # change from those two means. This is NOT the average of the per-member
    # changes above (those two quantities differ whenever members have
    # different baselines) and NOT diffing members against this mean hist.
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
    df["prec_fut_mean_in"] = prec_fut_mm * MM_TO_IN
    df["ng_hist_mean_in"] = ng_hist_mm * MM_TO_IN
    df["ng_fut_mean_in"] = ng_fut_mm * MM_TO_IN

    # per-member hist magnitude, in inches, for panel annotations
    hist_in_cols = []
    for m in members:
        df["prec_hist_in_%s" % m] = df["prec_hist_%s" % m] * MM_TO_IN
        df["ng_hist_in_%s" % m] = df["ng_hist_%s" % m] * MM_TO_IN
        hist_in_cols += ["prec_hist_in_%s" % m, "ng_hist_in_%s" % m]

    out = df[["lat", "lon",
              "prec_hist_mean_in", "prec_fut_mean_in",
              "ng_hist_mean_in", "ng_fut_mean_in",
              "prec_chg", "ng_chg"] + chg_cols + hist_in_cols].copy()

    print("ensemble cells after inner merge: %d" % len(out))
    print("domain-mean change (ensemble-mean based): PREC-IDF %+.2f %%, "
          "NG-IDF %+.2f %%"
          % (np.nanmean(out["prec_chg"].values),
             np.nanmean(out["ng_chg"].values)))
    for m in members:
        print("  member %s own-baseline change: PREC-IDF %+.2f %%, "
              "NG-IDF %+.2f %%"
              % (m, np.nanmean(out["prec_chg_%s" % m].values),
                 np.nanmean(out["ng_chg_%s" % m].values)))
    return out


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def generate_cesm_idf_figure(base_dir, land_cover, duration, ari,
                             fig_file=None, lulc_file=None,
                             members=MEMBERS,
                             vmax_fixed=None, dlim_fixed=None,
                             save_pdf=True):
    """
    Generate the 2x3 CESM ensemble-mean IDF figure.

    Parameters
    ----------
    base_dir : str
        Directory holding the CESM_LE*_Mid_Century folders.
    land_cover : str
        "open" | "evergreen" | "deciduous" (used for the default filename).
    duration : str or int
        Storm duration in hours, e.g. "24".
    ari : str or int
        Average recurrence interval in years, e.g. "25".
    vmax_fixed : float or None
        Upper limit of the magnitude colorbar, in inches. None autoscales to
        the 98th percentile of this figure. IDF magnitude scales strongly with
        duration, so a value pinned for 24h will not suit 1h.
    dlim_fixed : float or None
        Symmetric limit of the change colorbar, in percent. None autoscales to
        the 98th percentile of |change|. Percent change is dimensionless, so a
        single value can safely be shared across durations and ARIs.
    save_pdf : bool
        Also write a vector PDF next to the PNG.

    Returns
    -------
    str
        Base64-encoded PNG prefixed with "data:image/png;base64,".
    """
    df = load_cesm_ensemble(base_dir, duration, ari, members=members)

    # ---- shared color limits ----
    mag_vals = np.concatenate([df["prec_hist_mean_in"].values,
                               df["ng_hist_mean_in"].values])
    vmax = float(vmax_fixed) if vmax_fixed is not None \
        else float(np.nanpercentile(mag_vals, 98))

    chg_vals = np.concatenate([df["prec_chg"].values, df["ng_chg"].values])
    dlim = float(dlim_fixed) if dlim_fixed is not None \
        else float(np.nanpercentile(np.abs(chg_vals), 98))

    data_crs = ccrs.PlateCarree()

    # ---- interpolate onto a regular grid ----
    pad = 0.05
    nx, ny = 400, 400
    grid_lon = np.linspace(df["lon"].min() + pad, df["lon"].max() - pad, nx)
    grid_lat = np.linspace(df["lat"].min() + pad, df["lat"].max() - pad, ny)
    GX, GY = np.meshgrid(grid_lon, grid_lat)
    pts = df[["lon", "lat"]].values

    map_cols = ["prec_hist_mean_in", "ng_hist_mean_in", "prec_chg", "ng_chg"]
    fields = {c: griddata(pts, df[c].values, (GX, GY), method="linear")
              for c in map_cols}

    extent = [df["lon"].min(), df["lon"].max(),
              df["lat"].min(), df["lat"].max()]
    ext_pad = [extent[0] - 0.1, extent[1] + 0.1,
               extent[2] - 0.1, extent[3] + 0.1]

    # ---- 2x3 layout ----
    fig = plt.figure(figsize=(13.5, 8.4), layout="compressed")
    gs = fig.add_gridspec(2, 3)
    ax_lulc = fig.add_subplot(gs[0, 0], projection=PROJ)
    ax_ph = fig.add_subplot(gs[0, 1], projection=PROJ)
    ax_nh = fig.add_subplot(gs[0, 2], projection=PROJ)
    ax_txt = fig.add_subplot(gs[1, 0])
    ax_pc = fig.add_subplot(gs[1, 1], projection=PROJ)
    ax_nc = fig.add_subplot(gs[1, 2], projection=PROJ)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01,
                                wspace=0.02, hspace=0.02)

    # ===== Panel (a): land cover =====
    ax_lulc.set_extent(ext_pad, crs=data_crs)
    ax_lulc.add_image(TILER, ZOOM)

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
        lulc_mesh = ax_lulc.pcolormesh(lc.x.values, lc.y.values, lc.values,
                                       cmap=cmap_lc, norm=norm_lc,
                                       transform=data_crs, alpha=0.75,
                                       shading="auto", zorder=2)
        handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=n)
                   for c, n in zip(colors, names)]
        ax_lulc.legend(handles=handles, loc="lower left", fontsize=8,
                       framealpha=0.85)

    ax_lulc.plot([extent[0], extent[1], extent[1], extent[0], extent[0]],
                 [extent[2], extent[2], extent[3], extent[3], extent[2]],
                 color="blue", linewidth=1.8, transform=data_crs, zorder=5,
                 label="Study domain")
    add_sites(ax_lulc, extent, data_crs, fontsize=5, star=12)
    add_grid(ax_lulc)
    ax_lulc.set_title("(a) Land Cover / Study Domain", fontsize=9)
    if lulc_mesh is None:
        ax_lulc.legend(loc="lower left", fontsize=9, framealpha=0.85)

    # ===== map panels =====
    def draw(ax, col, title, cmap, vmin_, vmax_):
        ax.set_extent(ext_pad, crs=data_crs)
        ax.add_image(TILER, ZOOM)
        mesh = ax.pcolormesh(GX, GY, fields[col], cmap=cmap,
                             vmin=vmin_, vmax=vmax_, transform=data_crs,
                             alpha=BASEMAP_ALPHA, shading="auto", zorder=3,
                             rasterized=True)
        add_sites(ax, extent, data_crs)
        add_grid(ax)
        ax.set_title(title, fontsize=9)
        return mesh

    mag_mesh = draw(ax_ph, "prec_hist_mean_in",
                    "(b) PREC-IDF Historical (ensemble mean)",
                    "viridis", 0.0, vmax)
    draw(ax_nh, "ng_hist_mean_in",
         "(c) NG-IDF Historical (ensemble mean)", "viridis", 0.0, vmax)

    chg_mesh = draw(ax_pc, "prec_chg",
                    "(d) PREC-IDF Change (ensemble mean)", "RdBu_r", -dlim, dlim)
    draw(ax_nc, "ng_chg",
         "(e) NG-IDF Change (ensemble mean)", "RdBu_r", -dlim, dlim)

    # ===== blank cell: ensemble member names =====
    ax_txt.set_axis_off()
    ax_txt.text(0.5, 0.66, "Mid-Century Ensemble Mean",
                ha="center", va="center", fontsize=9, weight="bold",
                transform=ax_txt.transAxes)
    ax_txt.text(0.5, 0.52, "Average of 4 model simulations",
                ha="center", va="center", fontsize=9,
                transform=ax_txt.transAxes)
    ax_txt.text(0.5, 0.40,
                "Members 1-4: CESM-LE2, LE4, LE7, LE9",
                ha="center", va="center", fontsize=9,
                transform=ax_txt.transAxes)

    # ===== shared colorbars =====
    cb1 = fig.colorbar(mag_mesh, ax=[ax_ph, ax_nh], orientation="vertical",
                       fraction=0.046, pad=0.01, aspect=20)
    cb1.set_label("%sh %s-yr Magnitude (in)" % (duration, ari), fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    cb1.solids.set_alpha(1.0)

    cb2 = fig.colorbar(chg_mesh, ax=[ax_pc, ax_nc], orientation="vertical",
                       fraction=0.046, pad=0.01, aspect=20)
    cb2.set_label("Change (%)", fontsize=9)
    cb2.ax.tick_params(labelsize=8)
    cb2.solids.set_alpha(1.0)

    # invisible colorbars on column 0 so all three columns share a width
    for ax_blank in (ax_lulc, ax_txt):
        cbx = fig.colorbar(mag_mesh, ax=ax_blank, orientation="vertical",
                           fraction=0.046, pad=0.01, aspect=20)
        cbx.outline.set_visible(False)
        cbx.ax.set_facecolor("none")
        cbx.ax.tick_params(size=0, labelsize=0, colors="none")
        for s in cbx.ax.spines.values():
            s.set_visible(False)
        cbx.solids.set_alpha(0.0)

    # ---- save ----
    if fig_file is None:
        fig_file = "./cesm_idf_ensmean_%s_%sh_%syr.png" % (
            land_cover, duration, ari)
    fig_file = os.path.abspath(fig_file)
    d = os.path.dirname(fig_file)
    if d:
        os.makedirs(d, exist_ok=True)

    plt.savefig(fig_file, dpi=300, bbox_inches="tight", pad_inches=0.02)
    if save_pdf:
        plt.savefig(os.path.splitext(fig_file)[0] + ".pdf",
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved:", fig_file, os.path.exists(fig_file))

    with open(fig_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return "data:image/png;base64," + encoded


# ---------------------------------------------------------------------------
# companion figure: individual member change (2 x 4, basemap + sites)
# ---------------------------------------------------------------------------
def generate_cesm_member_figure(base_dir, land_cover, duration, ari,
                                fig_file=None, members=MEMBERS,
                                dlim_fixed=None, save_pdf=True):
    """
    Generate the 2x4 individual-member change figure: rows are PREC-IDF and
    NG-IDF, columns are the four members. Each member's change is computed
    from ITS OWN hist and future (see load_cesm_ensemble), never from the
    ensemble-mean hist.

    Panels use the same projection, basemap tiles, and site markers as
    generate_cesm_idf_figure (see the module-level PROJ/TILER/add_sites/
    add_grid helpers), so panel shape and geographic content match exactly.

    Pass dlim_fixed = the same dlim used for panels (d)/(e) of the mean
    figure so the two figures are read on one consistent color scale.
    """
    df = load_cesm_ensemble(base_dir, duration, ari, members=members)

    prec_chg_cols = ["prec_chg_%s" % m for m in members]
    ng_chg_cols = ["ng_chg_%s" % m for m in members]

    if dlim_fixed is not None:
        dlim = float(dlim_fixed)
    else:
        all_chg = df[prec_chg_cols + ng_chg_cols].values
        dlim = float(np.nanpercentile(np.abs(all_chg), 98))

    data_crs = ccrs.PlateCarree()

    # ---- interpolate each member's own change onto a regular grid ----
    pad = 0.05
    nx, ny = 300, 300
    grid_lon = np.linspace(df["lon"].min() + pad, df["lon"].max() - pad, nx)
    grid_lat = np.linspace(df["lat"].min() + pad, df["lat"].max() - pad, ny)
    GX, GY = np.meshgrid(grid_lon, grid_lat)
    pts = df[["lon", "lat"]].values

    fields = {c: griddata(pts, df[c].values, (GX, GY), method="linear")
              for c in prec_chg_cols + ng_chg_cols}

    extent = [df["lon"].min(), df["lon"].max(),
              df["lat"].min(), df["lat"].max()]
    ext_pad = [extent[0] - 0.1, extent[1] + 0.1,
               extent[2] - 0.1, extent[3] + 0.1]

    ncols = len(members)
    fig = plt.figure(figsize=(4.5 * ncols, 10.0), layout="compressed")
    gs = fig.add_gridspec(2, ncols)
    axes = [[fig.add_subplot(gs[i, j], projection=PROJ) for j in range(ncols)]
            for i in range(2)]
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02,
                                wspace=0.02, hspace=0.02)

    rows = (("PREC-IDF", prec_chg_cols), ("NG-IDF", ng_chg_cols))
    letters = "abcdefgh"
    mesh = None
    for j, m in enumerate(members):
        for i, (var, cols) in enumerate(rows):
            ax = axes[i][j]
            col = cols[j]
            ax.set_extent(ext_pad, crs=data_crs)
            ax.add_image(TILER, ZOOM)
            mesh = ax.pcolormesh(GX, GY, fields[col], cmap="RdBu_r",
                                 vmin=-dlim, vmax=dlim, transform=data_crs,
                                 alpha=BASEMAP_ALPHA, shading="auto",
                                 zorder=3, rasterized=True)
            add_sites(ax, extent, data_crs)
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

    cb = fig.colorbar(mesh, ax=[a for row in axes for a in row],
                      orientation="horizontal", location="bottom",
                      fraction=0.035, pad=0.04, aspect=40, shrink=0.6)
    cb.set_label("Change (%)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.solids.set_alpha(1.0)

    fig.suptitle("%sh, %s-yr return level: individual member change"
                 % (duration, ari), fontsize=11)

    if fig_file is None:
        fig_file = "./cesm_idf_members_%s_%sh_%syr.png" % (
            land_cover, duration, ari)
    fig_file = os.path.abspath(fig_file)
    d = os.path.dirname(fig_file)
    if d:
        os.makedirs(d, exist_ok=True)

    plt.savefig(fig_file, dpi=300, bbox_inches="tight", pad_inches=0.02)
    if save_pdf:
        plt.savefig(os.path.splitext(fig_file)[0] + ".pdf",
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved:", fig_file, os.path.exists(fig_file))

    with open(fig_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return "data:image/png;base64," + encoded


if __name__ == "__main__":
    LAND_COVER = "open"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LULC_FILE = None

    DURATION = 24
    ARI = 25

    # None = autoscale to the mean figure's own 98th percentile. Pin these
    # only when you want several figures/durations to share one colorbar.
    VMAX = None
    DLIM = None

    for m in MEMBERS:
        p = os.path.join(BASE_DIR, "CESM_%s_Mid_Century" % m)
        if not os.path.isdir(p):
            raise SystemExit("missing member folder: %s" % os.path.abspath(p))

    # 1) ensemble-mean figure -- also returns the dlim it used, so the
    #    member figure below can be forced onto the identical color scale
    df_check = load_cesm_ensemble(BASE_DIR, DURATION, ARI)
    if DLIM is None:
        chg_vals = np.concatenate([df_check["prec_chg"].values,
                                   df_check["ng_chg"].values])
        DLIM = float(np.nanpercentile(np.abs(chg_vals), 98))

    generate_cesm_idf_figure(
        BASE_DIR, LAND_COVER, DURATION, ARI,
        fig_file="./cesm_idf_ensmean_%s_%sh_%syr.png"
                 % (LAND_COVER, DURATION, ARI),
        lulc_file=LULC_FILE,
        vmax_fixed=VMAX, dlim_fixed=DLIM)

    # 2) individual-member change figure, forced onto the same DLIM
    generate_cesm_member_figure(
        BASE_DIR, LAND_COVER, DURATION, ARI,
        fig_file="./cesm_idf_members_%s_%sh_%syr.png"
                 % (LAND_COVER, DURATION, ARI),
        dlim_fixed=DLIM)