"""
Standalone test script: CESM mid-century Active Layer Thickness (ALT) figure.

Layout (2 x 4):

  Row 1: (a) Land cover / study domain
         (b) Ensemble-mean historical ALT (ft)
         (c) Ensemble-mean mid-century ALT (ft)
         (d) Ensemble-mean change (ft)

  Row 2: (e)-(h) Individual member change (ft), each against its OWN baseline

(b)(c) share one magnitude colorbar; (d) and the whole second row share one
change colorbar, so the ensemble panel and the member panels are directly
comparable.

Data layout expected:

  ./spatial_map/{land_cover}/CESM_{member}_Mid_Century/ng_idf_ald_results.csv

with columns: lat, lon, mean_ALD_hist, mean_ALD_future
(the future column name is auto-detected from a few common variants).

Basemap setup (tiler, ZOOM, ALPHA, ext_pad) matches the WRF / Daymet / CESM IDF
figures so panels line up across the paper.

Run:
    python3 plot_cesm_mid_century_alt.py --land-cover open
    python3 plot_cesm_mid_century_alt.py --land-cover evergreen --dpi 300 --save-pdf
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from scipy.interpolate import griddata
import base64


M_TO_FT = 3.28084
LATLON_DEC = 5          # rounding used to build the merge key

SITES = [
    ("Fort Wainwright", -147.6389, 64.8283),
    ("North Pole",      -147.3494, 64.7511),
    ("Delta Junction",  -145.7336, 64.0378),
]

FUTURE_CANDIDATES = ["mean_ALD_future", "mean_ALD_future_mid_century",
                     "mean_ALD_fut", "mean_ALD_future_medium"]


def _resolve_base_dir(base_dir, land_cover, members):
    """
    Find the directory that holds the CESM_{member}_Mid_Century folders.

    Works whether the script is run from the NG_IDF_AWS root, from inside
    spatial_map/, or from inside spatial_map/{land_cover}/.
    """
    probe = "CESM_%s_Mid_Century" % members[0]
    if base_dir is not None:
        cands = [base_dir]
    else:
        cands = [os.path.join(".", "spatial_map", land_cover),
                 ".",
                 os.path.join(".", land_cover),
                 os.path.join("..", land_cover),
                 os.path.join("..", "spatial_map", land_cover)]
    for c in cands:
        if os.path.isdir(os.path.join(c, probe)):
            return c
    raise FileNotFoundError(
        "Could not locate '%s' under any of: %s (cwd: %s). "
        "Pass --base-dir explicitly."
        % (probe, [os.path.abspath(c) for c in cands], os.getcwd()))


def _find_future_col(cols, path):
    for c in FUTURE_CANDIDATES:
        if c in cols:
            return c
    raise ValueError("No future ALD column in %s; looked for %s; found: %s"
                     % (path, FUTURE_CANDIDATES, list(cols)))


def generate_cesm_mid_century_alt_figure(land_cover, spatial_scenario="cesm_mid_century",
                                         duration=None, ari=None,
                                         fig_file=None, lulc_file=None,
                                         members=("LE2", "LE4", "LE7", "LE9"),
                                         base_dir=None,
                                         vmin_mag_fixed=None, vmax_mag_fixed=None,
                                         chg_lim_fixed=None,
                                         dpi=150, save_pdf=False,
                                         require_all_members=True):
    """
    Build the 2x4 CESM mid-century ALT figure.

    Parameters
    ----------
    land_cover : str
        "open" | "evergreen" | "deciduous"
    spatial_scenario : str
        Kept for signature consistency with the IDF figure functions.
    duration, ari : ignored
        Kept so the call signature matches generate_cesm_mid_century_idf_figure.
    fig_file : str or None
        Output PNG path. Defaults to "./cesm_alt_2x4_{land_cover}.png".
    lulc_file : str or None
        Optional land cover raster for panel (a).
    members : sequence of str
        CESM ensemble member codes. Four members give a clean 2x4; more members
        widen the second row and leave row 1 padded on the right.
    base_dir : str or None
        Directory containing the CESM_{member}_Mid_Century folders. None
        auto-detects, so the script runs from the NG_IDF_AWS root or from
        inside spatial_map/{land_cover}/ without edits.
    vmin_mag_fixed, vmax_mag_fixed : float or None
        Magnitude colorbar limits in ft. None autoscales to the 2nd/98th
        percentile of the pooled hist + future ensemble means.
    chg_lim_fixed : float or None
        Change colorbar limit in ft. None autoscales to the 98th percentile of
        |change| pooled over the ensemble mean and all members.
    require_all_members : bool
        If True, a cell must be valid in every member to enter the ensemble
        mean. Prevents the mean from shifting where members drop out.

    Returns
    -------
    str
        Base64 PNG data URI.
    """

    members = list(members)
    base_dir = _resolve_base_dir(base_dir, land_cover, members)
    print("base dir   :", os.path.abspath(base_dir))

    # ---- Load: one CSV per member ----
    frames = []
    for m in members:
        path = os.path.join(base_dir, "CESM_%s_Mid_Century" % m,
                            "ng_idf_ald_results.csv")
        if not os.path.exists(path):
            raise FileNotFoundError("CSV not found: %s" % os.path.abspath(path))

        d = pd.read_csv(path)
        for c in ("lat", "lon", "mean_ALD_hist"):
            if c not in d.columns:
                raise ValueError("%s is missing column '%s'; found: %s"
                                 % (path, c, list(d.columns)))
        fut_col = _find_future_col(d.columns, path)

        d = d.dropna(subset=["lat", "lon"])
        d["lat"] = d["lat"].round(LATLON_DEC)
        d["lon"] = d["lon"].round(LATLON_DEC)
        d = d.drop_duplicates(subset=["lat", "lon"])

        # ALD == 0 means no thaw / masked cell; exclude so it does not bias
        # the color scales. Applied only to the ALD columns, never lat/lon,
        # so the domain extent stays identical to the IDF figures.
        d = d.rename(columns={"mean_ALD_hist": "alt_hist_%s" % m,
                              fut_col: "alt_fut_%s" % m})
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

    # ---- ensemble means, then the ensemble change ----
    skipna = not require_all_members
    df["alt_hist_mean"] = df[["alt_hist_%s" % m for m in members]].mean(
        axis=1, skipna=skipna)
    df["alt_fut_mean"] = df[["alt_fut_%s" % m for m in members]].mean(
        axis=1, skipna=skipna)
    df["alt_chg_mean"] = df["alt_fut_mean"] - df["alt_hist_mean"]

    print("ensemble cells after inner merge: %d" % len(df))
    print("valid ensemble-mean cells: %d" % int(np.isfinite(df["alt_chg_mean"]).sum()))
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
    mag_vals = np.concatenate([df["alt_hist_mean"].values, df["alt_fut_mean"].values])
    mag_vals = mag_vals[np.isfinite(mag_vals)]
    vmin_mag = float(vmin_mag_fixed) if vmin_mag_fixed is not None \
        else float(np.nanpercentile(mag_vals, 2))
    vmax_mag = float(vmax_mag_fixed) if vmax_mag_fixed is not None \
        else float(np.nanpercentile(mag_vals, 98))

    chg_vals = np.concatenate([df["alt_chg_mean"].values] +
                              [df[c].values for c in chg_cols])
    chg_vals = chg_vals[np.isfinite(chg_vals)]
    if chg_lim_fixed is not None:
        clim = float(chg_lim_fixed)
        chg_cmap, chg_vmin, chg_vmax = "RdBu_r", -clim, clim
    elif np.nanmin(chg_vals) < 0:
        clim = float(np.nanpercentile(np.abs(chg_vals), 98))
        chg_cmap, chg_vmin, chg_vmax = "RdBu_r", -clim, clim
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
            raise ValueError("Column %s has only %d valid cells; cannot interpolate."
                             % (c, int(ok.sum())))
        fields[c] = griddata(df.loc[ok, ["lon", "lat"]].values,
                             df.loc[ok, c].values, (GX, GY), method="linear")

    # ---- basemap (same tiler / ZOOM / ALPHA as the IDF figures) ----
    tiler = cimgt.OSM()
    proj = tiler.crs
    ZOOM = 9
    ALPHA = 0.55

    extent = [df["lon"].min(), df["lon"].max(), df["lat"].min(), df["lat"].max()]
    ext_pad = [extent[0] - 0.1, extent[1] + 0.1, extent[2] - 0.1, extent[3] + 0.1]
    print("ALT extent :", [round(v, 4) for v in extent])

    def add_sites(ax, fontsize=5, star=10):
        for name, clon, clat in SITES:
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

    # ================= 2 x N layout (N = number of members, 4 -> 2x4) =========
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

    # ===== Row 1 =====
    add_lulc(ax_lulc)
    mag_mesh = draw(ax_h, "alt_hist_mean",
                    "(b) Historical ALT (ensemble mean)",
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
             "(%s) Member %d (CESM-%s) ALT Change" % (letters[4 + j], j + 1, m),
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

    # keep column 0 the same width as the map columns
    blank_colorbar(fig, mag_mesh, ax_lulc)

    # ---- save ----
    if fig_file is None:
        fig_file = "./cesm_alt_2x4_%s.png" % land_cover
    fig_file = os.path.abspath(fig_file)
    d = os.path.dirname(fig_file)
    if d:
        os.makedirs(d, exist_ok=True)

    fig.savefig(fig_file, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    if save_pdf:
        fig.savefig(os.path.splitext(fig_file)[0] + ".pdf",
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved:", fig_file, os.path.exists(fig_file))

    with open(fig_file, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode("utf-8")


def main():
    p = argparse.ArgumentParser(description="CESM mid-century ALT 2x4 figure")
    p.add_argument("--land-cover", default="open",
                   choices=["open", "evergreen", "deciduous"])
    p.add_argument("--members", nargs="+", default=["LE2", "LE4", "LE7", "LE9"])
    p.add_argument("--base-dir", default=None,
                   help="Directory holding CESM_{member}_Mid_Century folders. "
                        "Auto-detected if omitted.")
    p.add_argument("--fig-file", default=None)
    p.add_argument("--lulc-file", default=None)
    p.add_argument("--vmin-mag", type=float, default=None)
    p.add_argument("--vmax-mag", type=float, default=None)
    p.add_argument("--chg-lim", type=float, default=None)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--save-pdf", action="store_true")
    p.add_argument("--allow-partial-members", action="store_true",
                   help="Let the ensemble mean use whatever members are valid "
                        "in a cell instead of requiring all of them.")
    a = p.parse_args()

    generate_cesm_mid_century_alt_figure(
        land_cover=a.land_cover,
        members=tuple(a.members),
        base_dir=a.base_dir,
        fig_file=a.fig_file,
        lulc_file=a.lulc_file,
        vmin_mag_fixed=a.vmin_mag,
        vmax_mag_fixed=a.vmax_mag,
        chg_lim_fixed=a.chg_lim,
        dpi=a.dpi,
        save_pdf=a.save_pdf,
        require_all_members=not a.allow_partial_members,
    )


if __name__ == "__main__":
    main()