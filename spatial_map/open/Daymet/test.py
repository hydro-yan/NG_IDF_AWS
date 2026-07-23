import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from scipy.interpolate import griddata

# ---- Load data ----
df = pd.read_csv("ng_idf_return_24h_2yr.csv")
df["diff_pct"] = (df["ng_idf_hist"] - df["prec_idf_hist"]) / df["prec_idf_hist"] * 100.0

shared_vmax = np.nanpercentile(
    np.concatenate([df["prec_idf_hist"].values, df["ng_idf_hist"].values]), 98)
dlim = np.nanpercentile(np.abs(df["diff_pct"].values), 98)

panels = [
    ("prec_idf_hist", "(b) PREC-IDF", "viridis", 0, shared_vmax, "Magnitude (mm)"),
    ("ng_idf_hist",   "(c) NG-IDF",   "viridis", 0, shared_vmax, "Magnitude (mm)"),
    ("diff_pct",      "(d) Difference (NG - PREC)", "RdBu_r", -dlim, dlim, "Difference (%)"),
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

def add_sites(ax, fontsize=9, star=12):
    for name, clon, clat in sites:
        if extent[0] <= clon <= extent[1] and extent[2] <= clat <= extent[3]:
            ax.plot(clon, clat, marker="*", color="red", markersize=star,
                    markeredgecolor="white", markeredgewidth=0.9,
                    transform=data_crs, zorder=6)
            ax.text(clon+0.05, clat+0.05, name, fontsize=fontsize, color="black",
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
fig, axes = plt.subplots(2, 2, figsize=(17, 15),
                         subplot_kw={"projection": proj},
                         constrained_layout=True)
axf = axes.flatten()

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

add_sites(ax, fontsize=10, star=14)
add_grid(ax)
ax.set_title("(a) Study Area: Interior Alaska", fontsize=14)
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
    ax.set_title(title, fontsize=14)

    cb = fig.colorbar(mesh, ax=ax, orientation="vertical",
                      fraction=0.046, pad=0.03)
    cb.set_label(clabel, fontsize=11)
    cb.solids.set_alpha(1.0)

# ---- invisible colorbar on (a) so it aligns with (c) ----
cb_a = fig.colorbar(mesh_ref, ax=axf[0], orientation="vertical",
                    fraction=0.046, pad=0.03)
cb_a.ax.set_visible(False)

fig.suptitle("24h 2-yr Return Level: PREC-IDF vs NG-IDF, Interior Alaska",
             fontsize=16)

plt.savefig("ng_idf_spatial_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: ng_idf_spatial_comparison.png")