"""Secondary metrics for the world gradient field: climate (NASA POWER), deforestation (OWID) and wildfire (NASA EONET).

Every metric is put onto the same units as the BSF estimates (world, World Bank regions, countries, GADM admin-1, in
the feature order of dashboard/gpu/data/world/*.geojson), so the browser runs it through the same GPU join / blur /
zoom blend and can colour it against BSF as a bivariate (BSF x metric) or trivariate (BSF x temperature x humidity) map.

  temp       NASA POWER T2M, annual climatology (C)                     area mean of 0.5 deg grid cells
  humidity   NASA POWER RH2M, annual climatology (%)                    area mean
  precip     NASA POWER PRECTOTCORR, annual climatology (mm/yr)         area mean
  forest_loss  OWID tree-cover loss, % of land area per year, per preset window      country level; admin-1 inherits
  fires      NASA EONET wildfire events per 10,000 km2, per preset window (EONET starts in 2016)   admin-1 and up

Climate source: NASA POWER when all its boxes can be fetched (or are cached); otherwise, since POWER throttles this
client hard at times, ERA5 1991-2020 normals from the World Bank Climate Change Knowledge Portal at country level
(temperature, relative humidity) plus WorldClim 2.1 normals at admin-1 (temperature, precipitation; admin-1 humidity
then inherits its country). metrics.json records which source each metric came from.

NASA POWER is pulled with the regional climatology endpoint, one 10 x 10 degree box and one parameter per call, only
for boxes that touch land. It resets fresh TLS connections often, so one keep-alive session is reused (about 0.4 s
per call), failures are retried, and every box is cached.

Reads  dashboard/gpu/data/world/{nation,regions,states,counties}.geojson  (python bsfl_world_field.py)
       data/raw/owid_tree-cover-loss.csv, data/raw/eonet_wildfires.csv
Writes data/raw/power/<param>_<lat>_<lon>.json (cache), data/processed/power_climatology_grid.csv,
       dashboard/gpu/data/world/metrics.json
"""

import json
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

from bsfl_sources import PROCESSED, RAW
from bsfl_world_field import OUT, WINDOWS

POWER = RAW / "power"
POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/regional"
PARAMS = {"T2M": "temp", "RH2M": "humidity", "PRECTOTCORR": "precip"}
LAT_RANGE, STEP = (-60, 80), 10
GRID_DEG = 0.5
EQUAL_AREA = "EPSG:6933"
TIERS = ["nation", "regions", "states", "counties"]
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "BSFL-Mapping/0.1 (research)"


# ---------------------------------------------------------------- NASA POWER
def power_box(param: str, lat: int, lon: int, tries=7) -> list:
    path = POWER / f"{param}_{lat}_{lon}.json"
    if path.exists():
        return json.loads(path.read_text())
    params = {"parameters": param, "community": "AG", "latitude-min": lat, "latitude-max": lat + STEP,
              "longitude-min": lon, "longitude-max": lon + STEP, "format": "JSON"}
    for attempt in range(tries):
        try:
            r = SESSION.get(POWER_URL, params=params, timeout=120)
            if r.status_code == 200:
                pts = [(f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1], f["properties"]["parameter"][param]["ANN"])
                       for f in r.json().get("features", [])]
                path.write_text(json.dumps(pts))
                return pts
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"POWER {r.status_code}: {r.text[:200]}")
        except requests.RequestException:
            pass
        time.sleep(min(60, 2 * 2 ** attempt))
    raise RuntimeError(f"POWER gave up on {param} {lat} {lon}")


CCKP = "https://cckpapi.worldbank.org/cckp/v1/era5-x0.25_climatology_{var}_climatology_annual_1991-2020_mean_historical_era5_x0.25_mean/{codes}"


def cckp_era5(var: str, iso3: list[str]) -> pd.Series:
    path = RAW / f"cckp_era5_{var}_1991-2020.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)[var]
    out = {}
    for i in range(0, len(iso3), 40):
        r = SESSION.get(CCKP.format(var=var, codes=",".join(iso3[i:i + 40])), params={"_format": "json"}, timeout=120)
        r.raise_for_status()
        data = r.json()["data"]
        out |= {code: next(iter(v.values())) for code, v in (data.items() if isinstance(data, dict) else []) if v}
    s = pd.Series(out, name=var)
    s.to_frame().to_csv(path)
    return s


def climate_fallback(tiers, area_km2) -> tuple[dict, str]:
    """Country: ERA5 (CCKP). Admin-1: WorldClim temperature / precipitation, humidity inherited from the country.
    Regions and world: area-weighted means of countries."""
    st, co = tiers["states"], tiers["counties"]
    iso = list(st["ab"])
    era = {"temp": cckp_era5("tas", iso), "humidity": cckp_era5("hurs", iso), "precip": cckp_era5("pr", iso)}
    units = pd.read_csv(PROCESSED / "admin1_units.csv").set_index("gid1")
    clim = {k: {} for k in TIERS}
    for key, s in era.items():
        c = st["ab"].map(s).to_numpy(float)
        if key == "humidity":
            a = c[co["st"].to_numpy()]
        else:
            a = co["gid"].map(units["tas" if key == "temp" else "pr"]).to_numpy(float)
            a = np.where(np.isnan(a), c[co["st"].to_numpy()], a)
        rg, w = st["rg"].to_numpy(), np.where(np.isnan(c), 0, area_km2["states"])
        reg = np.bincount(rg, weights=np.nan_to_num(c) * w, minlength=len(tiers["regions"])) / np.maximum(np.bincount(rg, weights=w, minlength=len(tiers["regions"])), 1e-9)
        clim["states"][key], clim["counties"][key], clim["regions"][key] = c, a, reg
        clim["nation"][key] = np.array([np.nansum(c * w) / w.sum()])
    return clim, "ERA5 1991-2020 normals (World Bank CCKP) by country; WorldClim 2.1 by admin-1"


def power_grid(land) -> pd.DataFrame:
    if (PROCESSED / "power_climatology_grid.csv").exists():      # already assembled from the cached boxes
        return pd.read_csv(PROCESSED / "power_climatology_grid.csv")
    POWER.mkdir(parents=True, exist_ok=True)
    boxes = [(lat, lon) for lat in range(*LAT_RANGE, STEP) for lon in range(-180, 180, STEP)
             if land.intersects(box(lon, lat, lon + STEP, lat + STEP))]
    todo = sum(not (POWER / f"{p}_{la}_{lo}.json").exists() for p in PARAMS for la, lo in boxes)
    print(f"NASA POWER: {len(boxes)} land boxes x {len(PARAMS)} parameters, {todo} calls to make", flush=True)
    frames, done = [], 0
    for p, key in PARAMS.items():
        rows = []
        for la, lo in boxes:
            cached = (POWER / f"{p}_{la}_{lo}.json").exists()
            rows += power_box(p, la, lo, tries=7 if cached or done else 3)
            if not cached:
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{todo}", flush=True)
                time.sleep(0.3)
        frames.append(pd.DataFrame(rows, columns=["lon", "lat", key]).groupby(["lon", "lat"]).mean())
    g = pd.concat(frames, axis=1).reset_index()
    g["precip"] = g["precip"] * 365.25          # mm/day -> mm/yr
    g.to_csv(PROCESSED / "power_climatology_grid.csv", index=False)
    return g


def to_raster(g: pd.DataFrame, col: str):
    """Scatter the POWER points (0.5 x 0.625 deg) onto a regular 0.5 deg lat/lon grid by nearest cell."""
    W, H = int(360 / GRID_DEG), int((LAT_RANGE[1] - LAT_RANGE[0]) / GRID_DEG)
    arr = np.full((H, W), np.nan)
    i = ((LAT_RANGE[1] - g["lat"]) / GRID_DEG).astype(int).clip(0, H - 1)
    j = ((g["lon"] + 180) / GRID_DEG).astype(int).clip(0, W - 1)
    arr[i, j] = g[col]
    # fill the 0.625-deg column gaps from the nearest filled neighbour along the row
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(np.isnan(arr), return_distances=False, return_indices=True)
    return arr[tuple(idx)], from_origin(-180, LAT_RANGE[1], GRID_DEG, GRID_DEG)


def zonal_means(geoms, rasters: dict, transform, shape) -> dict:
    """Mean of grid cells whose centres fall inside each polygon; tiny units take the cell under their centroid."""
    ids = rasterize(((g, i + 1) for i, g in enumerate(geoms) if g is not None and not g.is_empty), out_shape=shape,
                    transform=transform, fill=0, dtype="int32")
    n = len(geoms)
    counts = np.bincount(ids.ravel(), minlength=n + 1)[1:]
    out = {}
    for key, arr in rasters.items():
        sums = np.bincount(ids.ravel(), weights=np.nan_to_num(arr.ravel()), minlength=n + 1)[1:]
        vals = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        for k in np.where(counts == 0)[0]:
            c = geoms[k].representative_point()
            r, cidx = int((LAT_RANGE[1] - c.y) / GRID_DEG), int((c.x + 180) / GRID_DEG)
            if 0 <= r < shape[0] and 0 <= cidx < shape[1]:
                vals[k] = arr[r, cidx]
        out[key] = vals
    return out


# ---------------------------------------------------------------- assembly
def main():
    tiers = {k: gpd.read_file(OUT / f"{k}.geojson").sort_values("id").reset_index(drop=True) for k in TIERS}
    for g in tiers.values():
        g.set_crs(4326, inplace=True, allow_override=True)
    area_km2 = {k: g.to_crs(EQUAL_AREA).area.to_numpy() / 1e6 for k, g in tiers.items()}
    land = tiers["nation"].geometry.iloc[0]

    # climate: NASA POWER if reachable, else ERA5 / WorldClim
    try:
        grid = power_grid(land)
        rasters = {}
        for key in PARAMS.values():
            rasters[key], tr = to_raster(grid, key)
        shape = next(iter(rasters.values())).shape
        clim = {k: zonal_means(list(g.geometry), rasters, tr, shape) for k, g in tiers.items()}
        clim_src = "NASA POWER climatology (annual), 0.5 deg area means"
    except RuntimeError as e:
        print(f"NASA POWER unavailable ({e}); using ERA5 / WorldClim normals", flush=True)
        clim, clim_src = climate_fallback(tiers, area_km2)

    # deforestation: country-level tree-cover loss as % of land area per year, per preset window
    st = tiers["states"]
    loss = pd.read_csv(RAW / "owid_tree-cover-loss.csv")
    col = [c for c in loss.columns if c not in ("iso3", "year")][0]
    forest = []
    for a, b in WINDOWS:
        w = loss[loss["year"].between(a, b)].groupby("iso3")[col].mean()          # ha per year
        ha = st["ab"].map(w).to_numpy()
        forest.append(ha / 100 / area_km2["states"] * 100)                          # km2/yr over land km2, as %
    c_of_a = tiers["counties"]["st"].to_numpy()

    def roll_up_rate(num_by_country, den_by_country, groups, n):
        num = np.bincount(groups, weights=np.nan_to_num(num_by_country), minlength=n)
        den = np.bincount(groups, weights=np.where(np.isnan(num_by_country), 0, den_by_country), minlength=n)
        return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)

    rg = st["rg"].to_numpy()
    forest_t = []
    for p in range(len(WINDOWS)):
        lost_km2 = forest[p] / 100 * area_km2["states"]
        forest_t.append({
            "nation": [float(np.nansum(lost_km2) / area_km2["states"][~np.isnan(lost_km2)].sum() * 100)],
            "regions": list(roll_up_rate(lost_km2, area_km2["states"], rg, len(tiers["regions"])) * 100),
            "states": list(forest[p]),
            "counties": list(forest[p][c_of_a]),
        })

    # wildfires: EONET events per 10,000 km2, admin-1 by point-in-polygon, rolled up by summing events
    ev = pd.read_csv(RAW / "eonet_wildfires.csv").dropna(subset=["lat", "lon"])
    ev["year"] = ev["date"].str[:4].astype(int)
    pts = gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev["lon"], ev["lat"]), crs=4326)
    hit_c = gpd.sjoin(pts, st[["geometry"]].reset_index(names="ci"), predicate="within", how="inner")
    hit_a = gpd.sjoin(pts, tiers["counties"][["geometry"]].reset_index(names="ai"), predicate="within", how="inner")
    fires_t = []
    for a, b in WINDOWS:
        yrs = max(1, b - max(a, 2016) + 1)
        nc = np.bincount(hit_c.loc[hit_c["year"].between(a, b), "ci"], minlength=len(st)).astype(float)
        na = np.bincount(hit_a.loc[hit_a["year"].between(a, b), "ai"], minlength=len(tiers["counties"])).astype(float)
        dens = lambda n, km2: n / km2 * 1e4 / yrs
        fires_t.append({"nation": [float(dens(nc.sum(), area_km2["states"].sum()))],
                        "regions": list(dens(np.bincount(rg, weights=nc, minlength=len(tiers["regions"])),
                                             np.bincount(rg, weights=area_km2["states"], minlength=len(tiers["regions"])))),
                        "states": list(dens(nc, area_km2["states"])), "counties": list(dens(na, area_km2["counties"]))})

    def pct(vals, q):
        v = np.asarray(vals, float); v = v[np.isfinite(v)]
        return float(np.percentile(v, q)) if len(v) else 0.0

    def finite(x):
        return [None if not np.isfinite(v) else round(float(v), 4) for v in x]

    metrics = []
    labels = {"temp": ("Temperature", "°C", "lin"), "humidity": ("Relative humidity", "%", "lin"), "precip": ("Precipitation", "mm/yr", "log")}
    for key, (label, unit, scale) in labels.items():
        tv = {k: finite(clim[k][key]) for k in TIERS}
        a = np.array([v for v in tv["counties"] if v is not None])
        a = np.log10(np.maximum(a, 1)) if scale == "log" else a
        metrics.append({"key": key, "label": label, "unit": unit, "scale": scale, "source": clim_src + (", humidity at country level" if key == "humidity" and "ERA5" in clim_src else ""),
                        "perPreset": False, "lo": pct(a, 5), "hi": pct(a, 95), "terciles": [pct(a, 33.3), pct(a, 66.7)], "tiers": tv})
    for key, label, unit, src, per in [("forest_loss", "Tree-cover loss", "% of land / yr", "OWID (Global Forest Watch / Hansen), country level", forest_t),
                                       ("fires", "Wildfire events", "per 10,000 km² / yr", "NASA EONET, 2016 onward", fires_t)]:
        allv = np.concatenate([np.array([v for v in d["counties"] if v is not None and np.isfinite(v)], float) for d in per])
        lg = np.log10(allv + 1e-3)
        nz = lg[allv > 0]
        if (allv == 0).mean() > 1 / 3:     # zero-heavy (e.g. wildfires): classes = none / below / above the non-zero median
            lo, hi, terc = float(np.log10(1e-3)), pct(nz, 95), [float(np.log10(1e-3)) + 1e-6, pct(nz, 50)]
        else:
            lo, hi, terc = pct(lg, 5), pct(lg, 97), [pct(lg, 33.3), pct(lg, 66.7)]
        metrics.append({"key": key, "label": label, "unit": unit, "scale": "log", "source": src, "perPreset": True,
                        "lo": lo, "hi": hi, "terciles": terc,
                        "tiers": [{k: finite(d[k]) for k in TIERS} for d in per]})
    (OUT / "metrics.json").write_text(json.dumps({"metrics": metrics, "trivariate": ["temp", "humidity"]}, separators=(",", ":")))
    for m in metrics:
        print(f"{m['key']:12s} lo {m['lo']:.2f} hi {m['hi']:.2f} ({m['scale']}), admin-1 missing: "
              f"{sum(v is None for v in (m['tiers'][0] if m['perPreset'] else m['tiers'])['counties'])}")


if __name__ == "__main__":
    main()
