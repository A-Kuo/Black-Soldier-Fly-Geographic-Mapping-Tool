"""Precompute every tile the WebGL map draws, so the browser never cuts GeoJSON or blurs fields at runtime.

1. Vector: tippecanoe (run inside WSL) cuts the five tiers plus an "outside the US" mask into one PMTiles archive,
   simplifying per zoom. Each tier is only tiled over the zooms where it is visible.
2. Raster: the Canvas prototype's smooth gradient (normalised Gaussian blur of each tier's log rate) is computed
   here on a 2 km Albers grid and baked to pre-coloured WebP tiles: one small PMTiles archive per tier per preset.

Reads  data/raw/us_tiers/prototype_data.json   (python bsfl_us_prototype.py)
       dashboard/gpu/data/*.geojson             (python bsfl_gpu_export.py, run automatically if missing)
Writes dashboard/gpu/tiles/us.pmtiles, dashboard/gpu/tiles/grad_{tier}_{preset}.pmtiles, dashboard/gpu/tiles/names.json

Needs tippecanoe in WSL Ubuntu at ~/.mamba/envs/tiles/bin (micromamba, conda-forge; no root required).
"""

import io
import json
import math
import shlex
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer
from pyproj import Transformer
from rasterio import features
from rasterio.transform import from_origin
from scipy import ndimage
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape

import bsfl_gpu_export
from bsfl_sources import ROOT
from bsfl_us_prototype import ALBERS, DATA_CACHE

GPU = ROOT / "dashboard" / "gpu"
GEOJSON = GPU / "data"
TILES = GPU / "tiles"
WSL_DISTRO = "Ubuntu"
TC_BIN = "~/.mamba/envs/tiles/bin"

TIERS = ["nation", "regions", "states", "counties", "cells"]
# vector zoom ranges per layer; the page cross-fades tiers at z 2.6, 3.4, 5.0 and 6.6 and overzooms past z9.
# Every layer that stays visible when zoomed in must reach the same max zoom, or it is missing from the deepest tiles.
VECTOR_ZOOMS = {"outside": (0, 9), "nation": (0, 9), "regions": (0, 9), "states": (0, 9), "counties": (4, 9), "cells": (6, 9)}
SHARED_BORDERS = {"regions", "states", "counties"}
# raster zoom ranges per tier: from one zoom before it fades in to the zoom past which overzooming is smooth enough
RASTER_ZOOMS = {"nation": (0, 3), "regions": (1, 4), "states": (2, 6), "counties": (4, 7), "cells": (5, 7)}
BOUNDS = (-126.0, 23.5, -65.5, 50.0)      # lon/lat window that holds the contiguous US

# gradient field, matching dashboard/us_prototype_template.html (sigmas in km)
RES, PAD = 2.0, 70.0
SIG = {"s": 40, "rBlur": 230, "rLight": 80, "n": 480, "c": 14, "k": 9}
EPS = {"s": 0.6, "r": 0.2, "c": 0.5, "k": 0.5}
LO, HI = math.log10(2), math.log10(2000)
RAMP = ["#f2f0f7", "#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#4a1486"]   # same stops as the vector layers
TILE = 256


# ---------------------------------------------------------------- vector tiles
def wsl_path(p) -> str:
    s = str(p.resolve())
    return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")


def wsl(cmd: str):
    subprocess.run(["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc", cmd], check=True)


def write_outside_mask():
    """World minus the US outline: drawn in the background colour above the gradient rasters, so the coast stays crisp."""
    nation = shape(json.loads((GEOJSON / "nation.geojson").read_text(encoding="utf-8"))["features"][0]["geometry"])
    nation = make_valid(nation)                           # rounding to 5 decimals can pinch narrow inlets
    outside = box(-180, -85, 180, 85).difference(nation)
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": mapping(outside)}]}
    (GEOJSON / "outside.geojson").write_text(json.dumps(fc, separators=(",", ":")), encoding="utf-8")


def build_vector():
    t0 = time.time()
    tmp = "$HOME/.cache/bsfl_tiles"                           # WSL's own disk: /tmp is wiped on VM restart, /mnt/c breaks SQLite
    parts = []
    cmds = [f"mkdir -p {tmp}"]
    for name, (z0, z1) in VECTOR_ZOOMS.items():
        out = f"{tmp}/{name}.pmtiles"
        src = GEOJSON / f"{name}.geojson"
        flags = ["--force", "--quiet", "-l", name, f"-Z{z0}", f"-z{z1}", "--no-feature-limit", "--no-tile-size-limit",
                 "--no-tiny-polygon-reduction"] + (["--detect-shared-borders"] if name in SHARED_BORDERS else [])
        run = f"{TC_BIN}/tippecanoe -o {out} {' '.join(flags)} {shlex.quote(wsl_path(src))}"
        stamp = f"{run} # {src.stat().st_mtime_ns}"            # re-cut a layer only when its flags or its GeoJSON change
        cmds.append(f"if [ \"$(cat {out}.stamp 2>/dev/null)\" != {shlex.quote(stamp)} ]; then {run} && "
                    f"printf %s {shlex.quote(stamp)} > {out}.stamp; fi")
        parts.append(out)
    cmds.append(f"{TC_BIN}/tile-join --force --quiet --no-tile-size-limit -o {tmp}/us.pmtiles " + " ".join(parts))
    cmds.append(f"cp {tmp}/us.pmtiles {shlex.quote(wsl_path(TILES / 'us.pmtiles'))}")     # copy the finished file to Windows
    wsl(" && ".join(cmds))
    print(f"vector: us.pmtiles {(TILES / 'us.pmtiles').stat().st_size / 1e6:.1f} MB in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- gradient fields
def decode(polys):
    parts = []
    for p in polys:
        rings = [np.cumsum(np.asarray(r, dtype=np.int64).reshape(-1, 2), axis=0) / 10.0 for r in p]
        parts.append(Polygon(rings[0], rings[1:]))
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def blur(a: np.ndarray, sigma_km: float) -> np.ndarray:
    """Gaussian blur with zero padding. Wide kernels run on a block-averaged grid, then upsample."""
    s = sigma_km / RES
    f = max(1, int(s // 6))
    if f == 1:
        return ndimage.gaussian_filter(a, s, mode="constant", truncate=3.0)
    H, W = a.shape
    b = np.zeros((-(-H // f) * f, -(-W // f) * f), np.float32)
    b[:H, :W] = a
    b = b.reshape(b.shape[0] // f, f, b.shape[1] // f, f).mean(axis=(1, 3))
    g = ndimage.gaussian_filter(b, s / f, mode="constant", truncate=3.0)
    return ndimage.zoom(g, f, order=1)[:H, :W]


def build_fields(D):
    """Returns the grid georeference and fields[preset][tier] -> float32 log10 rate (NaN where undefined)."""
    bx0, by0, bx1, by1 = D["bbox"]
    x0, y1 = bx0 - PAD, by1 + PAD
    W, H = math.ceil((bx1 - bx0 + 2 * PAD) / RES), math.ceil((by1 - by0 + 2 * PAD) / RES)
    tf = from_origin(x0, y1, RES, RES)
    raster = lambda units: features.rasterize([(decode(u["polys"]), i + 1) for i, u in enumerate(units)],
                                              out_shape=(H, W), transform=tf, fill=0, dtype="int32")
    ids, cids, kids = raster(D["states"]), raster(D["counties"]), raster(D["cells"])
    mask = (ids > 0).astype(np.float32)
    region_of = np.array([0] + [s["region"] for s in D["states"]])[ids]
    print(f"grid {W} x {H} at {RES:g} km", flush=True)

    dens = {}
    def ball(vals, sk):
        if sk not in dens:
            dens[sk] = blur(mask, sk)
        den = dens[sk]
        num = blur(vals, sk)
        return np.where(den > 2e-3, num / np.maximum(den, 1e-9), np.nan).astype(np.float32)

    lg = lambda d, p: math.log10(max(d[p][2], 1e-4))
    out = []
    for p in range(len(D["presets"])):
        sv = np.array([0.0] + [lg(u["d"], p) for u in D["states"]], np.float32)
        cv = np.array([0.0] + [lg(u["d"], p) for u in D["counties"]], np.float32)
        kv = np.array([0.0] + [lg(u["d"], p) for u in D["cells"]], np.float32)
        vS = sv[ids] * mask
        vC = np.where(cids > 0, cv[cids], vS) * mask
        vK = np.where(kids > 0, kv[kids], vC) * mask
        inside = mask > 0
        bS, bR = ball(vS, SIG["s"]), ball(vS, SIG["rLight"])
        bg = np.full((H, W), np.nan, np.float32)
        for g in range(len(D["regions"])):
            m = (inside & (region_of == g)).astype(np.float32)
            num, den = blur(vS * m, SIG["rBlur"]), blur(m, SIG["rBlur"])
            sel = m > 0
            bg[sel] = num[sel] / den[sel]
        bC, bK = ball(vC, SIG["c"]), ball(vK, SIG["k"])
        out.append({
            "nation": ball(vS, SIG["n"]),
            "regions": np.where(inside, (1 - EPS["r"]) * bg + EPS["r"] * bR, bR),
            "states": np.where(inside, (1 - EPS["s"]) * vS + EPS["s"] * bS, bS),
            "counties": np.where(inside, (1 - EPS["c"]) * vC + EPS["c"] * bC, bC),
            "cells": np.where(inside, (1 - EPS["k"]) * vK + EPS["k"] * bK, bK),
        })
        print(f"fields preset {p} done", flush=True)
    return (x0, y1), out


# ---------------------------------------------------------------- raster tiles
def lut() -> np.ndarray:
    stops = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in RAMP], float)
    t = np.linspace(0, 1, 256)[:, None] * (len(RAMP) - 1)
    i = np.minimum(t.astype(int), len(RAMP) - 2)
    return np.round(stops[i[:, 0]] + (stops[i[:, 0] + 1] - stops[i[:, 0]]) * (t - i)).astype(np.uint8)


def tile_range(z):
    n = 2 ** z
    tx = lambda lon: int((lon + 180) / 360 * n)
    ty = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return [(x, y) for x in range(tx(BOUNDS[0]), tx(BOUNDS[2]) + 1) for y in range(ty(BOUNDS[3]), ty(BOUNDS[1]) + 1)]


def build_rasters(D):
    t0 = time.time()
    (x0, y1), fields = build_fields(D)
    colours = lut()
    to_albers = Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True)
    P = len(fields)
    writers, files = {}, []
    for tier in TIERS:
        for p in range(P):
            f = open(TILES / f"grad_{tier}_{p}.pmtiles", "wb")
            files.append(f)
            writers[(tier, p)] = Writer(f)
    z_lo, z_hi = min(r[0] for r in RASTER_ZOOMS.values()), max(r[1] for r in RASTER_ZOOMS.values())
    jobs = sorted((zxy_to_tileid(z, x, y), z, x, y) for z in range(z_lo, z_hi + 1) for x, y in tile_range(z))
    written = 0
    for _, z, x, y in jobs:                                    # tile-id order, as each archive requires
        tiers = [t for t in TIERS if RASTER_ZOOMS[t][0] <= z <= RASTER_ZOOMS[t][1]]
        if not tiers:
            continue
        n = 2 ** z
        px = (x + (np.arange(TILE) + 0.5) / TILE) / n * 360 - 180
        py = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + (np.arange(TILE) + 0.5) / TILE) / n))))
        lon, lat = np.meshgrid(px, py)
        X, Y = to_albers.transform(lon.ravel(), lat.ravel())
        cols, rows = (np.asarray(X) - x0) / RES - 0.5, (y1 - np.asarray(Y)) / RES - 0.5
        for tier in tiers:
            for p in range(P):
                v = ndimage.map_coordinates(fields[p][tier], [rows, cols], order=1, mode="constant", cval=np.nan)
                ok = np.isfinite(v)
                if not ok.any():
                    continue
                t = np.clip((np.nan_to_num(v, nan=LO) - LO) / (HI - LO), 0, 1)
                rgba = np.zeros((TILE * TILE, 4), np.uint8)
                rgba[:, :3] = colours[np.round(t * 255).astype(int)]
                rgba[:, 3] = np.where(ok, 255, 0)
                buf = io.BytesIO()
                Image.fromarray(rgba.reshape(TILE, TILE, 4), "RGBA").save(buf, "WEBP", quality=88, method=4)
                writers[(tier, p)].write_tile(zxy_to_tileid(z, x, y), buf.getvalue())
                written += 1
    e7 = lambda v: int(round(v * 1e7))
    for (tier, p), w in writers.items():
        header = {"tile_type": TileType.WEBP, "tile_compression": Compression.NONE,
                  "min_lon_e7": e7(BOUNDS[0]), "min_lat_e7": e7(BOUNDS[1]), "max_lon_e7": e7(BOUNDS[2]), "max_lat_e7": e7(BOUNDS[3]),
                  "center_zoom": RASTER_ZOOMS[tier][0], "center_lon_e7": e7(-96), "center_lat_e7": e7(38)}
        w.finalize(header, {"name": f"BSF gradient, {tier}, preset {p}", "preset": D["presets"][p]})
    for f in files:
        f.close()
    size = sum((TILES / f"grad_{t}_{p}.pmtiles").stat().st_size for t in TIERS for p in range(P))
    print(f"raster: {written} tiles in {len(writers)} archives, {size / 1e6:.1f} MB, {time.time() - t0:.0f}s", flush=True)


def write_names(D):
    abbr = [s["abbr"] for s in D["states"]]
    names = {"states": [s["name"] for s in D["states"]], "abbr": abbr,
             "counties": [f"{c['n']}, {abbr[c['s']]}" for c in D["counties"]]}
    (TILES / "names.json").write_text(json.dumps(names, separators=(",", ":")), encoding="utf-8")


def main():
    steps = [s for s in sys.argv[1:] if not s.startswith("-")] or ["vector", "raster"]
    D = json.loads(DATA_CACHE.read_text(encoding="utf-8"))
    TILES.mkdir(parents=True, exist_ok=True)
    if not all((GEOJSON / f"{t}.geojson").exists() for t in TIERS):
        bsfl_gpu_export.main()
    write_names(D)
    if "vector" in steps:
        write_outside_mask()
        build_vector()
    if "raster" in steps:
        build_rasters(D)


if __name__ == "__main__":
    main()
