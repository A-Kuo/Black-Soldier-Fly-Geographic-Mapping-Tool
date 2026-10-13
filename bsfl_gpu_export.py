"""Export the US multi-scale prototype's precomputed tiers as GeoJSON for the WebGL (MapLibre) proof of concept.

The statistics are not recomputed: this reads the build cache written by bsfl_us_prototype.py, decodes its
delta-coded Albers rings back to lon/lat, and flattens every preset into feature properties so the browser can
switch presets by swapping a paint expression (no per-feature JavaScript work).

Properties per feature, for each preset p = 0..5:
  e{p}  log10 of the shrunk rate (BSF records per million insect records)
  l{p}, h{p}  95% interval (per million), c{p} confidence class (0 high, 1 medium, 2 low), b{p} BSF, n{p} insects

Reads  data/raw/us_tiers/prototype_data.json   (python bsfl_us_prototype.py)
Writes dashboard/gpu/data/{nation,regions,states,counties,cells}.geojson, dashboard/gpu/data/meta.json
"""

import json
import math

import numpy as np
from pyproj import Transformer

from bsfl_sources import ROOT
from bsfl_us_prototype import ALBERS, DATA_CACHE

OUT = ROOT / "dashboard" / "gpu" / "data"
TO_LONLAT = Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True)


def rings_lonlat(polys) -> list:
    """Prototype polys ([[delta-coded ring, ...], ...] in 0.1 km Albers) -> GeoJSON MultiPolygon coordinates."""
    out = []
    for p in polys:
        rings = []
        for r in p:
            xy = np.cumsum(np.asarray(r, dtype=np.int64).reshape(-1, 2), axis=0) / 10.0
            lon, lat = TO_LONLAT.transform(xy[:, 0], xy[:, 1])
            ring = np.round(np.column_stack([lon, lat]), 5).tolist()
            rings.append(ring + [ring[0]])
        out.append(rings)
    return out


def props(d: list) -> dict:
    out = {}
    for p, (B, N, est, lo, hi, conf) in enumerate(d):
        out |= {f"e{p}": round(math.log10(max(est, 1e-4)), 3), f"l{p}": lo, f"h{p}": hi, f"c{p}": conf, f"b{p}": B, f"n{p}": N}
    return out


def write(name: str, feats: list):
    fc = {"type": "FeatureCollection", "features": feats}
    path = OUT / f"{name}.geojson"
    path.write_text(json.dumps(fc, separators=(",", ":")), encoding="utf-8")
    print(f"{name}: {len(feats)} features, {path.stat().st_size / 1e6:.1f} MB")


def feature(i: int, polys, d, **extra) -> dict:
    return {"type": "Feature", "id": i, "properties": {"id": i, **extra, **props(d)},
            "geometry": {"type": "MultiPolygon", "coordinates": rings_lonlat(polys)}}


def main():
    D = json.loads(DATA_CACHE.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    regions = D["regions"]
    write("nation", [feature(0, D["nation"]["polys"], D["nation"]["d"], nm=D["nation"]["name"])])
    write("regions", [feature(i, r["polys"], r["d"], nm=r["name"]) for i, r in enumerate(regions)])
    write("states", [feature(i, s["polys"], s["d"], nm=s["name"], ab=s["abbr"], rg=s["region"]) for i, s in enumerate(D["states"])])
    write("counties", [feature(i, c["polys"], c["d"], nm=c["n"], st=c["s"]) for i, c in enumerate(D["counties"])])
    write("cells", [feature(i, k["polys"], k["d"], ct=k["c"]) for i, k in enumerate(D["cells"])])
    meta = {"presets": D["presets"], "k": D["k"], "cellKm": D["cellKm"], "built": D["built"],
            "regions": [r["name"] for r in regions]}
    (OUT / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


if __name__ == "__main__":
    main()
