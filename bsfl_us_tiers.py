"""Pull sub-state BSF data for the contiguous US: county (GADM level 2) facets and ~30 km cell counts.

Presets = 3 year windows x 2 denominators ("all" = every Insecta / BSF record, "inat" = iNaturalist-only on both sides).

Counties   GBIF occurrence facets on GADM_LEVEL_1_GID / GADM_LEVEL_2_GID, per window, denominator and taxon.
Cells      BSF raw points (exact) binned into the Web-Mercator grid that GBIF's z7 density tiles use (8 x 8 squares
           per tile, ~26-35 km on the ground); Insecta counts per cell come from those tiles. The "all" denominator
           uses the exact density endpoint; the "inat" denominator needs the ad-hoc endpoint (datasetKey filter),
           whose totals agree with the occurrence API to about +/-2%.

Writes data/processed/us_tier_facets.csv, data/raw/us_tiers/bsf_points.csv, data/raw/us_tiers/cells_<den>_<window>.csv
Resumable: finished pieces are skipped on re-run.
"""

import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import mapbox_vector_tile as mvt
import pandas as pd
import requests

from bsfl_sources import PROCESSED, RAW

TIERS = RAW / "us_tiers"
TIERS.mkdir(exist_ok=True, parents=True)
BSF, INSECTA = 1577153, 216
INAT = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
WINDOWS = [(2010, 2024), (2018, 2024), (2022, 2024)]
DENS = ["all", "inat"]
Z, SQUARE = 5, 64            # z5 tiles, 64 squares per tile edge -> 2048 squares around the world, ~15-20 km on the ground
CPT = 4096 // SQUARE         # squares per tile edge (tile extent is 4096 units, squareSize is in those units)
NCELL = 2 ** Z * CPT
CLEAN = dict(hasCoordinate="true", hasGeospatialIssue="false", occurrenceStatus="PRESENT", country="US")
LAT0, LAT1, LON0, LON1 = 24.0, 49.7, -125.0, -66.0
S = requests.Session()


def fetch(url, params, binary=False, tries=8):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=120)
        except requests.RequestException:
            time.sleep(2 ** min(i, 5)); continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** min(i, 5)); continue
        r.raise_for_status()
        return r.content if binary else r.json()
    raise RuntimeError(f"gave up on {url} {params}")


def window_params(win, den):
    p = {**CLEAN, "year": f"{win[0]},{win[1]}"}
    if den == "inat":
        p["datasetKey"] = INAT
    return p


def facet(level, taxon, win, den):
    out, off = {}, 0
    while True:
        j = fetch("https://api.gbif.org/v1/occurrence/search",
                  {**window_params(win, den), "taxonKey": taxon, "limit": 0, "facet": f"GADM_LEVEL_{level}_GID",
                   "facetLimit": 1000, "facetOffset": off})
        counts = j["facets"][0]["counts"] if j.get("facets") else []
        out.update({c["name"]: c["count"] for c in counts})
        if len(counts) < 1000:
            return out
        off += 1000


def pull_facets():
    path = PROCESSED / "us_tier_facets.csv"
    if path.exists():
        return
    rows = []
    for level in (1, 2):
        for win in WINDOWS:
            for den in DENS:
                for taxon, tname in ((BSF, "bsf"), (INSECTA, "insecta")):
                    f = facet(level, taxon, win, den)
                    rows += [(level, f"{win[0]}-{win[1]}", den, tname, gid, n) for gid, n in f.items()]
                    print(f"facet L{level} {win} {den} {tname}: {len(f)} units, {sum(f.values())} records", flush=True)
    pd.DataFrame(rows, columns=["level", "window", "den", "taxon", "gid", "count"]).to_csv(path, index=False)


def pull_points():
    path = TIERS / "bsf_points.csv"
    if path.exists():
        return
    rows, off = [], 0
    while True:
        j = fetch("https://api.gbif.org/v1/occurrence/search",
                  {**CLEAN, "year": "2010,2024", "taxonKey": BSF, "decimalLatitude": f"{LAT0},{LAT1}",
                   "decimalLongitude": f"{LON0},{LON1}", "limit": 300, "offset": off})
        rows += [(r["year"], r["decimalLatitude"], r["decimalLongitude"], r.get("datasetKey", "")) for r in j["results"]]
        if j["endOfRecords"]:
            break
        off += 300
    pd.DataFrame(rows, columns=["year", "lat", "lon", "dataset"]).to_csv(path, index=False)
    print(f"BSF points: {len(rows)}", flush=True)


def tile_range():
    n = 2 ** Z
    x = lambda lon: int((lon + 180) / 360 * n)
    y = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return range(x(LON0), x(LON1) + 1), range(y(LAT1), y(LAT0) + 1)


def tile_cells(tx, ty, win, den):
    base = {**window_params(win, den), "taxonKey": INSECTA, "srs": "EPSG:3857", "bin": "square", "squareSize": SQUARE}
    url = (f"https://api.gbif.org/v2/map/occurrence/{'adhoc' if den == 'inat' else 'density'}/{Z}/{tx}/{ty}.mvt")
    if den == "inat":
        base["style"] = "classic.point"
    try:
        content = fetch(url, base, binary=True)
    except requests.HTTPError as e:                             # the ad-hoc endpoint answers 400 for a tile with no records
        if e.response is not None and e.response.status_code == 400 and "missing the expected layer" in e.response.text:
            return []
        raise
    d = mvt.decode(content, default_options={"y_coord_down": True})
    out = []
    for layer in d.values():
        extent = layer.get("extent", 4096)
        assert extent // SQUARE == CPT, f"unexpected tile extent {extent}"      # squareSize is in tile units: 4096/64 = 64 squares per tile edge
        for f in layer["features"]:
            ring = f["geometry"]["coordinates"][0]
            x0, y0 = min(p[0] for p in ring), min(p[1] for p in ring)
            w = max(p[0] for p in ring) - x0
            assert abs(w - SQUARE) <= 1, f"unexpected cell width {w}"
            out.append((tx * CPT + int(round(x0 / SQUARE)), ty * CPT + int(round(y0 / SQUARE)), f["properties"]["total"]))
    return out


def pull_cells():
    xs, ys = tile_range()
    tiles = [(x, y) for x in xs for y in ys]
    for den in DENS:
        for win in WINDOWS:
            path = TIERS / f"cells_{den}_{win[0]}-{win[1]}.csv"
            if path.exists():
                continue
            t0 = time.time()
            with ThreadPoolExecutor(3) as ex:
                res = list(ex.map(lambda t: tile_cells(t[0], t[1], win, den), tiles))
            rows = [r for tile in res for r in tile]
            pd.DataFrame(rows, columns=["col", "row", "n"]).to_csv(path, index=False)
            print(f"cells {den} {win}: {len(rows)} cells, {sum(r[2] for r in rows)} insect records, "
                  f"{len(tiles)} tiles in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    steps = sys.argv[1:] or ["facets", "points", "cells"]
    if "facets" in steps: pull_facets()
    if "points" in steps: pull_points()
    if "cells" in steps: pull_cells()
    print("done", flush=True)
