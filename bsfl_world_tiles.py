"""World admin-2 tier and vector tiles for the GPU gradient field (dashboard/gpu/field.html?set=world).

Adds GADM 4.1 level-2 units (counties, districts, municipios...) as the fifth tier, then cuts every tier into one
PMTiles archive so the browser loads only the tiles in view instead of ~15 MB+ of GeoJSON.

  admin-2 estimates  GBIF GADM_LEVEL_2_GID facets per preset window, Gamma-Poisson shrinkage (K = 1000 insect records)
                     toward the unit's admin-1 estimate, or its country's where GADM has no admin-1 parent in our set
  units.json         every tier's per-unit properties (names, parents, e/l/h/c/b/n per preset), no geometry
  world.pmtiles      layers nation, regions, states (countries), counties (admin-1), cells (admin-2); each feature has
                     id (index within its tier), uid (24-bit id painted by the browser's ID pass) and e/c per preset

uid spaces, so one RGB pixel names the most detailed unit under it: admin-2 = 1..nK, admin-1 = nK+1..nK+nC,
countries = nK+nC+1..

Reads  dashboard/gpu/data/world/*.geojson (python bsfl_world_field.py), data/raw/gadm41_admin1.gpkg,
       GADM level-2 per country (downloaded to data/raw/gadm41/), Natural Earth 50m
Writes data/processed/gbif_admin2_window.csv, dashboard/gpu/data/world/{units.json, units_cells.json, world.pmtiles, config.json}
Needs  tippecanoe in WSL Ubuntu at ~/.mamba/envs/tiles/bin (see bsfl_gpu_tiles.py)
"""

import json
import math
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from shapely.ops import unary_union

from bsfl_sources import PROCESSED, RAW, natural_earth
from bsfl_world_field import K, OUT, WINDOWS, posterior, props

GADM_DIR = RAW / "gadm41"
GBIF = "https://api.gbif.org/v1/occurrence/search"
CLEAN = dict(hasCoordinate="true", hasGeospatialIssue="false", occurrenceStatus="PRESENT")
BSF, INSECTA = 1577153, 216
TC_BIN, WSL_DISTRO = "~/.mamba/envs/tiles/bin", "Ubuntu"
# tile zoom range per layer; all share one max zoom (overzoomed past it), or tile-join leaves the shallower layers
# missing from the deepest tiles and their borders vanish when zoomed in
ZOOMS = {"nation": (0, 10), "regions": (0, 10), "states": (0, 10), "counties": (2, 10), "cells": (5, 10)}   # admin-2 appears ~z5.2
SESSION = requests.Session()


# ---------------------------------------------------------------- inputs
def gadm_level2(isos: list[str]) -> gpd.GeoDataFrame:
    def fetch(iso):
        path = GADM_DIR / f"gadm41_{iso}_2.json.zip"
        if not path.exists():
            r = SESSION.get(f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/{path.name}", timeout=300)
            if r.status_code == 404:
                return None                                   # no level 2 in GADM for this country
            r.raise_for_status()
            path.write_bytes(r.content)
        return gpd.read_file(f"zip://{path}")[["GID_0", "GID_1", "GID_2", "NAME_1", "NAME_2", "COUNTRY", "geometry"]]
    with ThreadPoolExecutor(4) as ex:
        parts = [p for p in ex.map(fetch, isos) if p is not None]
    return pd.concat(parts, ignore_index=True).set_crs(4326, allow_override=True)


def facet_level2(taxon: int, a: int, b: int) -> dict[str, int]:
    out, off = {}, 0
    while True:
        for attempt in range(6):
            try:
                r = SESSION.get(GBIF, params={**CLEAN, "taxonKey": taxon, "year": f"{a},{b}", "limit": 0,
                                              "facet": "GADM_LEVEL_2_GID", "facetLimit": 1000, "facetOffset": off}, timeout=180)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2 ** attempt)
        counts = r.json()["facets"][0]["counts"] if r.json()["facets"] else []
        out.update({c["name"]: c["count"] for c in counts})
        if len(counts) < 1000:
            return out
        off += 1000


def admin2_counts() -> pd.DataFrame:
    path = PROCESSED / "gbif_admin2_window.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for a, b in WINDOWS:
        bsf, ins = facet_level2(BSF, a, b), facet_level2(INSECTA, a, b)
        rows += [{"gid2": g, "window": f"{a}-{b}", "bsf": bsf.get(g, 0), "insecta": ins.get(g, 0)} for g in set(bsf) | set(ins)]
        print(f"  GBIF admin-2 {a}-{b}: {len(bsf)} units with BSF, {len(ins)} with insects", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------- tiles
def wsl_path(p) -> str:
    s = str(p.resolve())
    return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")


def write_seq(path, gdf: gpd.GeoDataFrame, prop_rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for g, pr in zip(gdf.geometry, prop_rows):
            if g is None or g.is_empty:
                continue
            f.write(json.dumps({"type": "Feature", "id": pr["id"], "properties": pr, "geometry": g.__geo_interface__}, separators=(",", ":")) + "\n")


def tile(seqs: dict):
    tmp = "$HOME/.cache/bsfl_world_tiles"
    cmds, parts = [f"mkdir -p {tmp}"], []
    for name, src in seqs.items():
        z0, z1 = ZOOMS[name]
        out = f"{tmp}/{name}.pmtiles"
        flags = ["--force", "--quiet", "-l", name, f"-Z{z0}", f"-z{z1}", "-P", "--no-feature-limit", "--no-tile-size-limit",
                 "--no-tiny-polygon-reduction", "--detect-shared-borders"]   # ids come from the "id" property (promoteId); tippecanoe cannot store id 0
        cmds.append(f"{TC_BIN}/tippecanoe -o {out} {' '.join(flags)} {shlex.quote(wsl_path(src))}")
        parts.append(out)
    cmds.append(f"{TC_BIN}/tile-join --force --quiet --no-tile-size-limit -o {tmp}/world.pmtiles " + " ".join(parts))
    cmds.append(f"cp {tmp}/world.pmtiles {shlex.quote(wsl_path(OUT / 'world.pmtiles'))}")
    subprocess.run(["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc", " && ".join(cmds)], check=True)


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    read = lambda k: json.loads((OUT / f"{k}.geojson").read_text(encoding="utf-8"))["features"]
    tiers = {k: read(k) for k in ("nation", "regions", "states", "counties")}
    S, C = tiers["states"], tiers["counties"]
    s_index = {f["properties"]["ab"]: i for i, f in enumerate(S)}
    c_index = {f["properties"]["gid"]: i for i, f in enumerate(C)}
    est = lambda f, p: 10 ** f["properties"][f"e{p}"] / 1e6

    # ---- admin-2 units and estimates
    a2 = gadm_level2(sorted(s_index))
    a2 = a2[a2["GID_0"].isin(s_index)].sort_values("GID_2").reset_index(drop=True)
    a2["st"] = a2["GID_0"].map(s_index)
    a2["ct"] = a2["GID_1"].map(c_index).fillna(-1).astype(int)
    nK, nC, nS = len(a2), len(C), len(S)
    print(f"admin-2: {nK} units in {a2['st'].nunique()} countries; {(a2['ct'] < 0).sum()} without an admin-1 parent in the set", flush=True)

    # ---- units.json: static properties (names, parents) for every tier, aligned with tile feature ids. Estimates are not
    # stored here: the timeline step files (bsfl_world_timeline.py) carry them per span and year.
    cells = [{"id": i, "nm": f"{row.NAME_2}, {row.NAME_1}, {row.COUNTRY}", "gid": row.GID_2, "ct": int(row.ct), "st": int(row.st)}
             for i, row in enumerate(a2.itertuples())]
    # admin-2 properties go in their own file: the page needs them only once it zooms into the detail view
    static = lambda pr: {k: v for k, v in pr.items() if not (len(k) == 2 and k[0] in "elhcbn" and k[1].isdigit())}
    units = {k: [static(f["properties"]) for f in v] for k, v in tiers.items()}
    (OUT / "units.json").write_text(json.dumps(units, separators=(",", ":")), encoding="utf-8")
    (OUT / "units_cells.json").write_text(json.dumps(cells, separators=(",", ":")), encoding="utf-8")

    # ---- tile sources: full-resolution admin-1 and admin-2, compact properties
    keep = lambda pr, uid: {"id": pr["id"], "uid": uid}      # values come from the timeline files, via feature-state
    seq = {}
    seq_dir = OUT / "_tiles_src"; seq_dir.mkdir(exist_ok=True)
    # country, region and world outlines from Natural Earth 10m (the field build's 50m copies are simplified for speed);
    # tippecanoe does its own per-zoom simplification, so the deep zooms keep ~1 km detail
    ne = gpd.read_file(f"zip://{natural_earth('10m', 'admin_0_countries')}")
    ne["iso3"] = np.where(ne["ISO_A3_EH"] != "-99", ne["ISO_A3_EH"], ne["ADM0_A3"])
    ne10 = ne.set_geometry(ne.geometry.make_valid()).dissolve("iso3").geometry
    g_states = [ne10.get(f["properties"]["ab"]) or gpd.GeoSeries.from_features([f]).iloc[0] for f in S]
    rg = np.array([f["properties"]["rg"] for f in S])
    g_regions = [unary_union([g for g, r in zip(g_states, rg) if r == i]) for i in range(len(tiers["regions"]))]
    hi_res = {"states": g_states, "regions": g_regions, "nation": [unary_union(g_regions)]}
    for k, offset in (("nation", None), ("regions", None), ("states", nK + nC)):
        g = gpd.GeoDataFrame(geometry=hi_res[k], crs=4326)
        rows = [keep(f["properties"], (offset + i + 1) if offset is not None else 0) for i, f in enumerate(tiers[k])]
        write_seq(seq_dir / f"{k}.geojsons", g, rows); seq[k] = seq_dir / f"{k}.geojsons"
    a1 = gpd.read_file(RAW / "gadm41_admin1.gpkg").set_index("gid1")
    g1 = gpd.GeoDataFrame(geometry=[a1.geometry.get(f["properties"]["gid"]) for f in C], crs=4326)
    write_seq(seq_dir / "counties.geojsons", g1, [keep(f["properties"], nK + i + 1) for i, f in enumerate(C)]); seq["counties"] = seq_dir / "counties.geojsons"
    write_seq(seq_dir / "cells.geojsons", a2, [keep(c, i + 1) for i, c in enumerate(cells)]); seq["cells"] = seq_dir / "cells.geojsons"
    tile(seq)

    cfg = json.loads((OUT / "config.json").read_text())
    cfg.update({"tierNames": ["World", "Regions", "Countries", "Admin-1", "Admin-2"], "bound": [-0.6, 0.6, 2.2, 4.0],
                "vector": "world.pmtiles", "detail": {"fromZoom": 1.6, "uid": {"nK": nK, "nC": nC, "nS": nS}}})
    (OUT / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"world.pmtiles {(OUT / 'world.pmtiles').stat().st_size / 1e6:.0f} MB, units.json "
          f"{(OUT / 'units.json').stat().st_size / 1e6:.1f} MB + units_cells.json "
          f"{(OUT / 'units_cells.json').stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
