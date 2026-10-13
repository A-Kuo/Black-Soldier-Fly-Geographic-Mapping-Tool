"""Build the continental-US multi-scale prototype: nation -> 4 Census regions -> states -> counties -> ~30 km cells.

Everything is precomputed here so the browser only blends and draws:
  * six presets = 3 year windows x 2 denominators (all insect records / iNaturalist only)
  * hierarchical Gamma-Poisson shrinkage: nation prior -> state -> county -> cell, prior strength K = 1000 insect records
  * 95% intervals and the same confidence classes as the main dashboard (interval ratio <=1.5 high, <=3 medium, else low)
  * geometry in a US Albers equal-area plane (km); states, regions and the nation are unions of the simplified counties,
    so every tier shares the same borders; cells are GBIF's Web-Mercator density squares clipped to the US outline

Reads  data/processed/us_tier_facets.csv, data/raw/us_tiers/*  (python bsfl_us_tiers.py), data/raw/gadm41/gadm41_USA_2.json.zip,
       data/raw/gadm41_admin1.gpkg, Natural Earth admin-0 (cached)
Writes dashboard/us_prototype.html  fully self-contained (no CDN, no tiles)
"""

import json
import math
import sys
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.stats import gamma
from shapely import STRtree
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from bsfl_admin1 import GADM_GPKG
from bsfl_sources import PROCESSED, RAW, ROOT, natural_earth
from bsfl_us_tiers import DENS, INAT, NCELL, SQUARE, TIERS, WINDOWS, Z

OUT = ROOT / "dashboard"
TEMPLATE = OUT / "us_prototype_template.html"
DATA_CACHE = TIERS / "prototype_data.json"      # build cache, kept under data/raw (git-ignored)
ALBERS = "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +datum=WGS84 +units=km +no_defs"
K = 1000
COUNTY_SIMPLIFY_KM = 2.0
GADM_USA2 = RAW / "gadm41" / "gadm41_USA_2.json.zip"

STATES = {
    "Alabama": ("AL", "South"), "Arizona": ("AZ", "West"), "Arkansas": ("AR", "South"), "California": ("CA", "West"),
    "Colorado": ("CO", "West"), "Connecticut": ("CT", "Northeast"), "Delaware": ("DE", "South"),
    "DistrictofColumbia": ("DC", "South"), "Florida": ("FL", "South"), "Georgia": ("GA", "South"),
    "Idaho": ("ID", "West"), "Illinois": ("IL", "Midwest"), "Indiana": ("IN", "Midwest"), "Iowa": ("IA", "Midwest"),
    "Kansas": ("KS", "Midwest"), "Kentucky": ("KY", "South"), "Louisiana": ("LA", "South"),
    "Maine": ("ME", "Northeast"), "Maryland": ("MD", "South"), "Massachusetts": ("MA", "Northeast"),
    "Michigan": ("MI", "Midwest"), "Minnesota": ("MN", "Midwest"), "Mississippi": ("MS", "South"),
    "Missouri": ("MO", "Midwest"), "Montana": ("MT", "West"), "Nebraska": ("NE", "Midwest"),
    "Nevada": ("NV", "West"), "NewHampshire": ("NH", "Northeast"), "NewJersey": ("NJ", "Northeast"),
    "NewMexico": ("NM", "West"), "NewYork": ("NY", "Northeast"), "NorthCarolina": ("NC", "South"),
    "NorthDakota": ("ND", "Midwest"), "Ohio": ("OH", "Midwest"), "Oklahoma": ("OK", "South"),
    "Oregon": ("OR", "West"), "Pennsylvania": ("PA", "Northeast"), "RhodeIsland": ("RI", "Northeast"),
    "SouthCarolina": ("SC", "South"), "SouthDakota": ("SD", "Midwest"), "Tennessee": ("TN", "South"),
    "Texas": ("TX", "South"), "Utah": ("UT", "West"), "Vermont": ("VT", "Northeast"), "Virginia": ("VA", "South"),
    "Washington": ("WA", "West"), "WestVirginia": ("WV", "South"), "Wisconsin": ("WI", "Midwest"),
    "Wyoming": ("WY", "West"),
}
REGIONS = ["Northeast", "Midwest", "South", "West"]
CONF_HIGH, CONF_MED = 3.0, 10.0          # 95% interval ratio hi/lo
PRESETS = [(f"{w[0]}-{w[1]}", d) for w in WINDOWS for d in DENS]       # index = window * 2 + denominator
TYPE_SUFFIX = {"County": "County", "Parish": "Parish", "Borough": "Borough", "CensusArea": "Census Area",
               "IndependentCity": "(independent city)", "CityandBorough": "City and Borough"}


def uncamel(s: str) -> str:
    out = []
    for i, ch in enumerate(s):
        if i and ch.isupper() and (s[i - 1].islower() or s[i - 1] == "."):
            out.append(" ")
        out.append(ch)
    return "".join(out).replace("ofColumbia", "of Columbia")


def sig3(x: float) -> float:
    return float(f"{x:.3g}")


def polys(geom, min_area=0.0, min_hole=3.0) -> list:
    """(Multi)Polygon -> [[ring, ...], ...], outer rings CCW and holes CW. A ring is a flat, delta-coded list of
    integers in 0.1 km: [x0, y0, dx1, dy1, ...]. The page decodes it back to km."""
    parts = [p for p in getattr(geom, "geoms", [geom]) if isinstance(p, Polygon) and not p.is_empty]
    keep = [p for p in parts if p.area >= min_area] or [max(parts, key=lambda p: p.area)]
    out = []
    for p in keep:
        p = orient(p, 1.0)
        rings = []
        for r in [p.exterior, *[i for i in p.interiors if Polygon(i).area >= min_hole]]:
            c = np.round(np.asarray(r.coords)[:-1] * 10).astype(np.int64)      # integer units of 0.1 km
            if len(c) >= 3:
                d = np.diff(c, axis=0, prepend=np.zeros((1, 2), dtype=np.int64))   # first point absolute, the rest deltas
                rings.append(d.ravel().tolist())
        out.append(rings)
    return out


def anchor(geom) -> list:
    c = geom.centroid
    if not geom.contains(c):
        c = max(getattr(geom, "geoms", [geom]), key=lambda p: p.area).representative_point()
    return [round(c.x, 1), round(c.y, 1)]


def posterior(B, N, prior):
    """Gamma-Poisson shrinkage toward `prior` (a rate). Returns est, lo, hi (rates) and the confidence class 0/1/2."""
    B, N, prior = np.asarray(B, float), np.asarray(N, float), np.asarray(prior, float)
    a, b = K * prior + B, K + N
    est = a / b
    lo, hi = gamma.ppf(0.025, a, scale=1 / b), gamma.ppf(0.975, a, scale=1 / b)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = np.where(lo > 0, hi / lo, np.inf)
    # about 12+ BSF records -> high, 4-11 -> medium, fewer -> low (most units hold none: their value is inherited)
    return est, lo, hi, np.where(ratio <= CONF_HIGH, 0, np.where(ratio <= CONF_MED, 1, 2))


def pack(B, N, est, lo, hi, conf):
    """Per-preset row: [B, N, est, lo, hi, conf] with rates in records per million."""
    return [int(B), int(N), sig3(est * 1e6), sig3(lo * 1e6), sig3(hi * 1e6), int(conf)]


def pack_unit(Bp, Np, estp, lop, hip, confp, i):
    return [pack(Bp[p][i], Np[p][i], estp[p][i], lop[p][i], hip[p][i], confp[p][i]) for p in range(len(PRESETS))]


def main():
    facets = pd.read_csv(PROCESSED / "us_tier_facets.csv")
    lookup = {}
    for (level, window, den, taxon), g in facets.groupby(["level", "window", "den", "taxon"]):
        lookup[(level, window, den, taxon)] = dict(zip(g["gid"], g["count"]))

    def counts(level, gids, p, taxon):
        window, den = PRESETS[p]
        m = lookup[(level, window, den, taxon)]
        return np.array([m.get(g, 0) for g in gids], dtype=float)

    # ---- geometry: counties (simplified together) -> states -> regions -> nation
    l1 = gpd.read_file(GADM_GPKG)
    l1 = l1[(l1["iso3"] == "USA") & l1["name"].isin(STATES)]
    gid1_name = dict(zip(l1["gid1"], l1["name"]))
    cg = gpd.read_file(f"zip://{GADM_USA2}")
    cg = cg[cg["GID_1"].isin(gid1_name) & ~cg["ENGTYPE_2"].str.contains("Water", na=False)].copy()
    cg = cg.to_crs(ALBERS).sort_values("GID_2").reset_index(drop=True)
    try:
        simp = list(shapely.coverage_simplify(np.array(cg.geometry.values), COUNTY_SIMPLIFY_KM))
    except Exception as e:                                     # GADM edges are not always a clean coverage
        print("coverage_simplify failed, falling back:", e)
        simp = [g.simplify(COUNTY_SIMPLIFY_KM, preserve_topology=True) for g in cg.geometry]
    cg["geometry"] = simp
    cg["geometry"] = cg.geometry.make_valid()
    gids1 = sorted(gid1_name, key=lambda g: STATES[gid1_name[g]][0])
    s_index = {g: i for i, g in enumerate(gids1)}

    state_geoms = []
    for g1 in gids1:
        u = unary_union(list(cg.geometry[cg["GID_1"] == g1])).buffer(0.3).buffer(-0.3)
        state_geoms.append(u)
    region_of = [REGIONS.index(STATES[gid1_name[g]][1]) for g in gids1]
    region_geoms = [unary_union([state_geoms[i] for i in range(len(gids1)) if region_of[i] == r]).buffer(0.3).buffer(-0.3)
                    for r in range(len(REGIONS))]
    nation_geom = unary_union(region_geoms).buffer(0.3).buffer(-0.3)
    region_draw = [g.simplify(1.5) for g in region_geoms]       # outlines only, so they can be coarser than the units
    nation_draw = nation_geom.simplify(1.5)

    # ---- state / region / nation estimates per preset
    P = len(PRESETS)
    sB = [counts(1, gids1, p, "bsf") for p in range(P)]
    sN = [counts(1, gids1, p, "insecta") for p in range(P)]
    prior0 = [sB[p].sum() / sN[p].sum() for p in range(P)]
    s_est, s_lo, s_hi, s_conf = zip(*[posterior(sB[p], sN[p], prior0[p]) for p in range(P)])

    def group_unit(idx):
        out = []
        for p in range(P):
            B, N = sB[p][idx].sum(), sN[p][idx].sum()
            e, l, h, c = posterior(B, N, prior0[p])
            out.append(pack(B, N, e, l, h, c))
        return out

    states = []
    for i, g1 in enumerate(gids1):
        abbr, region = STATES[gid1_name[g1]]
        states.append({"abbr": abbr, "name": uncamel(gid1_name[g1]), "region": region_of[i], "anchor": anchor(state_geoms[i]),
                       "polys": polys(state_geoms[i], 20), "d": pack_unit(sB, sN, s_est, s_lo, s_hi, s_conf, i)})
    regions = []
    for r, name in enumerate(REGIONS):
        idx = [i for i in range(len(gids1)) if region_of[i] == r]
        regions.append({"name": name, "n": len(idx), "anchor": anchor(region_geoms[r]), "polys": polys(region_draw[r], 20),
                        "d": group_unit(idx)})
    nation = {"name": "Continental United States", "n": len(gids1), "anchor": anchor(nation_geom),
              "polys": polys(nation_draw, 20), "d": group_unit(list(range(len(gids1))))}

    # ---- counties
    cgid = list(cg["GID_2"])
    cB = [counts(2, cgid, p, "bsf") for p in range(P)]
    cN = [counts(2, cgid, p, "insecta") for p in range(P)]
    c_state = [s_index[g] for g in cg["GID_1"]]
    c_est, c_lo, c_hi, c_conf = [], [], [], []
    for p in range(P):
        prior = np.array([s_est[p][i] for i in c_state])
        e, l, h, c = posterior(cB[p], cN[p], prior)
        c_est.append(e), c_lo.append(l), c_hi.append(h), c_conf.append(c)
    counties = []
    for j, row in cg.iterrows():
        suffix = TYPE_SUFFIX.get(row["ENGTYPE_2"], "")
        counties.append({"n": f"{uncamel(row['NAME_2'])} {suffix}".strip(), "s": c_state[j], "polys": polys(row.geometry, 2),
                         "d": pack_unit(cB, cN, c_est, c_lo, c_hi, c_conf, j)})

    # ---- cells: GBIF z5 density squares (64 x 64 per tile, 2048 around the world), clipped to the US outline
    cells_by_preset = []
    for window, den in PRESETS:
        cells_by_preset.append(pd.read_csv(TIERS / f"cells_{den}_{window}.csv").set_index(["col", "row"])["n"])
    keys = sorted(set().union(*[set(s.index) for s in cells_by_preset]))
    ncell = NCELL
    world = 2 * 20037508.342789244
    size = world / ncell
    to_albers = Transformer.from_crs("EPSG:3857", ALBERS, always_xy=True)
    arr = np.array(keys)
    x0 = -20037508.342789244 + arr[:, 0] * size
    y1 = 20037508.342789244 - arr[:, 1] * size
    squares = []
    for xa, ya in zip(x0, y1):
        xs, ys = to_albers.transform([xa, xa + size, xa + size, xa], [ya, ya, ya - size, ya - size])
        squares.append(Polygon(zip(xs, ys)))
    squares = np.array(squares, dtype=object)
    shapely.prepare(nation_geom)
    inside = shapely.contains(nation_geom, squares)
    clipped = squares.copy()
    for i in np.where(~inside)[0]:
        clipped[i] = shapely.intersection(squares[i], nation_draw).simplify(2.5)   # coastal cells: keep the shape, not every inlet
    area_ok = np.array([(not c.is_empty) and c.area >= 0.06 * s.area for c, s in zip(clipped, squares)])
    keep = np.where(area_ok)[0]
    print(f"cells: {len(keys)} with insect records, {len(keep)} overlap the US outline")
    ccell = [clipped[i] for i in keep]
    ckeys = [keys[i] for i in keep]
    # county for each cell: the county holding its representative point, else the nearest
    tree = STRtree(list(cg.geometry))
    pts = [c.representative_point() for c in ccell]
    hit = tree.query(pts, predicate="within")
    cell_county = np.full(len(ccell), -1)
    cell_county[hit[0]] = hit[1]
    for i in np.where(cell_county < 0)[0]:
        cell_county[i] = tree.nearest(pts[i])
    # numerators from exact BSF points
    bp = pd.read_csv(TIERS / "bsf_points.csv")
    mx, my = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(bp["lon"].values, bp["lat"].values)
    bp["col"] = np.floor((mx + 20037508.342789244) / size).astype(int)
    bp["row"] = np.floor((20037508.342789244 - my) / size).astype(int)
    index = {k: i for i, k in enumerate(ckeys)}
    xB = [np.zeros(len(ckeys)) for _ in range(P)]
    xN = [np.zeros(len(ckeys)) for _ in range(P)]
    for p, (window, den) in enumerate(PRESETS):
        y0, y1_ = map(int, window.split("-"))
        sel = bp[(bp["year"] >= y0) & (bp["year"] <= y1_)]
        if den == "inat":
            sel = sel[sel["dataset"] == INAT]
        for (c, r), n in sel.groupby(["col", "row"]).size().items():
            if (c, r) in index:
                xB[p][index[(c, r)]] = n
        for (c, r), n in cells_by_preset[p].items():
            if (c, r) in index:
                xN[p][index[(c, r)]] = n
        xN[p] = np.maximum(xN[p], xB[p])                       # tiles and points can differ by ~1-2%
    x_est, x_lo, x_hi, x_conf = [], [], [], []
    for p in range(P):
        prior = np.array([c_est[p][j] for j in cell_county])
        e, l, h, c = posterior(xB[p], xN[p], prior)
        x_est.append(e), x_lo.append(l), x_hi.append(h), x_conf.append(c)
    cells = [{"c": int(cell_county[i]), "polys": polys(ccell[i], 0, 0.5), "d": pack_unit(xB, xN, x_est, x_lo, x_hi, x_conf, i)}
             for i in range(len(ccell))]

    # ---- sanity report
    for p, (window, den) in enumerate(PRESETS):
        print(f"preset {p} {window} {den:4s}: states B={sB[p].sum():.0f} N={sN[p].sum():.0f} | counties B={cB[p].sum():.0f} "
              f"N={cN[p].sum():.0f} | cells B={xB[p].sum():.0f} N={xN[p].sum():.0f} ({xN[p].sum() / sN[p].sum():.0%} of state N) "
              f"| cells B>=10: {(xB[p] >= 10).sum()} B>=3: {(xB[p] >= 3).sum()} B>0: {(xB[p] > 0).sum()} "
              f"| conf high/med/low cells {[(x_conf[p] == k).sum() for k in range(3)]} counties {[(c_conf[p] == k).sum() for k in range(3)]}")

    # ---- backdrop and bbox
    ne = gpd.read_file(f"zip://{natural_earth('50m', 'admin_0_countries')}")[["ADM0_A3", "geometry"]]
    ne = ne[ne["ADM0_A3"] != "USA"]
    back = gpd.clip(ne, gpd.GeoSeries([box(-135, 14, -52, 62)], crs=4326)).to_crs(ALBERS)
    backdrop = polys(unary_union(back.geometry.simplify(4)).buffer(0), 300)
    xs = np.concatenate([np.cumsum(np.asarray(r)[0::2]) / 10 for s in states for p in s["polys"] for r in p])
    ys = np.concatenate([np.cumsum(np.asarray(r)[1::2]) / 10 for s in states for p in s["polys"] for r in p])
    data = {"states": states, "regions": regions, "nation": nation, "counties": counties, "cells": cells, "backdrop": backdrop,
            "presets": [{"window": w, "den": d} for w, d in PRESETS], "bbox": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
            "k": K, "cellKm": round(size / 1000 * 0.8, 0), "built": date.today().isoformat()}
    payload = json.dumps(data, separators=(",", ":"))
    DATA_CACHE.write_text(payload, encoding="utf-8")
    out = OUT / "us_prototype.html"
    out.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload), encoding="utf-8")
    nv = lambda items: sum(len(r) // 2 for s in items for p in s["polys"] for r in p)
    print(f"{len(states)} states ({nv(states)} vertices), {len(counties)} counties ({nv(counties)}), {len(cells)} cells ({nv(cells)}); "
          f"{out.stat().st_size / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    if "--template-only" in sys.argv and DATA_CACHE.exists():       # re-embed the last computed data into the current template
        (OUT / "us_prototype.html").write_text(
            TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", DATA_CACHE.read_text(encoding="utf-8")), encoding="utf-8")
        print("template re-embedded")
    else:
        main()
