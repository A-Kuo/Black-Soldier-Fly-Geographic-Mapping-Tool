"""World dataset for the GPU gradient field (dashboard/gpu/field.html?set=world).

Same tier structure and file layout as the US build, so one renderer serves both:
  slot "nation"   = the world (one unit)
  slot "regions"  = World Bank regions (7)
  slot "states"   = countries (Natural Earth 50m, Antarctica included)
  slot "counties" = GADM 4.1 admin-1 units with GBIF records
  slot "cells"    = empty for now (country fallback everywhere)

Estimates per preset (year window) use the US prototype's hierarchical Gamma-Poisson shrinkage, prior strength
K = 1000 insect records: world -> region -> country -> admin-1. A country or admin-1 unit with no insect records
simply inherits its parent's rate and is flagged low confidence.

Reads  data/processed/gbif_country_year.csv, gbif_admin1_year.csv, data/raw/gadm41_admin1.gpkg, countries.csv,
       Natural Earth 50m admin-0 (cached)
Writes dashboard/gpu/data/world/{nation,regions,states,counties,cells}.geojson, ids.png, cells.png, grid.json,
       anchors.json, meta.json, config.json
"""

import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.stats import gamma
from shapely.geometry import mapping
from shapely.ops import unary_union

from bsfl_gpu_export import OUT as US_OUT
from bsfl_sources import PROCESSED, RAW, natural_earth

OUT = US_OUT / "world"
K = 1000
CONF_HIGH, CONF_MED = 3.0, 10.0
WINDOWS = [(2010, 2024), (2018, 2024), (2022, 2024)]
LAT0, LAT1 = -85.0511, 85.0511                 # the full Web-Mercator world, Antarctica and the Arctic Ocean included
GRID_W = 2048                                  # ~19.6 km per pixel at the equator
R = 6378137.0
SIMPLIFY = {"countries": 0.04, "admin1": 0.03, "regions": 0.08, "world": 0.15}   # degrees


def posterior(B, N, prior):
    B, N, prior = (np.asarray(x, float) for x in (B, N, prior))
    a, b = K * prior + B, K + N
    lo, hi = gamma.ppf(0.025, a, scale=1 / b), gamma.ppf(0.975, a, scale=1 / b)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(lo > 0, hi / lo, np.inf)
    return a / b, lo, hi, np.where(ratio <= CONF_HIGH, 0, np.where(ratio <= CONF_MED, 1, 2))


def props(est, lo, hi, conf, B, N) -> dict:
    """Same per-preset property names as bsfl_gpu_export: e (log10 per million), l, h, c, b, n."""
    out = {}
    for p in range(len(WINDOWS)):
        out |= {f"e{p}": round(math.log10(max(est[p] * 1e6, 1e-4)), 3), f"l{p}": float(f"{lo[p] * 1e6:.3g}"),
                f"h{p}": float(f"{hi[p] * 1e6:.3g}"), f"c{p}": int(conf[p]), f"b{p}": int(B[p]), f"n{p}": int(N[p])}
    return out


def anchor(geom) -> list:
    c = geom.centroid
    if not geom.contains(c):
        c = max(getattr(geom, "geoms", [geom]), key=lambda p: p.area).representative_point()
    return [round(c.x, 4), round(c.y, 4)]


def write_fc(name: str, gdf: gpd.GeoDataFrame, prop_rows: list[dict]):
    feats = [{"type": "Feature", "id": i, "properties": {"id": i, **pr}, "geometry": mapping(g)}
             for i, (g, pr) in enumerate(zip(gdf.geometry, prop_rows))]
    path = OUT / f"{name}.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")), encoding="utf-8")
    print(f"{name}: {len(feats)} features, {path.stat().st_size / 1e6:.1f} MB")


def window_sums(df: pd.DataFrame, key: str) -> list[pd.DataFrame]:
    return [df[df["year"].between(a, b)].groupby(key)[["bsf", "insecta"]].sum() for a, b in WINDOWS]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = len(WINDOWS)

    # ---- countries: Natural Earth geometry, World Bank region
    ne = gpd.read_file(f"zip://{natural_earth('50m', 'admin_0_countries')}")
    ne = ne.copy()
    ne["iso3"] = np.where(ne["ISO_A3_EH"] != "-99", ne["ISO_A3_EH"], ne["ADM0_A3"])
    wb = pd.read_csv(RAW / "countries.csv").set_index("iso3")
    # Natural Earth names a few World Bank regions differently; its "Antarctica" region (the continent and sub-Antarctic
    # islands) has no World Bank counterpart, so it becomes a region of its own
    ne_to_wb = {"Middle East & North Africa": "Middle East, North Africa, Afghanistan & Pakistan"}
    ne["region"] = ne["iso3"].map(wb["region"]).fillna(ne["REGION_WB"].replace(ne_to_wb)).str.strip()
    ne["geometry"] = ne.geometry.make_valid()
    ne = ne.dissolve("iso3", aggfunc={"NAME": "first", "region": "first"}).reset_index().sort_values("iso3").reset_index(drop=True)
    regions = sorted(ne["region"].unique())
    ne["rg"] = ne["region"].map({r: i for i, r in enumerate(regions)})
    c_index = {c: i for i, c in enumerate(ne["iso3"])}

    cy = pd.read_csv(PROCESSED / "gbif_country_year.csv").dropna(subset=["iso3"])
    csum = window_sums(cy, "iso3")
    cB = [s["bsf"].reindex(ne["iso3"]).fillna(0).to_numpy() for s in csum]
    cN = [s["insecta"].reindex(ne["iso3"]).fillna(0).to_numpy() for s in csum]

    # ---- hierarchy: world -> regions -> countries
    world_rate = [cB[p].sum() / cN[p].sum() for p in range(P)]
    rB = [np.array([cB[p][ne["rg"] == r].sum() for r in range(len(regions))]) for p in range(P)]
    rN = [np.array([cN[p][ne["rg"] == r].sum() for r in range(len(regions))]) for p in range(P)]
    reg = [posterior(rB[p], rN[p], world_rate[p]) for p in range(P)]
    cty = [posterior(cB[p], cN[p], reg[p][0][ne["rg"].to_numpy()]) for p in range(P)]
    wld = [posterior(cB[p].sum(), cN[p].sum(), world_rate[p]) for p in range(P)]

    # ---- admin-1: GADM units with GBIF records, shrunk toward their country
    a1 = gpd.read_file(RAW / "gadm41_admin1.gpkg")
    a1 = a1[a1["iso3"].isin(c_index)].sort_values("gid1").reset_index(drop=True)
    a1["st"] = a1["iso3"].map(c_index)
    ay = pd.read_csv(PROCESSED / "gbif_admin1_year.csv")
    asum = window_sums(ay, "gid1")
    aB = [s["bsf"].reindex(a1["gid1"]).fillna(0).to_numpy() for s in asum]
    aN = [s["insecta"].reindex(a1["gid1"]).fillna(0).to_numpy() for s in asum]
    adm = [posterior(aB[p], aN[p], cty[p][0][a1["st"].to_numpy()]) for p in range(P)]

    # ---- geometry
    a1["geometry"] = a1.geometry.make_valid().simplify(SIMPLIFY["admin1"], preserve_topology=True)
    reg_geom = gpd.GeoSeries([unary_union(ne.geometry[ne["rg"] == r]) for r in range(len(regions))], crs=4326)
    world_geom = gpd.GeoSeries([unary_union(reg_geom)], crs=4326)
    ne_s = ne.copy(); ne_s["geometry"] = ne.geometry.simplify(SIMPLIFY["countries"], preserve_topology=True)

    pick = lambda res, i: [[r[k][i] for r in res] for k in range(4)]          # per preset -> per unit
    write_fc("nation", world_geom.simplify(SIMPLIFY["world"]).to_frame("geometry"),
             [{"nm": "World", **props(*[[w[k] for w in wld] for k in range(4)], [b.sum() for b in cB], [n.sum() for n in cN])}])
    write_fc("regions", reg_geom.simplify(SIMPLIFY["regions"]).to_frame("geometry"),
             [{"nm": r, **props(*pick(reg, i), [b[i] for b in rB], [n[i] for n in rN])} for i, r in enumerate(regions)])
    write_fc("states", ne_s, [{"nm": row.NAME, "ab": row.iso3, "rg": int(row.rg), **props(*pick(cty, i), [b[i] for b in cB], [n[i] for n in cN])}
                              for i, row in enumerate(ne.itertuples())])
    write_fc("counties", a1, [{"nm": f"{row.name}, {row.country}", "gid": row.gid1, "st": int(row.st), **props(*pick(adm, i), [b[i] for b in aB], [n[i] for n in aN])}
                              for i, row in enumerate(a1.itertuples())])
    (OUT / "cells.geojson").write_text('{"type":"FeatureCollection","features":[]}')

    # ---- ID rasters on a global Web-Mercator grid
    merc = lambda lat: R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    x0, x1, y0, y1 = -math.pi * R, math.pi * R, merc(LAT0), merc(LAT1)
    W = GRID_W; H = round(W * (y1 - y0) / (x1 - x0))
    tr = from_bounds(x0, y0, x1, y1, W, H)
    def burn(gdf):
        g = gdf.set_crs(4326, allow_override=True).clip_by_rect(-180, LAT0, 180, LAT1).to_crs(3857)
        return rasterize(((geom, i + 1) for i, geom in enumerate(g) if not geom.is_empty), out_shape=(H, W), transform=tr, fill=0, dtype="uint32")
    st, co = burn(ne.geometry), burn(a1.geometry)
    assert st.max() < 256 and co.max() < 65536
    ids = np.zeros((H, W, 4), np.uint8)
    ids[..., 0], ids[..., 1], ids[..., 2], ids[..., 3] = st, co >> 8, co & 255, 255
    Image.fromarray(ids, "RGBA").save(OUT / "ids.png", optimize=True)
    cells = np.zeros((H, W, 4), np.uint8); cells[..., 3] = 255
    Image.fromarray(cells, "RGBA").save(OUT / "cells.png", optimize=True)
    mx = lambda x: x / (2 * math.pi * R) + 0.5
    my = lambda y: 0.5 - y / (2 * math.pi * R)
    px_km = (x1 - x0) / W / 1000
    (OUT / "grid.json").write_text(json.dumps({"w": W, "h": H, "pxKm": round(px_km, 2), "x0": mx(x0), "x1": mx(x1), "y0": my(y1), "y1": my(y0)}))

    anchors = {"nation": [anchor(world_geom[0])], "regions": [anchor(g) for g in reg_geom],
               "states": [anchor(g) for g in ne.geometry], "counties": [anchor(g) for g in a1.geometry]}
    (OUT / "anchors.json").write_text(json.dumps(anchors, separators=(",", ":")))
    (OUT / "meta.json").write_text(json.dumps({"presets": [{"window": f"{a}-{b}", "den": "all"} for a, b in WINDOWS], "k": K}))
    (OUT / "config.json").write_text(json.dumps({
        "title": "Black Soldier Fly record rate, world",
        "bounds": [[-170, -72], [180, 85]],
        "tierNames": ["World", "Regions", "Countries", "Admin-1", "Cells"],
        "nationLabel": "World",
        # tier boundaries as zoom offsets from the fit-to-bounds zoom, and blur widths in km (equator)
        "bound": [-0.6, 0.6, 2.2, 9], "sigma": {"k": 60, "c": 60, "s": 160, "rLight": 300, "rBlur": 900, "n": 3000},
        "hmax": [900000, 900000, 600000, 150000, 50000],
    }, indent=1))

    for p, (a, b) in enumerate(WINDOWS):
        print(f"{a}-{b}: world {world_rate[p] * 1e6:.0f}/M | regions " +
              ", ".join(f"{r.split(' ')[0]} {reg[p][0][i] * 1e6:.0f}" for i, r in enumerate(regions)) +
              f" | countries conf h/m/l {[(cty[p][3] == k).sum() for k in range(3)]} | admin-1 conf {[(adm[p][3] == k).sum() for k in range(3)]}")
    print(f"grid {W}x{H} ({px_km:.1f} km at equator); countries on grid {len(np.unique(st)) - 1}/{len(ne)}, admin-1 {len(np.unique(co)) - 1}/{len(a1)}")


if __name__ == "__main__":
    main()
