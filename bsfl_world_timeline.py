"""Timeline steps for the world map (dashboard/gpu/field.html?set=world).

For every span (cumulative since 2010, 3-year window, single year) and every end year 2010-2025, runs the same
hierarchical Gamma-Poisson shrinkage as bsfl_world_field.py (K = 1000 insect records; world -> region -> country ->
admin-1 -> admin-2) and packs all five tiers into one gzip file, so the page can fetch one small file per slider step.

Step file layout (little-endian), n units in tier order nation, regions, countries, admin-1, admin-2:
  Uint32 x n   BSF records            Uint32 x n   insect records
  Int16  x n   log10 rate per million, x1000 (posterior mean)
  Int16  x n   same, 2.5 % bound      Int16  x n   same, 97.5 % bound
  Uint8  x n   confidence class (0 high, 1 medium, 2 low)

Climate change comes from ERA5 annual country means (World Bank CCKP): the mean of the step's window minus the
1961-1990 mean, for temperature (C), precipitation (% of baseline) and relative humidity (points). Admin-1 and admin-2
inherit their country's value in the page.

Antarctica (the continent and sub-Antarctic islands, its own region) has no records and BSF cannot breed there, so
it is pinned at the plot floor with high confidence instead of inheriting the world rate.

Reads  dashboard/gpu/data/world/{units.json, units_cells.json, states.geojson}, data/processed/gbif_{country,admin1,admin2}_year.csv
Writes dashboard/gpu/data/world/timeline.json, timeline/<span>_<year>.bin.gz, timeline/climate.json,
       data/raw/cckp_era5_ts_<var>.csv
"""

import gzip
import json
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from bsfl_sources import PROCESSED, RAW
from bsfl_world_field import K, OUT, posterior

YEARS = list(range(2010, 2026))
SPANS = [("cum", "Cumulative since 2010", lambda y: (2010, y)),
         ("w3", "3-year window", lambda y: (max(2010, y - 2), y)),
         ("yr", "Single year", lambda y: (y, y))]
FLOOR = 1e-4                                           # per million; same plot floor as the static build
BASELINE = (1961, 1990)
CCKP_TS = "https://cckpapi.worldbank.org/cckp/v1/era5-x0.25_timeseries_{var}_timeseries_annual_1950-2025_mean_historical_era5_x0.25_mean/{codes}"
CLIMATE = {"temp": "tas", "pr": "pr", "hum": "hurs"}
EQUAL_AREA = "EPSG:6933"
SESSION = requests.Session()


def era5_series(var: str, iso3: list[str]) -> pd.DataFrame:
    path = RAW / f"cckp_era5_ts_{var}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    out = {}
    for i in range(0, len(iso3), 40):
        r = SESSION.get(CCKP_TS.format(var=var, codes=",".join(iso3[i:i + 40])), params={"_format": "json"}, timeout=180)
        r.raise_for_status()
        data = r.json()["data"]
        for code, series in (data.items() if isinstance(data, dict) else []):
            out[code] = {int(k[:4]): v for k, v in series.items()}
        time.sleep(0.3)
    df = pd.DataFrame(out).T.sort_index(axis=1)
    df.to_csv(path)
    return df


def pivot(df: pd.DataFrame, key: str, index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """(units x years) BSF and insect matrices for YEARS, aligned to `index`."""
    d = df[df["year"].isin(YEARS)]
    out = []
    for col in ("bsf", "insecta"):
        m = d.pivot_table(index=key, columns="year", values=col, aggfunc="sum").reindex(index=index, columns=YEARS).fillna(0)
        out.append(m.to_numpy(float))
    return out[0], out[1]


def window(m: np.ndarray, a: int, b: int) -> np.ndarray:
    return m[:, YEARS.index(a):YEARS.index(b) + 1].sum(1)


def scaled(x):
    return np.clip(np.round(np.log10(np.maximum(np.asarray(x, float) * 1e6, FLOOR)) * 1000), -4000, 32000).astype(np.int16)


def main():
    t0 = time.time()
    (OUT / "timeline").mkdir(exist_ok=True)
    units = json.loads((OUT / "units.json").read_text(encoding="utf-8"))
    cells = json.loads((OUT / "units_cells.json").read_text(encoding="utf-8"))
    R, S, C, Kn = len(units["regions"]), len(units["states"]), len(units["counties"]), len(cells)
    iso = [u["ab"] for u in units["states"]]
    rg = np.array([u["rg"] for u in units["states"]])
    c_st = np.array([u["st"] for u in units["counties"]])
    k_ct = np.array([u["ct"] for u in cells]); k_st = np.array([u["st"] for u in cells])
    anta = np.array([u["nm"] == "Antarctica" for u in units["regions"]])
    n = 1 + R + S + C + Kn
    off = {"nation": 0, "regions": 1, "states": 1 + R, "counties": 1 + R + S, "cells": 1 + R + S + C}

    cB, cN = pivot(pd.read_csv(PROCESSED / "gbif_country_year.csv").dropna(subset=["iso3"]), "iso3", pd.Index(iso))
    aB, aN = pivot(pd.read_csv(PROCESSED / "gbif_admin1_year.csv"), "gid1", pd.Index([u["gid"] for u in units["counties"]]))
    kB, kN = pivot(pd.read_csv(PROCESSED / "gbif_admin2_year.csv"), "gid2", pd.Index([u["gid"] for u in cells]))
    print(f"counts: countries {cB.sum():.0f}/{cN.sum():.0f}, admin-1 {aB.sum():.0f}/{aN.sum():.0f}, admin-2 {kB.sum():.0f}/{kN.sum():.0f} (BSF/insects)", flush=True)

    # ---- one hierarchy per (span, end year)
    for key, label, win in SPANS:
        for y in YEARS:
            a, b = win(y)
            B = [window(m, a, b) for m in (cB, aB, kB)]
            N = [window(m, a, b) for m in (cN, aN, kN)]
            world = cB[:, YEARS.index(a):YEARS.index(b) + 1].sum() / max(cN[:, YEARS.index(a):YEARS.index(b) + 1].sum(), 1)
            rB = np.bincount(rg, weights=B[0], minlength=R); rN = np.bincount(rg, weights=N[0], minlength=R)
            res = {"nation": posterior(B[0].sum(), N[0].sum(), world)}
            res["regions"] = posterior(rB, rN, world)
            res["states"] = posterior(B[0], N[0], res["regions"][0][rg])
            res["counties"] = posterior(B[1], N[1], res["states"][0][c_st])
            prior_k = np.where(k_ct >= 0, res["counties"][0][np.maximum(k_ct, 0)], res["states"][0][k_st])
            res["cells"] = posterior(B[2], N[2], prior_k)
            est, lo, hi, conf = (np.concatenate([np.atleast_1d(res[t][i]) for t in off]) for i in range(4))
            Bs = np.concatenate([[B[0].sum()], rB, B[0], B[1], B[2]]).astype(np.uint32)
            Ns = np.concatenate([[N[0].sum()], rN, N[0], N[1], N[2]]).astype(np.uint32)
            # Antarctica: no records and not viable, so zero with high confidence
            pin = np.zeros(n, bool)
            pin[off["regions"]:off["regions"] + R] = anta
            pin[off["states"]:off["states"] + S] = anta[rg]
            pin[off["counties"]:off["counties"] + C] = anta[rg][c_st]
            pin[off["cells"]:] = anta[rg][k_st]
            est, lo, hi, conf = np.where(pin, 0, est), np.where(pin, 0, lo), np.where(pin, 0, hi), np.where(pin, 0, conf)
            blob = b"".join(x.astype("<" + x.dtype.str[1:]).tobytes() for x in
                            (Bs, Ns, scaled(est), scaled(lo), scaled(hi), conf.astype(np.uint8)))
            (OUT / "timeline" / f"{key}_{y}.bin.gz").write_bytes(gzip.compress(blob, 9))
        print(f"  {key}: {len(YEARS)} steps, {time.time() - t0:.0f}s", flush=True)

    # ---- climate change by country (ERA5), rolled up to regions and the world by land area
    states = gpd.read_file(OUT / "states.geojson").sort_values("id").set_crs(4326, allow_override=True)
    area = states.to_crs(EQUAL_AREA).area.to_numpy() / 1e6
    series = {k: era5_series(v, iso) for k, v in CLIMATE.items()}
    cc = {}
    for key, label, win in SPANS:
        cc[key] = {}
        for y in YEARS:
            a, b = win(y)
            step = {}
            for var, df in series.items():
                base = df[[c for c in df.columns if BASELINE[0] <= int(c) <= BASELINE[1]]].mean(1)
                cur = df[[c for c in df.columns if a <= int(c) <= b]].mean(1)
                delta = (cur - base) if var != "pr" else (cur / base - 1) * 100
                v = delta.reindex(iso).to_numpy(float)
                w = np.where(np.isnan(v), 0, area)
                reg = np.bincount(rg, weights=np.nan_to_num(v) * w, minlength=R) / np.maximum(np.bincount(rg, weights=w, minlength=R), 1e-9)
                reg = np.where(np.bincount(rg, weights=w, minlength=R) > 0, reg, np.nan)
                world = np.nansum(np.nan_to_num(v) * w) / w.sum()
                step[var] = [None if not np.isfinite(x) else round(float(x), 2) for x in np.concatenate([[world], reg, v])]
            cc[key][y] = step
    pooled = {var: np.array([x for key in cc for y in cc[key] for x in cc[key][y][var][1 + R:] if x is not None]) for var in CLIMATE}
    stats = {var: {"lo": float(np.percentile(a, 5)), "hi": float(np.percentile(a, 95)),
                   "terciles": [float(np.percentile(a, 33.3)), float(np.percentile(a, 66.7))]} for var, a in pooled.items()}
    (OUT / "timeline" / "climate.json").write_text(json.dumps({"baseline": f"{BASELINE[0]}-{BASELINE[1]}", "stats": stats, "steps": cc},
                                                               separators=(",", ":")), encoding="utf-8")
    print("climate change stats:", {k: [round(x, 2) for x in (v["lo"], v["hi"])] for k, v in stats.items()})

    (OUT / "timeline.json").write_text(json.dumps({
        "years": YEARS, "spans": [{"key": k, "label": l} for k, l, _ in SPANS], "n": n, "k": K,
        "tiers": {t: [o, o + c] for (t, o), c in zip(off.items(), (1, R, S, C, Kn))}, "baseline": f"{BASELINE[0]}-{BASELINE[1]}"}))
    size = sum(f.stat().st_size for f in (OUT / "timeline").glob("*.bin.gz")) / 1e6
    print(f"{len(SPANS) * len(YEARS)} step files, {size:.1f} MB total, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
