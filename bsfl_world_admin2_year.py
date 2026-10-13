"""Per-year GBIF counts at GADM level 2, for the timeline slider (dashboard/gpu/field.html?set=world).

One facet pull (GADM_LEVEL_2_GID, paged by 1000) per taxon and year, cached one file per (taxon, year) in
data/raw/gbif_admin2_year/, so an interrupted run resumes.

Writes data/processed/gbif_admin2_year.csv  (gid2, year, bsf, insecta)
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from bsfl_sources import PROCESSED, RAW

YEARS = range(2010, 2026)
GBIF = "https://api.gbif.org/v1/occurrence/search"
CLEAN = dict(hasCoordinate="true", hasGeospatialIssue="false", occurrenceStatus="PRESENT")
TAXA = {"bsf": 1577153, "insecta": 216}
CACHE = RAW / "gbif_admin2_year"
SESSION = requests.Session()


def facet(taxon: int, year: int) -> dict[str, int]:
    out, off = {}, 0
    while True:
        r = None
        for attempt in range(8):
            try:
                r = SESSION.get(GBIF, params={**CLEAN, "taxonKey": taxon, "year": year, "limit": 0,
                                              "facet": "GADM_LEVEL_2_GID", "facetLimit": 1000, "facetOffset": off}, timeout=240)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(min(60, 2 ** attempt))
        else:
            raise RuntimeError(f"GBIF gave up: taxon {taxon} year {year} offset {off}")
        counts = r.json()["facets"][0]["counts"] if r.json()["facets"] else []
        out.update({c["name"]: c["count"] for c in counts})
        if len(counts) < 1000:
            return out
        off += 1000


def pull(job):
    name, year = job
    path = CACHE / f"{name}_{year}.json"
    if path.exists():
        return job, None
    t0 = time.time()
    d = facet(TAXA[name], year)
    path.write_text(json.dumps(d), encoding="utf-8")
    return job, (len(d), time.time() - t0)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [(n, y) for y in YEARS for n in TAXA]
    with ThreadPoolExecutor(3) as ex:
        for (name, year), info in ex.map(pull, jobs):
            if info:
                print(f"{name} {year}: {info[0]} units, {info[1]:.0f}s", flush=True)
    rows = []
    for y in YEARS:
        b = json.loads((CACHE / f"bsf_{y}.json").read_text())
        i = json.loads((CACHE / f"insecta_{y}.json").read_text())
        rows += [{"gid2": g, "year": y, "bsf": b.get(g, 0), "insecta": i.get(g, 0)} for g in set(b) | set(i)]
    pd.DataFrame(rows).to_csv(PROCESSED / "gbif_admin2_year.csv", index=False)
    print(f"gbif_admin2_year.csv: {len(rows)} rows")


if __name__ == "__main__":
    main()
