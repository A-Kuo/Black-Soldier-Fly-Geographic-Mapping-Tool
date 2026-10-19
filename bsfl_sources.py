"""Fetchers for every candidate data source. Each returns a tidy DataFrame and caches raw output in data/raw."""

import io
import json
import warnings
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore", message="Numeric taxonomy keys", category=DeprecationWarning)

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
for d in (RAW, PROCESSED):
    d.mkdir(parents=True, exist_ok=True)

BSF_NAME = "Hermetia illucens"
INSECTA_KEY = 216
INAT_GBIF_DATASET = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"

WDI_INDICATORS = {
    "AG.LND.FRST.ZS": "forest_pct_land",
    "AG.LND.AGRI.ZS": "agri_pct_land",
    "SP.URB.TOTL.IN.ZS": "urban_pct_pop",
    # EN.ATM.CO2E.PC was archived by the World Bank; this is its replacement.
    "EN.GHG.CO2.PC.CE.AR5": "co2_t_per_capita",
}

OWID_SERIES = {
    "co-emissions-per-capita": "owid_co2_t_per_capita",
    "tree-cover-loss": "owid_tree_cover_loss_ha",
}

CCKP_SERIES = {
    "tas": "cru-x0.5_timeseries_tas_timeseries_annual_1901-2022_mean_historical_cru_ts4.07_mean",
    "pr": "cru-x0.5_timeseries_pr_timeseries_annual_1901-2022_mean_historical_cru_ts4.07_mean",
}


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "BSFL-Mapping/0.1 (research)"
    retry = Retry(total=4, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    return s


HTTP = session()


def get_json(url: str, **params):
    r = HTTP.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def read_codes(path, **kw) -> pd.DataFrame:
    """read_csv that keeps Namibia's ISO2 code "NA" as a string instead of NaN."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""], **kw)


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(RAW / f"{name}.csv", index=False)
    return df


# ---------- reference ----------

def countries() -> pd.DataFrame:
    """World Bank country list: the ISO2 <-> ISO3 bridge between GBIF (ISO2) and everything else (ISO3)."""
    j = get_json("https://api.worldbank.org/v2/country", format="json", per_page=400)[1]
    df = pd.DataFrame(
        {
            "iso3": c["id"],
            "iso2": c["iso2Code"],
            "name": c["name"].strip(),
            "region": c["region"]["value"].strip(),
            "income": c["incomeLevel"]["value"].strip(),
            "lat": pd.to_numeric(c["latitude"] or None),
            "lon": pd.to_numeric(c["longitude"] or None),
        }
        for c in j
        if c["region"]["value"] != "Aggregates"
    )
    return save(df, "countries")


# ---------- species ----------

def bsf_taxon_key() -> int:
    """Integer GBIF backbone key; the occurrence and maps APIs still index on it."""
    return get_json("https://api.gbif.org/v1/species/match", name=BSF_NAME)["usageKey"]


def gbif_maps_capabilities(taxon_key: int, **filters) -> dict:
    return get_json(
        "https://api.gbif.org/v2/map/occurrence/density/capabilities.json", taxonKey=taxon_key, **filters
    )


def inat_summary() -> dict:
    base = "https://api.inaturalist.org/v1/observations"
    out = {}
    for grade in ("research", "needs_id", "casual"):
        out[grade] = get_json(base, taxon_name=BSF_NAME, quality_grade=grade, per_page=0)["total_results"]
    hist = get_json(f"{base}/histogram", taxon_name=BSF_NAME, interval="year", quality_grade="research")
    out["by_year"] = {k[:4]: v for k, v in hist["results"]["year"].items()}
    (RAW / "inat_summary.json").write_text(json.dumps(out, indent=2))
    return out


# ---------- climate ----------

def cckp(var: str, iso3: list[str]) -> pd.DataFrame:
    rows = []
    for i in range(0, len(iso3), 40):
        chunk = ",".join(iso3[i : i + 40])
        data = get_json(f"https://cckpapi.worldbank.org/cckp/v1/{CCKP_SERIES[var]}/{chunk}", _format="json")["data"]
        for code, series in data.items():
            rows += [{"iso3": code, "year": int(k[:4]), var: v} for k, v in series.items()]
    return save(pd.DataFrame(rows), f"cckp_{var}")


def open_meteo_archive(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    j = get_json(
        "https://archive-api.open-meteo.com/v1/archive",
        latitude=lat, longitude=lon, start_date=start, end_date=end,
        daily="temperature_2m_mean,precipitation_sum",
    )
    return pd.DataFrame(j["daily"]).assign(lat=lat, lon=lon, elevation=j.get("elevation"))


def open_meteo_climate(lat: float, lon: float, start="1950-01-01", end="2050-12-31", model="MRI_AGCM3_2_S") -> pd.DataFrame:
    j = get_json(
        "https://climate-api.open-meteo.com/v1/climate",
        latitude=lat, longitude=lon, start_date=start, end_date=end, models=model,
        daily="temperature_2m_mean,precipitation_sum",
    )
    return pd.DataFrame(j["daily"]).assign(lat=lat, lon=lon, model=model)


def nasa_power(lat: float, lon: float, start=2001, end=2023) -> pd.DataFrame:
    j = get_json(
        "https://power.larc.nasa.gov/api/temporal/monthly/point",
        parameters="T2M,PRECTOTCORR", community="AG", latitude=lat, longitude=lon,
        start=start, end=end, format="JSON",
    )
    return pd.DataFrame(j["properties"]["parameter"])


# ---------- human development / damage ----------

def wdi(indicator: str, years="2000:2024") -> pd.DataFrame:
    meta, rows = get_json(
        f"https://api.worldbank.org/v2/country/all/indicator/{indicator}", format="json", date=years, per_page=20000
    )
    df = pd.DataFrame(
        {"iso3": r["countryiso3code"], "year": int(r["date"]), WDI_INDICATORS.get(indicator, indicator): r["value"]}
        for r in rows
        if r["countryiso3code"]
    )
    return save(df, f"wdi_{indicator}")


def owid(slug: str) -> pd.DataFrame:
    r = HTTP.get(f"https://ourworldindata.org/grapher/{slug}.csv", timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    value_col = [c for c in df.columns if c not in ("Entity", "Code", "Year") and "Annotations" not in c][0]
    df = df.rename(columns={"Code": "iso3", "Year": "year", value_col: OWID_SERIES.get(slug, slug)})
    df = df[df["iso3"].notna() & ~df["iso3"].astype(str).str.startswith("OWID")]
    return save(df[["iso3", "year", OWID_SERIES.get(slug, slug)]], f"owid_{slug}")


def hansen_tiles(layer="lossyear", version="GFC-2023-v1.11") -> list[str]:
    r = HTTP.get(f"https://storage.googleapis.com/earthenginepartners-hansen/{version}/{layer}.txt", timeout=60)
    r.raise_for_status()
    return r.text.split()


def eonet(category="wildfires", start_year=2016, end_year=2025) -> pd.DataFrame:
    """Pulled one year at a time because a single call silently caps at the most recent events."""
    rows = []
    for year in range(start_year, end_year + 1):
        j = get_json(
            "https://eonet.gsfc.nasa.gov/api/v3/events",
            category=category, start=f"{year}-01-01", end=f"{year}-12-31", status="all",
        )
        for e in j["events"]:
            for g in e["geometry"]:
                if g["type"] == "Point":
                    rows.append({"id": e["id"], "title": e["title"], "date": g["date"], "lon": g["coordinates"][0], "lat": g["coordinates"][1]})
    return save(pd.DataFrame(rows).drop_duplicates(), f"eonet_{category}")


def faostat_bulk(name="Production_Crops_Livestock_E_All_Data_(Normalized).zip") -> Path:
    path = RAW / name
    if not path.exists():
        r = HTTP.get(f"https://bulks-faostat.fao.org/production/{name}", timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def gfw_datasets() -> list:
    return get_json("https://data-api.globalforestwatch.org/datasets")["data"]


def natural_earth(scale="110m", layer="admin_0_countries") -> Path:
    path = RAW / f"ne_{scale}_{layer}.zip"
    if not path.exists():
        r = HTTP.get(f"https://naciscdn.org/naturalearth/{scale}/cultural/ne_{scale}_{layer}.zip", timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path
