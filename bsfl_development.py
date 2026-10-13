"""BSF record rate against economic development, nationally and within countries.

The outcome is BSF records with insect records as exposure (offset), so every coefficient is on the record
*rate*, net of observation effort. Models are Poisson pseudo-maximum-likelihood (PPML) with robust standard
errors (clustered by country for admin-1 units), which tolerates the heavy overdispersion; a negative-binomial
fit is reported alongside for the national models. Development is log10 GDP per capita PPP, so exp(coef) is the
rate ratio for a 10x richer unit.

The within-country model adds country fixed effects: it asks whether richer regions of the same country record
BSF at a different rate, which nets out national differences in iNaturalist uptake and museum collecting.

Reads  data/processed/gbif_admin1_year.csv, admin1_units.csv, admin1_gdp_year.csv, gbif_country_year.csv,
       bsf_climate_zones.csv; data/raw/insecta_country_year.csv, kummu_tabulated_adm0_*.csv, UNDP HDR csv
Writes data/processed/development_models.csv   one row per model: rate ratio per 10x GDP, 95% CI, n
       data/processed/development_bins.csv     descriptive bins for the dashboard
       data/processed/admin1_window.csv        gid1 window totals joined to GDP, zone (dashboard + models)
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

from bsfl_admin1 import panel as admin1_panel
from bsfl_regional_estimates import load_panel
from bsfl_sources import HTTP, PROCESSED, RAW, read_codes

UNDP = "https://hdr.undp.org/sites/default/files/2023-24_HDR/HDR23-24_Composite_indices_complete_time_series.csv"
KUMMU_ADM0 = "https://zenodo.org/records/18429133/files/tabulated_adm0_gdp_perCapita.csv?download=1"


def cached_csv(url: str, name: str) -> pd.DataFrame:
    path = RAW / name
    if not path.exists():
        r = HTTP.get(url, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    return read_codes(path, encoding="latin-1")


def geo_mean(df: pd.DataFrame, key: str, col: str) -> pd.Series:
    return np.exp(np.log(df[col].where(df[col] > 0)).groupby(df[key]).mean())


# ---------- data ----------

def admin1_window(start: int, end: int) -> pd.DataFrame:
    p = admin1_panel().query("@start <= year <= @end")
    w = p.groupby(["gid1", "iso3"], as_index=False)[["bsf", "bsf_inat", "insecta"]].sum()
    units = read_codes(PROCESSED / "admin1_units.csv")
    gdp = read_codes(PROCESSED / "admin1_gdp_year.csv").query("@start <= year <= @end")
    w = w.merge(units.drop(columns="iso3"), on="gid1", how="left")
    w["gdp_pc"] = w["gid1"].map(geo_mean(gdp, "gid1", "gdp_pc"))
    w.to_csv(PROCESSED / "admin1_window.csv", index=False)
    return w


def national_window(start: int, end: int) -> pd.DataFrame:
    p = load_panel().query("@start <= year <= @end")
    bsf = p.groupby("iso3", as_index=False)[["bsf", "bsf_from_inat"]].sum()
    ins = read_codes(RAW / "insecta_country_year.csv").query("@start <= year <= @end")
    ref = read_codes(RAW / "countries.csv")[["iso2", "iso3", "region", "income"]]
    ins = ins.merge(ref, on="iso2").groupby("iso3", as_index=False)["insecta"].sum()
    df = ins.merge(bsf, on="iso3", how="left").fillna({"bsf": 0, "bsf_from_inat": 0})

    k0 = cached_csv(KUMMU_ADM0, "kummu_tabulated_adm0_gdp_perCapita.csv")
    yrs = [str(y) for y in range(start, end + 1) if str(y) in k0.columns]
    k0 = k0.melt(id_vars="iso3", value_vars=yrs, var_name="year", value_name="gdp_pc")
    df["gdp_pc"] = df["iso3"].map(geo_mean(k0, "iso3", "gdp_pc"))

    hdr = cached_csv(UNDP, "undp_hdr_composite_time_series.csv")
    hcols = [f"hdi_{y}" for y in range(start, end + 1) if f"hdi_{y}" in hdr.columns]
    df["hdi"] = df["iso3"].map(hdr.set_index("iso3")[hcols].mean(axis=1))

    zones = read_codes(PROCESSED / "bsf_climate_zones.csv")[["iso3", "temp_band", "zone"]]
    return df.merge(zones, on="iso3", how="left").merge(ref[["iso3", "region", "income"]], on="iso3", how="left")


# ---------- models ----------

def design(df: pd.DataFrame, x: str, fe: list[str]) -> pd.DataFrame:
    X = df[[x]].copy()
    if x == "log_gdp_sq":
        X.insert(0, "log_gdp", df["log_gdp"])
    for f in fe:
        X = X.join(pd.get_dummies(df[f], prefix=f, drop_first=True, dtype=float))
    return sm.add_constant(X)


def drop_all_zero(df: pd.DataFrame, fe: list[str]) -> pd.DataFrame:
    """Rows in a fixed-effect group with zero BSF carry no information and make PPML diverge."""
    for f in fe:
        df = df[df.groupby(f)["bsf"].transform("sum") > 0]
    return df


def ppml(df: pd.DataFrame, x: str, fe: list[str], cluster: str | None) -> dict:
    df = drop_all_zero(df.dropna(subset=[x, *fe]).query("insecta > 0"), fe)
    X = design(df, x, fe)
    m = sm.GLM(df["bsf"], X, family=sm.families.Poisson(), offset=np.log(df["insecta"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df[cluster])[0]}) if cluster else m.fit(cov_type="HC1")
    return summarise("PPML", df, fit, x)


def summarise(estimator: str, df: pd.DataFrame, fit, x: str) -> dict:
    r = {"estimator": estimator, "n_units": len(df), "n_countries": df["iso3"].nunique(), "bsf": int(df["bsf"].sum()),
         "coef": fit.params[x], "se": fit.bse[x], "p": fit.pvalues[x]}
    if x == "log_gdp_sq":  # peak of the rate curve, in GDP per capita
        b1, b2 = fit.params["log_gdp"], fit.params[x]
        r["turning_point_gdp"] = 10 ** (-b1 / (2 * b2)) if b2 < 0 else np.nan
    return r


def negbin(df: pd.DataFrame, x: str, fe: list[str]) -> dict:
    df = drop_all_zero(df.dropna(subset=[x, *fe]).query("insecta > 0"), fe)
    X = design(df, x, fe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.NegativeBinomial(df["bsf"], X, exposure=df["insecta"]).fit(disp=0, maxiter=500)
    return summarise("NegBin", df, fit, x)


def run_models(nat: pd.DataFrame, a1: pd.DataFrame) -> pd.DataFrame:
    nat = nat.assign(log_gdp=np.log10(nat["gdp_pc"]), hdi10=nat["hdi"] * 10)
    nat["log_gdp_sq"] = nat["log_gdp"] ** 2
    a1 = a1.assign(log_gdp=np.log10(a1["gdp_pc"]))
    a1["log_gdp_sq"] = a1["log_gdp"] ** 2
    sub = a1[a1["gdp_level"] == "subnational"]
    specs = [
        ("national", "GDP per capita, squared term", "temperature band", lambda: ppml(nat, "log_gdp_sq", ["temp_band"], None)),
        ("national", "GDP per capita, squared term", "temperature band", lambda: negbin(nat, "log_gdp_sq", ["temp_band"])),
        ("admin-1", "GDP per capita, squared term", "climate zone", lambda: ppml(sub, "log_gdp_sq", ["zone"], "iso3")),
        ("admin-1 within country", "GDP per capita, squared term", "climate zone + country",
         lambda: ppml(sub, "log_gdp_sq", ["zone", "iso3"], "iso3")),
        ("national", "GDP per capita", "none", lambda: ppml(nat, "log_gdp", [], None)),
        ("national", "GDP per capita", "none", lambda: negbin(nat, "log_gdp", [])),
        ("national", "GDP per capita", "temperature band", lambda: ppml(nat, "log_gdp", ["temp_band"], None)),
        ("national", "GDP per capita", "temperature band", lambda: negbin(nat, "log_gdp", ["temp_band"])),
        ("national", "GDP per capita", "temperature band, excl. USA",
         lambda: ppml(nat[nat.iso3 != "USA"], "log_gdp", ["temp_band"], None)),
        ("national", "HDI (per 0.1)", "temperature band", lambda: ppml(nat, "hdi10", ["temp_band"], None)),
        ("admin-1", "GDP per capita", "climate zone", lambda: ppml(sub, "log_gdp", ["zone"], "iso3")),
        ("admin-1 within country", "GDP per capita", "climate zone + country",
         lambda: ppml(sub, "log_gdp", ["zone", "iso3"], "iso3")),
        ("admin-1 within country", "GDP per capita", "climate zone + country, excl. USA",
         lambda: ppml(sub[sub.iso3 != "USA"], "log_gdp", ["zone", "iso3"], "iso3")),
    ]
    rows = []
    for level, measure, controls, fn in specs:
        r = fn()
        r.update(level=level, measure=measure, controls=controls)
        rows.append(r)
    out = pd.DataFrame(rows)
    out["rate_ratio"] = np.exp(out["coef"])
    out["rr_lo95"] = np.exp(out["coef"] - 1.96 * out["se"])
    out["rr_hi95"] = np.exp(out["coef"] + 1.96 * out["se"])
    cols = ["level", "measure", "controls", "estimator", "n_units", "n_countries", "bsf", "rate_ratio", "rr_lo95", "rr_hi95",
            "p", "turning_point_gdp"]
    return out.reindex(columns=cols)


# ---------- descriptive bins ----------

def poisson_ci(k, scale):
    lo = np.where(k > 0, chi2.ppf(0.025, 2 * k) / 2, 0) * scale
    hi = chi2.ppf(0.975, 2 * k + 2) / 2 * scale
    return lo, hi


def bins(a1: pd.DataFrame, nat: pd.DataFrame) -> pd.DataFrame:
    out = []
    # 1. pooled rate by GDP quintile, admin-1 units with sub-national GDP
    sub = a1[(a1["gdp_level"] == "subnational") & (a1["insecta"] > 0)].copy()
    sub["q"] = pd.qcut(np.log10(sub["gdp_pc"]), 5, labels=False)
    for q, g in sub.groupby("q"):
        B, N = g["bsf"].sum(), g["insecta"].sum()
        lo, hi = poisson_ci(B, 1e6 / N)
        out.append({"kind": "admin1_gdp_quintile", "bin": int(q) + 1, "gdp_lo": g["gdp_pc"].min(), "gdp_hi": g["gdp_pc"].max(),
                    "units": len(g), "bsf": B, "insecta": N, "value": B / N * 1e6, "lo": lo, "hi": hi})

    # 2. within-country: observed / expected BSF, expected = the country's pooled rate x the unit's insect records
    sub["country_rate"] = sub.groupby("iso3")["bsf"].transform("sum") / sub.groupby("iso3")["insecta"].transform("sum")
    sub["expected"] = sub["country_rate"] * sub["insecta"]
    sub["rel_gdp"] = np.log10(sub["gdp_pc"]) - np.log10(sub["gdp_pc"]).groupby(sub["iso3"]).transform("mean")
    edges = [-np.inf, -0.15, -0.05, 0.05, 0.15, np.inf]
    labels = ["<0.71x", "0.71-0.89x", "0.89-1.12x", "1.12-1.41x", ">1.41x"]
    sub["rb"] = pd.cut(sub["rel_gdp"], edges, labels=labels)
    for i, (lab, g) in enumerate(sub[sub["expected"] > 0].groupby("rb", observed=True)):
        B, E = g["bsf"].sum(), g["expected"].sum()
        lo, hi = poisson_ci(B, 1 / E)
        out.append({"kind": "within_country_rel_gdp", "bin": i + 1, "label": lab, "units": len(g), "bsf": B,
                    "expected": E, "value": B / E, "lo": lo, "hi": hi})

    # 3. national: effort and rate by income group
    for inc, g in nat[nat["insecta"] > 0].groupby("income"):
        B, N = g["bsf"].sum(), g["insecta"].sum()
        lo, hi = poisson_ci(B, 1e6 / N)
        out.append({"kind": "national_income_group", "label": inc, "units": len(g), "bsf": B, "insecta": N,
                    "inat_share": g["bsf_from_inat"].sum() / max(B, 1), "value": B / N * 1e6, "lo": lo, "hi": hi})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2010)
    ap.add_argument("--end", type=int, default=2024, help="last year with GDP data")
    args = ap.parse_args()

    a1 = admin1_window(args.start, args.end)
    nat = national_window(args.start, args.end)
    models = run_models(nat, a1)
    models.round(4).to_csv(PROCESSED / "development_models.csv", index=False)
    b = bins(a1, nat)
    b.round(4).to_csv(PROCESSED / "development_bins.csv", index=False)

    pd.set_option("display.width", 220)
    print(models.round(3).to_string(index=False))
    print()
    print(b.drop(columns=[c for c in ("gdp_lo", "gdp_hi") if c in b]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
