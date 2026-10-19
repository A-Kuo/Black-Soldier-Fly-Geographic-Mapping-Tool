"""Assemble the static production bundle in web/ (what a static host serves; nothing else is needed at runtime).

  web/index.html          dashboard/gpu/field.html with the CDN scripts replaced by vendored copies, the world dataset
                          as the default, and window.BSF pointing at the data and tile locations
  web/vendor/             maplibre-gl 5.24.0 and pmtiles 4.4.1 (downloaded once from unpkg, then cached)
  web/data/world/         only the files the page requests (see NEEDED), including the per-year timeline files
  web/data/world/world.pmtiles   only with --include-tiles; otherwise host the archive elsewhere and pass --tiles-url

The 190 MB tile archive is the one large file. Static hosts that cap file size, or that you would rather not
bill for tile bandwidth, can serve it from object storage with HTTP Range support and CORS (GET, HEAD, Range header).

Usage
  python bsfl_build_web.py --tiles-url https://<storage-host>/world.pmtiles
  python bsfl_build_web.py --include-tiles          # one self-contained folder, for hosts with no file limit

Reads  dashboard/gpu/field.html, dashboard/gpu/data/world/ (python bsfl_world_field.py, bsfl_world_metrics.py,
       bsfl_world_tiles.py, bsfl_world_timeline.py)
Writes web/
"""

import argparse
import json
import shutil
from pathlib import Path

import requests

from bsfl_sources import ROOT

SRC = ROOT / "dashboard" / "gpu"
DATA = SRC / "data" / "world"
WEB = ROOT / "web"            # --out overrides this
VENDOR = {                       # name -> (url, why)
    "maplibre-gl.js": "https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.js",
    "maplibre-gl.css": "https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css",
    "pmtiles.js": "https://unpkg.com/pmtiles@4.4.1/dist/pmtiles.js",
}
NEEDED = ["anchors.json", "cells.png", "config.json", "grid.json", "ids.png", "meta.json",
          "metrics.json", "timeline.json", "units.json", "units_cells.json"]
OPTIONAL = ["field_backdrop.geojson"]      # the page falls back when these are absent (the world set has none)
FAVICON = ('<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 32 32%27%3E'
           '%3Ccircle cx=%2716%27 cy=%2716%27 r=%2714%27 fill=%27%235b3a9e%27/%3E%3C/svg%3E">')


# Static-site config for a Vercel project whose root directory is this folder. No build step, no framework.
VERCEL = {
    "$schema": "https://openapi.vercel.sh/vercel.json",
    "framework": None,
    "buildCommand": None,
    "cleanUrls": True,
    "headers": [
        {"source": "/(.*)", "headers": [{"key": "X-Content-Type-Options", "value": "nosniff"}]},
        {"source": "/", "headers": [{"key": "Cache-Control", "value": "public, max-age=0, must-revalidate"}]},
        {"source": "/vendor/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=31536000, immutable"}]},
        {"source": "/data/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=3600, stale-while-revalidate=86400"}]},
    ],
}


def vendor_cache():
    cache = ROOT / "data" / "raw" / "vendor"
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in VENDOR.items():
        path = cache / name
        if not path.exists():
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            path.write_bytes(r.content)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-url", help="absolute URL of world.pmtiles when it is hosted outside the bundle")
    ap.add_argument("--include-tiles", action="store_true", help="copy world.pmtiles into the bundle")
    ap.add_argument("--out", help="output folder instead of web/ (for tests)")
    args = ap.parse_args()
    if not args.tiles_url and not args.include_tiles:
        ap.error("give --tiles-url or --include-tiles")

    global WEB
    if args.out:
        WEB = Path(args.out).resolve()
    if WEB.exists():
        shutil.rmtree(WEB)
    (WEB / "vendor").mkdir(parents=True)
    (WEB / "data" / "world" / "timeline").mkdir(parents=True)

    for name, path in ((n, vendor_cache() / n) for n in VENDOR):
        shutil.copy(path, WEB / "vendor" / name)
    for name in NEEDED + [n for n in OPTIONAL if (DATA / n).exists()]:
        shutil.copy(DATA / name, WEB / "data" / "world" / name)
    for f in (DATA / "timeline").iterdir():
        shutil.copy(f, WEB / "data" / "world" / "timeline" / f.name)
    if args.include_tiles:
        shutil.copy(DATA / "world.pmtiles", WEB / "data" / "world" / "world.pmtiles")

    html = (SRC / "field.html").read_text(encoding="utf-8")
    for old, new in (
        ('<link rel="stylesheet" href="https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css">', f'<link rel="stylesheet" href="vendor/maplibre-gl.css">\n{FAVICON}'),
        ('<script src="https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.js"></script>',
         '<script>window.BSF = ' + '{ base: "data/world/"' + (f', tilesUrl: "{args.tiles_url}"' if args.tiles_url else "") + ' };</script>\n<script src="vendor/maplibre-gl.js"></script>'),
        ('<script src="https://unpkg.com/pmtiles@4.4.1/dist/pmtiles.js"></script>', '<script src="vendor/pmtiles.js"></script>'),
    ):
        assert old in html, old
        html = html.replace(old, new)
    assert "unpkg.com" not in html
    (WEB / "index.html").write_text(html, encoding="utf-8")
    if not (DATA / "field_backdrop.geojson").exists():          # keep the console free of an expected 404
        (WEB / "data" / "world" / "field_backdrop.geojson").write_text('{"type":"FeatureCollection","features":[]}')

    (WEB / "vercel.json").write_text(json.dumps(VERCEL, indent=2) + "\n", encoding="utf-8")

    files = [f for f in WEB.rglob("*") if f.is_file()]
    big = sorted(files, key=lambda f: -f.stat().st_size)[:3]
    print(f"web/: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB; largest: " +
          ", ".join(f"{f.relative_to(WEB)} {f.stat().st_size / 1e6:.1f} MB" for f in big))
    print("tiles:", args.tiles_url or "bundled at data/world/world.pmtiles")


if __name__ == "__main__":
    main()
