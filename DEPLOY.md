# Deploying the world map (static site on Vercel)

The map is a static site: no server code and no build step on the host. `bsfl_build_web.py` assembles it into `web/`.
Nothing here has been deployed or uploaded; these are the steps for when the Vercel project exists.

## What is in `web/`

| Path | Size | Notes |
|---|---|---|
| `index.html` | 78 KB | `dashboard/gpu/field.html` with the CDN scripts swapped for the vendored copies and `window.BSF` set |
| `vendor/` | 1.2 MB | MapLibre GL 5.24.0 and pmtiles 4.4.1, so the page makes no third-party requests |
| `data/world/*.json`, `*.png`, `timeline/` | 13 MB | unit properties, ID rasters, metrics, per-year timeline steps (loaded as needed) |
| `vercel.json` | 1 KB | no framework, no build command, cache headers (`vendor/` immutable, `data/` 1 h + stale-while-revalidate) |
| `data/world/world.pmtiles` | **58 MB** | vector tiles (max zoom 7, overzoomed beyond); committed and deployed with the site |

The whole site is **72 MB**, under Vercel Hobby's 100 MB deployment cap and GitHub's 100 MB per-file limit, so the
tiles are committed in `web/` and served from the same origin (Vercel serves static files with HTTP Range, which PMTiles
needs). Build it with:

```bash
python bsfl_build_web.py --include-tiles
```

Earlier builds left `world.pmtiles` out of git (it was 193 MB at max zoom 10), so the deployed page had no vector tiles:
no extruded prisms, no borders and an empty map when zoomed in. If the archive ever grows past ~90 MB again, host it in
object storage that answers Range requests and sends CORS headers (Vercel Blob, Cloudflare R2, S3) and build with
`--tiles-url https://<store>/world.pmtiles` instead.

## Deploy

1. Build: `python bsfl_build_web.py --include-tiles` (needs `dashboard/gpu/data/world/`; regenerate with
   `bsfl_world_field.py`, `bsfl_world_metrics.py`, `bsfl_world_tiles.py`, `bsfl_world_timeline.py`).
2. Either run `vercel deploy --prod` inside `web/` (uploads local files, no git needed), or create a project from the repo
   with **Root Directory = `web`**, Framework Preset = Other, and commit `web/` (tiles included).

## Tested locally

Page on one origin and tiles on another, with every non-localhost request blocked: 17 tile requests answered `206` by
the tile host, 0 HTTP errors, 0 external requests, detail view drawn. `python dashboard/serve.py <port> <folder>` is the
local stand-in (it adds Range and CORS); `vercel.json` and Vercel's own range and CORS behaviour are **not** tested.

## Known gaps before launch

- Vercel behaviour above (headers, `cleanUrls`, `.bin.gz` served without `Content-Encoding`) is untested until deployed.
- The North Pole cap is not drawn yet; see `HANDOFF_NORTH_POLE.md`.
- `field_backdrop.geojson` is an empty placeholder in the world set.
