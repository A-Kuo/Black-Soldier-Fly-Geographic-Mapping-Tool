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
| `data/world/world.pmtiles` | **193 MB** | only with `--include-tiles`; see below |

Without the tiles the site is **14.2 MB**.

## Where the tiles go

Vercel's limits page says a CLI deployment is capped at **100 MB on Hobby** and 1 GB on Pro, so the 193 MB archive does
not belong in the deployment on Hobby. Host it in object storage that answers HTTP **Range** requests and sends CORS
headers, then bake its URL into the page:

```bash
python bsfl_build_web.py --tiles-url https://<your-store>/world.pmtiles
```

Vercel Blob (public store) documents range requests and CDN caching up to 512 MB per object. **It does not document CORS for
public blob URLs, so check it once the store exists** (replace the URL and the origin):

```bash
curl -s -D - -o /dev/null -H "Range: bytes=0-15" -H "Origin: https://<your-site>.vercel.app" https://<your-store>/world.pmtiles
```

Expect `206 Partial Content`, a `Content-Range` header and `Access-Control-Allow-Origin` (`*` or your site). If CORS is
missing, serve the archive from Cloudflare R2 or S3 with a CORS rule allowing `GET, HEAD` and the `Range` header, or use
`--include-tiles` on a Pro plan.

## Deploy

1. Build: `python bsfl_build_web.py --tiles-url <url>` (needs `dashboard/gpu/data/world/`; regenerate with
   `bsfl_world_field.py`, `bsfl_world_metrics.py`, `bsfl_world_tiles.py`, `bsfl_world_timeline.py`).
2. Upload `dashboard/gpu/data/world/world.pmtiles` to the tile host.
3. Either run `vercel deploy --prod` inside `web/` (uploads local files, no git needed), or create a project from the repo
   with **Root Directory = `web`**, Framework Preset = Other, and commit `web/` (the archive is git-ignored).

## Tested locally

Page on one origin and tiles on another, with every non-localhost request blocked: 17 tile requests answered `206` by
the tile host, 0 HTTP errors, 0 external requests, detail view drawn. `python dashboard/serve.py <port> <folder>` is the
local stand-in (it adds Range and CORS); `vercel.json` and Vercel's own range and CORS behaviour are **not** tested.

## Known gaps before launch

- Vercel behaviour above (headers, `cleanUrls`, `.bin.gz` served without `Content-Encoding`) is untested until deployed.
- The North Pole cap is not drawn yet; see `HANDOFF_NORTH_POLE.md`.
- `field_backdrop.geojson` is an empty placeholder in the world set.
