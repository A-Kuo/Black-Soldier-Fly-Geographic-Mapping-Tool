"""Offline unit-ID rasters and bubble anchors for the WebGL gradient field.

The browser never rasterises polygons. Each tier is burned once into an ID raster on a Web-Mercator grid; the GPU then
joins per-unit values onto those IDs (a texture lookup), blurs and blends them. Switching presets or, later, joining a
climate or development metric only uploads a new per-unit value array.

  ids.png    R = state id (1-based, 0 outside), G/B = county id (1-based, 16 bit)
  cells.png  R/G = cell id (1-based, 16 bit)
  grid.json  grid size and its bounds in MapLibre MercatorCoordinate units (0..1), pixel size in km
  anchors.json  lon/lat anchor per nation, region, state and county, for the bubble layer
  field_outside.geojson   everything outside the US (drawn over the field so its edge follows the vector coastline)
  field_backdrop.geojson  neighbouring countries (Natural Earth 50m), drawn as plain land

Reads  dashboard/gpu/data/{nation,regions,states,counties,cells}.geojson   (python bsfl_gpu_export.py)
Writes dashboard/gpu/data/{ids.png, cells.png, grid.json, anchors.json, field_outside.geojson, field_backdrop.geojson}
"""

import json
import math

import geopandas as gpd
import numpy as np
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box, mapping
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from bsfl_gpu_export import OUT
from bsfl_sources import natural_earth

LON0, LON1, LAT0, LAT1 = -126.0, -65.5, 23.5, 50.5
PX_KM = 3.0                                   # ground size of a pixel at the grid's mid-latitude
R = 6378137.0


def merc(lon, lat):
    return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def anchor(geom) -> list:
    """Centroid if it falls inside the unit, else a point inside its largest part (same rule as the Canvas build)."""
    c = geom.centroid
    if not geom.contains(c):
        c = max(getattr(geom, "geoms", [geom]), key=lambda p: p.area).representative_point()
    return [round(c.x, 4), round(c.y, 4)]


def burn(gdf, shape, transform) -> np.ndarray:
    return rasterize(((g, i + 1) for i, g in enumerate(gdf.geometry)), out_shape=shape, transform=transform,
                     fill=0, dtype="uint32")


def main():
    x0, y0 = merc(LON0, LAT0)
    x1, y1 = merc(LON1, LAT1)
    m_per_px = PX_KM * 1000 / math.cos(math.radians((LAT0 + LAT1) / 2))
    W, H = round((x1 - x0) / m_per_px), round((y1 - y0) / m_per_px)
    transform = from_bounds(x0, y0, x1, y1, W, H)

    tiers = {k: gpd.read_file(OUT / f"{k}.geojson").sort_values("id").reset_index(drop=True)
             for k in ("nation", "regions", "states", "counties", "cells")}
    m = {k: g.to_crs(3857) for k, g in tiers.items()}
    st, co, ce = burn(m["states"], (H, W), transform), burn(m["counties"], (H, W), transform), burn(m["cells"], (H, W), transform)
    assert co.max() < 65536 and ce.max() < 65536

    ids = np.zeros((H, W, 4), np.uint8)
    ids[..., 0], ids[..., 1], ids[..., 2], ids[..., 3] = st, co >> 8, co & 255, 255
    cells = np.zeros((H, W, 4), np.uint8)
    cells[..., 0], cells[..., 1], cells[..., 3] = ce >> 8, ce & 255, 255
    Image.fromarray(ids, "RGBA").save(OUT / "ids.png", optimize=True)
    Image.fromarray(cells, "RGBA").save(OUT / "cells.png", optimize=True)

    mx = lambda x: x / (2 * math.pi * R) + 0.5            # web-mercator metres -> MercatorCoordinate
    my = lambda y: 0.5 - y / (2 * math.pi * R)
    grid = {"w": W, "h": H, "pxKm": PX_KM, "x0": mx(x0), "x1": mx(x1), "y0": my(y1), "y1": my(y0)}   # y0 = top edge
    (OUT / "grid.json").write_text(json.dumps(grid))

    anchors = {k: [anchor(g) for g in tiers[k].geometry] for k in ("nation", "regions", "states", "counties")}
    (OUT / "anchors.json").write_text(json.dumps(anchors, separators=(",", ":")))

    nation = unary_union(tiers["nation"].geometry.make_valid())
    # RFC 7946 winding (outer CCW, holes CW): MapLibre fills a hole wound like its outer ring as a separate polygon
    outside = orient(box(-179.9, -84, 179.9, 84).difference(nation.buffer(0)), 1.0)
    (OUT / "field_outside.geojson").write_text(json.dumps({"type": "Feature", "properties": {}, "geometry": mapping(outside)}))
    ne = gpd.read_file(f"zip://{natural_earth('50m', 'admin_0_countries')}")[["ADM0_A3", "geometry"]]
    back = gpd.clip(ne[ne["ADM0_A3"] != "USA"], box(-140, 10, -50, 65))
    (OUT / "field_backdrop.geojson").write_text(back.geometry.simplify(0.02).to_json())

    inside = st > 0
    print(f"grid {W}x{H} at {PX_KM} km; {inside.mean():.0%} of pixels inside the US; "
          f"states {len(np.unique(st)) - 1}/{len(tiers['states'])}, counties {len(np.unique(co)) - 1}/{len(tiers['counties'])}, "
          f"cells {len(np.unique(ce)) - 1}/{len(tiers['cells'])} hit at least one pixel")


if __name__ == "__main__":
    main()
