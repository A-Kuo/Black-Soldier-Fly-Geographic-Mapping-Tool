# Handoff: draw the North Pole (and South Pole) on the GPU map

Status: **not done, handed to the next agent.** Everything else on `dashboard/gpu/field.html?set=world` works.

## The problem
The map is Web Mercator (MapLibre 5). Mercator stops at 85.0511 N / S, so 90 N cannot be drawn as a point and the
polar caps look stretched. Today the world data covers the full Mercator range (Antarctica and the Arctic Ocean
included, ids raster 2048 x 2048), the initial view reaches 85 N, and `drawLabels()` in `field.html` adds a text label
"North Pole, 90N (beyond the Mercator edge at 85N)" at lat 84.3. That label is a placeholder, not a drawing.

## What the user wants
"Incorporate Antarctica and the North Pole into the web GPU UI." Antarctica is in (own region, pinned to zero
with high confidence). The North Pole should be visible as a place on the map.

## Options to evaluate
1. **Polar caps as extra geometry**: add a small inset or an extended strip above 85 N (equirectangular or azimuthal)
   drawn as another custom layer, filled with the Arctic value (0 records). Cheapest; keeps Mercator everywhere else.
2. **Globe projection**: MapLibre 5 supports `projection: { type: "globe" }`. The custom WebGL layers (`field`, `dots`
   in `field.html`) use `opt.modelViewProjectionMatrix` with world-pixel mercator coordinates and would need the
   globe shader path (`opt.shaderData`, `defaultProjectionData`), and the endless horizontal copies (world wrap)
   are replaced by the sphere. Biggest change, best result.
3. **Polar inset map** (second small map in a corner, north-polar stereographic). Simple, but a second map to keep in sync.

## Where things live
- `dashboard/gpu/field.html`: UI, shaders, custom layers. Wrap logic: `worldCopy()`, loops over copies in `field.render`
  and `dots.render`, detail pass shifts `D.box` by the copy index.
- `bsfl_world_field.py`: LAT0/LAT1 (+-85.0511), ids/grid rasters, Antarctica as its own region.
- `bsfl_world_tiles.py`: PMTiles (tippecanoe in WSL). `bsfl_world_timeline.py`: per-year steps, climate change, Antarctica pin.
- Serve with `python dashboard/serve.py 8813`, open `http://localhost:8813/gpu/field.html?set=world`.

## Gotchas learned
- The Cursor browser tab renders only while visible: call `Emulation.setFocusEmulationEnabled {enabled: true}` via CDP
  first, and take two screenshots (the first is often one frame stale).
- Generated data (`dashboard/gpu/data/`, ~200 MB PMTiles) is git-ignored; rebuild with the scripts above.
