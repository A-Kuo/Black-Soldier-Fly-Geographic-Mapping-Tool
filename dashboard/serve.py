"""Local static server with HTTP Range support, for PMTiles (python -m http.server ignores Range).

    python dashboard/serve.py [port] [directory]   default: port 8812, the dashboard/ folder
    then open http://localhost:<port>/gpu/field.html?set=world        (or / when serving the web/ bundle)

Production hosting (Vercel static files / Vercel Blob) answers Range requests itself; this is only for development.
"""

import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(__file__).parent


class RangeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")           # lets a second dev origin play the tile host
        self.send_header("Access-Control-Expose-Headers", "ETag, Content-Range, Content-Length, Accept-Ranges")
        self.send_header("Cache-Control", "max-age=600")    # both maps (display + ID pass) share fetched tiles
        super().end_headers()

    def send_head(self):
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        start = int(m[1]) if m[1] else max(0, size - int(m[2] or 0))
        end = min(int(m[2]), size - 1) if m[1] and m[2] else size - 1
        if start >= size or start > end:
            self.send_response(416); self.send_header("Content-Range", f"bytes */{size}"); self.end_headers()
            return None
        f = open(path, "rb"); f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(65536, remaining))
            if not chunk:
                break
            outputfile.write(chunk); remaining -= len(chunk)
        self._remaining = None


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8812
    print(f"serving {ROOT} on http://localhost:{port}")
    ThreadingHTTPServer(("", port), RangeHandler).serve_forever()
