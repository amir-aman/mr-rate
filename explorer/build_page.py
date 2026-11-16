"""Build docs/index.html as one self-contained file (works from disk, in previews and on GitHub Pages)."""

import base64
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
here = root / "explorer"
subprocess.run(
    [
        "npx", "esbuild", "explorer/src/app.js", "--bundle", "--minify", "--format=iife",
        "--target=es2020", "--loader:.gz=base64", "--loader:.png=dataurl",
        "--outfile=explorer/build/app.js", "--log-level=warning",
    ],
    check=True,
    cwd=root,
)
html = (here / "src" / "index.html").read_text(encoding="utf-8")
js = (here / "build" / "app.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
fonts = {
    "FONT_DISPLAY": "bricolage-grotesque-latin-wght-normal.woff2",
    "FONT_BODY": "instrument-sans-latin-wght-normal.woff2",
}
for token, name in fonts.items():
    assert html.count(token) == 1, token
    data = base64.b64encode((here / "fonts" / name).read_bytes()).decode("ascii")
    html = html.replace(token, "data:font/woff2;base64," + data)
assert html.count("<!--APP-->") == 1
html = html.replace("<!--APP-->", "<script>" + js + "</script>")
out = root / "docs" / "index.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
