"""Inline the birdsong_3d_points.json into the HTML to bypass file:// CORS.

Browsers block fetch() from file:// origins. Embedding the JSON as a JS
variable makes the HTML fully self-contained — works on double-click.
"""
import pathlib, json

ROOT = pathlib.Path(r"D:/Bird Song/viz")
HTML_SRC = ROOT / "birdsong_3d.html"
JSON_SRC = ROOT / "birdsong_3d_points.json"
HTML_OUT = ROOT / "birdsong_3d_inline.html"

payload = json.loads(JSON_SRC.read_text(encoding="utf-8"))
html = HTML_SRC.read_text(encoding="utf-8")

# Replace the fetch() block with inlined data
old = "fetch('birdsong_3d_points.json').then(r => r.json()).then(data => {"
new = ("const data = " + json.dumps(payload, separators=(',', ':')) + ";\n"
       "(function(){")
html = html.replace(old, new)
# Close the outer wrapper instead of the .then() callback
html = html.replace("});\n\nfunction applyColors", "})();\n\nfunction applyColors")

HTML_OUT.write_text(html, encoding="utf-8")
print(f"-> {HTML_OUT}  ({HTML_OUT.stat().st_size:,} bytes)")
print(f"Open: file:///{str(HTML_OUT).replace(chr(92), '/')}")
