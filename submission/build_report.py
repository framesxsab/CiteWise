from pathlib import Path
import re
import subprocess

from markdown_it import MarkdownIt

root = Path(__file__).resolve().parents[1]
source = (root / "docs" / "FINAL_REPORT.md").read_text(encoding="utf-8")
chunks = re.split(r"(?m)^<!-- PAGE \d+ -->\s*$", source)
pages = [chunks[0] + chunks[1], *chunks[2:]]
if len(pages) != 10:
    raise SystemExit(f"Expected 10 report pages, found {len(pages)}")
md = MarkdownIt("default", {"html": True}).enable("table")
sections = "\n".join(f'<section class="page">{md.render(p)}</section>' for p in pages)
html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>CiteWise Final Report</title>
<style>
@page {{ size: Letter; margin: 0; }}
* {{ box-sizing: border-box; }}
html,body {{ margin:0;padding:0;color:#17212b;font:9pt/1.31 Georgia,"Times New Roman",serif; }}
.page {{ width:8.5in;height:11in;padding:.55in .62in;overflow:hidden;page-break-after:always; }}
.page:last-child {{ page-break-after:auto; }}
h1 {{ font-size:19pt;line-height:1.15;margin:0 0 12pt;color:#142b3a; }}
h2 {{ font-size:14pt;line-height:1.18;margin:0 0 10pt;color:#142b3a; }}
h3 {{ font-size:10pt;margin:8pt 0 3pt; }}
p {{ margin:0 0 7pt; }} ul,ol {{ margin:0 0 8pt;padding-left:20pt; }} li {{ margin:0 0 3pt; }}
table {{ border-collapse:collapse;width:100%;margin:7pt 0 9pt;font:7.5pt/1.2 Arial,sans-serif; }}
th,td {{ border-bottom:1px solid #aab6bf;padding:4px 5px;text-align:left;vertical-align:top; }}
th {{ border-top:1.5px solid #48606e;background:#f1f4f5; }}
code {{ font:7.7pt Consolas,monospace;overflow-wrap:anywhere; }}
img {{ max-width:100%;max-height:3.1in;object-fit:contain; }} figure {{ margin:5pt 0; }}
figcaption {{ font:7.5pt/1.2 Arial,sans-serif;margin-top:3pt; }}
.screenshot-pair {{ display:flex;gap:12pt;align-items:flex-start; }}
.screenshot-pair figure {{ width:50%; }} .screenshot-pair img {{ max-height:3.8in; }}
a {{ color:#174c68;text-decoration:none; }}
</style></head><body>{sections}</body></html>'''
out = root / "submission"
(out / "CiteWise_Final_Report.html").write_text(html, encoding="utf-8")
subprocess.run(["node", str(out / "render_report.mjs")], cwd=root, check=True)
