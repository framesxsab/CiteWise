"""Automated source checks. Does not replace axe or assistive-technology testing."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def contrast_ratio(foreground, background):
    def luminance(color):
        color = color.lstrip("#")
        if len(color) == 3:
            color = "".join(x * 2 for x in color)
        values = [int(color[i:i+2], 16) / 255 for i in (0, 2, 4)]
        linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055)**2.4 for v in values]
        return sum(a*b for a,b in zip(linear, (.2126, .7152, .0722)))
    a, b = sorted((luminance(foreground), luminance(background)))
    return (b + .05) / (a + .05)


def audit(path):
    html = Path(path).read_text(encoding="utf-8")
    page = Page()
    page.feed(html)
    elements = page.elements
    labels = {a.get("for") for t,a in elements if t == "label"}
    controls = [(t,a) for t,a in elements if t in ("select", "textarea", "input")]
    missing_labels = [a.get("id") for _,a in controls if a.get("id") not in labels and not a.get("aria-label")]
    checks = {
        "document_language": any(t == "html" and a.get("lang") for t,a in elements),
        "viewport_meta": any(t == "meta" and a.get("name") == "viewport" for t,a in elements),
        "document_title": bool(re.search(r"<title>\s*[^<]+</title>", html)),
        "one_main_landmark": sum(t == "main" for t,_ in elements) == 1,
        "one_primary_heading": sum(t == "h1" for t,_ in elements) == 1,
        "form_controls_have_labels": not missing_labels,
        "dynamic_status_announced": any(a.get("role") == "status" and a.get("aria-live") for _,a in elements),
        "visible_focus_style": ":focus-visible" in html,
        "no_positive_tabindex": all(int(a.get("tabindex", "0")) <= 0 for _,a in elements),
    }
    styles = {}
    sheet = re.search(r"<style>(.*?)</style>", html, re.S)
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", sheet.group(1) if sheet else ""):
        declarations = dict(part.split(":", 1) for part in body.split(";") if ":" in part)
        declarations = {k.strip(): v.strip() for k,v in declarations.items()}
        for selector in selectors.split(","):
            styles[selector.strip()] = declarations
    def color(selector, prop):
        value = styles.get(selector, {}).get(prop)
        return {"white": "#ffffff", "black": "#000000"}.get(value, value)
    background = color(":root", "background")
    # Declared palette only: cascades, visibility and computed styles require browser checks.
    pairs = [("body", color(":root", "color"), background), ("muted", color(".muted", "color"), background),
             ("metadata", color(".meta", "color"), color(".evidence", "background")),
             ("button", color("button", "color"), color("button", "background")),
             ("error", color("#status.error", "color"), background), ("eyebrow", color(".key", "color"), background)]
    contrast = [{"role": name, "foreground": a, "background": b,
                 "ratio": contrast_ratio(a,b) if a and b else None,
                 "passes_4_5": bool(a and b and contrast_ratio(a,b) >= 4.5)} for name,a,b in pairs]
    return {"scope": "source structure and declared palette only", "checks": checks,
            "missing_labels": missing_labels, "contrast": contrast,
            "passed": all(checks.values()) and all(c["passes_4_5"] for c in contrast),
            "browser_axe_status": "Not performed by this source-level audit",
            "unverified": ["computed styles", "screen-reader announcements", "current keyboard behavior", "mobile overflow", "browser axe rules"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(Path(__file__).with_name("index.html")))
    parser.add_argument("--output", default="artifacts/phase3/accessibility_source.json")
    args = parser.parse_args()
    result = audit(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
