"""Write one standalone HTML page with the plots of all plot configs.

The page embeds all images, so it needs no other files. For each plot config
it has a switch between the annotated plot, the plain plot, the memory plot
(CPU plots only) and the annotated plot without axis cuts.

    python -m spotter.gallery --output gallery.html
"""

import argparse
import base64
import html
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

# the order on the page; other plot configs follow in alphabetical order
ORDER = [
    "plot-full-reco-chain",
    "plot-trackfinding",
    "plot-ckf",
    "plot-seeding",
    "plot-clusterization",
    "plot-spacepointformation",
    "plot-ambiguity",
    "plot-efficiency",
]


def render(config: Path, directory: Path, data_dir: Path) -> dict[str, bytes]:
    """Render one plot config with and without axis cuts; return the PNG images."""
    with open(config) as f:
        output = yaml.safe_load(f).get("output", "cpu")
    for name, mode in (("cuts", "config"), ("nocuts", "off")):
        subprocess.run(
            [sys.executable, "-m", "spotter.plotter", "--config", str(config),
             "--data-dir", str(data_dir), "--plots-dir", str(directory / name), "--axis-cuts", mode],
            check=True, capture_output=True,
        )
    variants = {
        "Annotated": directory / "cuts" / f"{output}_annotated.png",
        "Plain": directory / "cuts" / f"{output}.png",
        "Memory": directory / "cuts" / "mem.png",
        "No axis cuts": directory / "nocuts" / f"{output}_annotated.png",
    }
    return {name: path.read_bytes() for name, path in variants.items() if path.exists()}


def last_nightly(data_dir: Path) -> str:
    """Return the date of the newest downloaded log."""
    dates = [pd.read_csv(p)["date"].max() for p in (data_dir / "raw").glob("*_logs.csv")]
    return max(dates) if dates else "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--configs", type=Path, nargs="+", help="the plot configs (default: config/plot-*.yaml)")
    parser.add_argument("--data-dir", type=Path, default="data")
    parser.add_argument("--output", type=Path, default="gallery.html")
    args = parser.parse_args()

    configs = args.configs or sorted(
        Path("config").glob("plot-*.yaml"),
        key=lambda p: (ORDER.index(p.stem) if p.stem in ORDER else len(ORDER), p.stem),
    )
    sections = []
    with tempfile.TemporaryDirectory() as tmp:
        for config in configs:
            with open(config) as f:
                cfg = yaml.safe_load(f)
            images = render(config, Path(tmp) / config.stem, args.data_dir)
            print(f"{config}: {', '.join(images)}")
            sections.append((config, cfg, images))

    nav = []
    body = []
    for config, cfg, images in sections:
        anchor = config.stem.removeprefix("plot-")
        title = html.escape(cfg.get("title", config.stem))
        nav.append(f'<li><a href="#{anchor}">{title}</a></li>')
        buttons = "".join(
            f'<button type="button" id="{anchor}-{i}" aria-pressed="{str(i == 0).lower()}" '
            f'data-src="data:image/png;base64,{base64.b64encode(png).decode()}">{html.escape(name)}</button>'
            for i, (name, png) in enumerate(images.items())
        )
        lines = ", ".join(html.escape(f["label"]) for f in cfg["flavors"].values())
        note = html.escape(cfg.get("description", ""))
        body.append(f'''
<section id="{anchor}">
  <header>
    <h2>{title}</h2>
    <p class="meta"><code>{config}</code><span>{len(cfg.get("axis_cuts", []))} axis cuts</span></p>
  </header>
  <p class="note">{note}</p>
  <p class="lines">{lines}</p>
  <div class="switch" role="group" aria-label="Plot variant">{buttons}</div>
  <figure><img src="" alt="{title}"></figure>
</section>''')

    sub = (
        "ITk tracking CPU and efficiency from the ATLAS SPOT nightlies, "
        f"up to {last_nightly(args.data_dir)} (generated {date.today()})"
    )
    page = (
        TEMPLATE.replace("{{NAV}}", "\n".join(nav))
        .replace("{{SECTIONS}}", "\n".join(body))
        .replace("{{SUB}}", html.escape(sub))
    )
    args.output.write_text(page)
    print(f"{args.output}: {args.output.stat().st_size / 1e6:.1f} MB")


TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SPOT Tracking Plots</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
:root {
  --ground: #f6f7f9;
  --surface: #ffffff;
  --ink: #15181d;
  --muted: #5c6370;
  --rule: #dde1e8;
  --accent: #2a78d6;
  --accent-ink: #ffffff;
  --frame: #ffffff;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #111418;
    --surface: #181c22;
    --ink: #e7eaef;
    --muted: #9aa2b0;
    --rule: #2a3039;
    --accent: #5598e7;
    --accent-ink: #0b1220;
    --frame: #ffffff;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --ground: #111418;
  --surface: #181c22;
  --ink: #e7eaef;
  --muted: #9aa2b0;
  --rule: #2a3039;
  --accent: #5598e7;
  --accent-ink: #0b1220;
  --frame: #ffffff;
  color-scheme: dark;
}
body {
  background: var(--ground);
  color: var(--ink);
  font: 15px/1.55 "IBM Plex Sans", system-ui, sans-serif;
}
.page {
  display: grid;
  grid-template-columns: 220px minmax(0, 1fr);
  gap: 40px;
  max-width: 1400px;
  margin: 0 auto;
  padding-inline: 24px;
  padding-block: 32px 64px;
}
nav { position: sticky; top: calc(env(safe-area-inset-top, 0px) + 24px); align-self: start; }
nav h1 {
  font: 600 22px/1.2 "IBM Plex Sans Condensed", sans-serif;
  margin: 0 0 6px;
  text-wrap: balance;
}
nav .sub { color: var(--muted); font-size: 13px; margin: 0 0 20px; }
nav ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; }
nav a {
  display: block; padding: 5px 10px; border-radius: 6px;
  color: var(--ink); text-decoration: none; font-size: 14px;
}
nav a:hover, nav a:focus-visible { background: var(--surface); outline: none; box-shadow: inset 0 0 0 1px var(--rule); }
main { display: grid; gap: 56px; min-width: 0; }
.intro {
  display: grid; gap: 8px; padding-bottom: 8px; border-bottom: 1px solid var(--rule);
  max-width: 72ch;
}
.intro p { margin: 0; color: var(--muted); }
.intro strong { color: var(--ink); font-weight: 500; }
section { display: grid; gap: 10px; min-width: 0; scroll-margin-top: 24px; }
section header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 4px 16px; }
h2 {
  font: 600 24px/1.2 "IBM Plex Sans Condensed", sans-serif;
  margin: 0; text-wrap: balance;
}
.meta { display: flex; gap: 14px; margin: 0; color: var(--muted); font-size: 13px; }
code { font: 13px "IBM Plex Mono", ui-monospace, monospace; }
.note { margin: 0; max-width: 72ch; }
.lines { margin: 0; color: var(--muted); font-size: 13px; letter-spacing: 0.01em; }
.switch { display: flex; flex-wrap: wrap; gap: 6px; }
.switch button {
  font: 500 13px "IBM Plex Sans", sans-serif;
  padding: 6px 12px; border-radius: 999px; cursor: pointer;
  border: 1px solid var(--rule); background: var(--surface); color: var(--ink);
}
.switch button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }
.switch button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
figure {
  margin: 0; background: var(--frame); border: 1px solid var(--rule); border-radius: 8px;
  padding: 12px; overflow-x: auto;
}
figure img { display: block; width: 100%; max-width: 100%; height: auto; min-width: 640px; }
@media (max-width: 900px) {
  .page { grid-template-columns: 1fr; gap: 24px; padding-inline: 16px; }
  nav { position: static; }
  nav ul { grid-auto-flow: row; }
}
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
</style>
<div class="page">
  <nav aria-label="Plots">
    <h1>SPOT Tracking Plots</h1>
    <p class="sub">{{SUB}}</p>
    <ul>
{{NAV}}
    </ul>
  </nav>
  <main>
    <div class="intro">
      <p><strong>Annotated</strong> shows numbered changes from <code>config/annotations.yaml</code>, checked against the athena history. A dotted line in a series color marks a move between SPOT jobs. The caption below each plot names the CPU of each machine and its HS23 (HEPScore23) benchmark. The CPU times are normalized to the reference machine: the values of the other machines are divided by the factor measured for each algorithm at the machine changes (<code>config/machines.yaml</code>), so that a line continues across a machine change. The unit HS23·s is the normalized time in s times the HS23 per core of the reference machine. Dashed black lines mark the last nightly of the public results ATL-PHYS-PUB-2024-017 (2024-09-10) and PLOTS-IDTR-2025-06 (about 2025-10-15).</p>
      <p><strong>No axis cuts</strong> shows the full time axis. The cuts hide periods in which no line changes by more than 5% for at least 60 days.</p>
    </div>
{{SECTIONS}}
  </main>
</div>
<script>
document.querySelectorAll(".switch").forEach(function (group) {
  var img = group.parentElement.querySelector("figure img");
  img.src = group.querySelector('button[aria-pressed="true"]').dataset.src;
  group.addEventListener("click", function (event) {
    var button = event.target.closest("button");
    if (!button) return;
    group.querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", String(b === button)); });
    img.src = button.dataset.src;
  });
});
</script>
"""

if __name__ == "__main__":
    main()
