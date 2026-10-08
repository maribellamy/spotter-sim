from datetime import date
import argparse
import re
from pathlib import Path
import textwrap

import yaml

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import mplhep as hep

from spotter.axis_cuts import CompressedAxis, find_quiet_periods, periods_to_cuts
from spotter.fitter import piecewise_const


# one fixed color and marker for each flavor, so that a flavor looks the same in every plot
FLAVOR_STYLE = {
    "acts_default": ("#2a78d6", "o"),
    "legacy_default": ("#eb6834", "s"),
    "acts_fast": ("#1baf7a", "^"),
    "legacy_fast": ("#eda100", "D"),
    "acts_legacy": ("#e87ba4", "v"),
}
FALLBACK_STYLE = ("#4a3aa7", "P")

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8985"
GRID = "#e4e3df"
GAP_FILL = "#f0efec"

# a plot of the key also shows the annotations of these points
RELATED_POINTS = {
    "track_finding": {"ckf", "seeding", "pixel_seeding", "strip_seeding"},
    "seeding": {"pixel_seeding", "strip_seeding"},
    "lrt": {"lrt_seeding", "lrt_ckf"},
    "full_track_chain_lrt": {"full_track_chain", "lrt", "lrt_seeding", "lrt_ckf"},
}

METRICS = {
    "cpu": "CPU time per event [s]",
    "malloc": "Memory allocations per event [MB]",
    "value": "Value",
}


def read_data(
    data_dir: Path, flavors: dict, start_date: date, end_date: date
) -> dict[str, pd.DataFrame]:
    """Read the CSV file of the configured point of each flavor."""
    data = {}
    for flavor, flavor_config in flavors.items():
        file = data_dir / f"{flavor}_{flavor_config['points']}.csv"
        df = pd.read_csv(file, parse_dates=["date"])
        df.index = df["date"]
        df = df.loc[start_date:end_date]
        if len(df) == 0:
            print(f"{file}: no data from {start_date} to {end_date}")
            continue
        data[flavor] = df
    return data


def smooth(df: pd.DataFrame, metric: str, window: int) -> pd.Series:
    """Rolling median, which follows the steps and ignores single outliers."""
    return df[metric].rolling(window, center=True, min_periods=1).median()


def infrastructure_changes(
    data: dict[str, pd.DataFrame], column: str
) -> list[tuple[pd.Timestamp, str]]:
    """Return the dates where the platform or the host of the SPOT logs changes.

    `column` is "platform" or "host". A change within 7 days of an earlier
    change of another flavor is the same change.
    """
    changes = {}
    for df in data.values():
        if column not in df:
            continue
        values = df[column]
        changed = values.ne(values.shift()) & values.shift().notna()
        for d, value in values[changed].items():
            if not any(abs((d - c).days) <= 7 for c in changes):
                changes[d] = value
    return sorted(changes.items())


def tag_changes(df: pd.DataFrame) -> list[tuple[pd.Timestamp, str, str]]:
    """Return the dates where a flavor moves from one SPOT job to another."""
    if "tag" not in df:
        return []
    tag = df["tag"]
    changed = tag.ne(tag.shift()) & tag.shift().notna()
    return [(d, tag.shift()[d], t) for d, t in tag[changed].items()]


def compute_axis_cuts(
    args, config: dict, data: dict[str, pd.DataFrame], keep: list[pd.Timestamp]
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return the axis cuts from the config, from the data, or none."""
    if args.axis_cuts == "off":
        return []
    if args.axis_cuts == "config":
        return [
            (pd.Timestamp(c["start"]), pd.Timestamp(c["end"]))
            for c in config.get("axis_cuts", [])
        ]
    periods = find_quiet_periods(
        [df[config.get("metric", "cpu")] for df in data.values()],
        threshold=args.cut_threshold or config.get("cut_threshold", 0.05),
        min_days=args.cut_min_days,
    )
    return periods_to_cuts(
        periods,
        margin_days=args.cut_margin_days,
        min_cut_days=args.cut_min_days - 2 * args.cut_margin_days,
        keep=keep,
    )


def mark_below_axis(
    ax, axis: CompressedAxis, data: dict[str, pd.DataFrame], metric: str, ymin: float, window: int
):
    """Mark each period in which a line is below the y axis with its minimum.

    A zoomed y axis hides large drops (for example a bug that halves the
    efficiency), so the plot shows them at the lower edge.
    """
    for flavor, df in data.items():
        color, _ = FLAVOR_STYLE.get(flavor, FALLBACK_STYLE)
        line = smooth(df, metric, window)
        below = line < ymin
        runs = (below != below.shift()).cumsum()[below]
        for _, run in line[below].groupby(runs):
            if axis.is_cut([run.idxmin()])[0]:
                continue
            x = axis.forward([run.index[len(run) // 2]])[0]
            edge = blended_transform_factory(ax.transData, ax.transAxes)
            ax.plot(x, 0.025, "v", ms=9, color=color, transform=edge, clip_on=False, zorder=6)
            ax.annotate(
                f"{run.min():.2f}",
                xy=(x, 0.025), xycoords=edge, xytext=(8, 0), textcoords="offset points",
                ha="left", va="center", fontsize=10, color=TEXT_PRIMARY,
            )


def find_machine(machines: list[dict], host: str, d: pd.Timestamp) -> dict | None:
    """Return the configuration of a SPOT machine on a date, from config/machines.yaml."""
    for m in machines:
        start = pd.Timestamp(m.get("start", "1900-01-01"))
        end = pd.Timestamp(m.get("end", "2100-01-01"))
        if m["host"] == host and start <= d <= end:
            return m
    return None


def per_core_label(machine: dict) -> str:
    """Return the HS23 per logical CPU, which is per thread if SMT is on."""
    unit = "thread" if "SMT on" in machine["cores"] else "core"
    return f"{machine['per_core']:.1f} per {unit}"


def short_cpu(machine: dict) -> str:
    """Return the CPU model without the vendor and the clock, for example "Xeon Gold 6326"."""
    return re.sub(r"^(AMD|Intel) ", "", machine["cpu"]).split(" @")[0]


def machine_captions(machines: list[dict], data: dict[str, pd.DataFrame]) -> list[str]:
    """Return one caption line for each machine configuration in the plotted data."""
    seen = []
    for df in data.values():
        if "host" not in df:
            continue
        for d, host in df["host"].items():
            m = find_machine(machines, host, d)
            if m is not None and m not in seen:
                seen.append(m)
    lines = [
        f"{m['host']}: {m['cpu']}, {m['cores']}, HS23 {m['hs23']:.1f} ({per_core_label(m)})"
        for m in seen
    ]
    if lines:
        lines.append(
            "HS23 = HEPScore23 benchmark score of the whole machine, higher is faster "
            "(HEPiX values for the same CPU model and core configuration, "
            "w3.hepix.org/benchmarking/scores_HS23.html)"
        )
    return lines


def plot_timeseries(
    *,
    args,
    config: dict,
    data: dict[str, pd.DataFrame],
    axis: CompressedAxis,
    metric: str,
    annotations: list[dict],
    output: Path,
    machines: list[dict],
    references: list[dict],
):
    """Plot one metric of all flavors over time.

    Each flavor shows its daily values as small markers and a rolling median
    (or the configured piecewise constant fit) as a line. The plot marks the
    changes of the platform and the moves of a flavor between SPOT jobs, and
    numbers the annotations in a list below the plot.
    """
    flavors = config["flavors"]
    # the dates of the config, so that plots with the same dates have the same x range
    start = pd.Timestamp(config["dates"]["start"])
    end = pd.Timestamp(config["dates"]["end"])

    # the layout in inches, so that it does not depend on the number of annotations
    columns = 2 if len(annotations) > 8 else 1
    rows = (len(annotations) + columns - 1) // columns
    visible = [
        r for r in references
        if start <= pd.Timestamp(r["date"]) <= end and not axis.is_cut([pd.Timestamp(r["date"])])[0]
    ]
    captions = [f"{r['name']}: data {r['range']}" for r in visible]
    captions += machine_captions(machines, data)
    width, h_head, h_plot, h_ticks, h_row, h_foot = 15.0, 1.15, 5.6, 0.75, 0.27, 0.35
    h_caption = 0.22 * len(captions) + (0.2 if captions else 0)
    height = h_head + h_plot + h_ticks + h_caption + rows * h_row + (h_foot if rows else 0.1)
    fig = plt.figure(figsize=(width, height), dpi=150)
    left, right = 0.9, 0.4
    ax = fig.add_axes(
        [
            left / width,
            (height - h_head - h_plot) / height,
            (width - left - right) / width,
            h_plot / height,
        ]
    )

    moves: list[tuple[float, str, str]] = []
    for flavor, df in data.items():
        if len(df) == 0:
            continue
        color, marker = FLAVOR_STYLE.get(flavor, FALLBACK_STYLE)
        x, y = axis.split_lines(df.index, df[metric])
        ax.plot(x, y, marker, ls="none", ms=3, color=color, alpha=0.3, mew=0, zorder=2)

        edges = flavors[flavor].get("piecewise_const", [])
        if args.line == "piecewise" and edges:
            px, py = piecewise_const(
                df["date"],
                df[metric],
                [(pd.Timestamp(e["start"]), pd.Timestamp(e["end"])) for e in edges],
            )
            lx, ly = axis.split_lines(pd.Series(px), pd.Series(py))
        else:
            lx, ly = axis.split_lines(df.index, smooth(df, metric, args.smooth_window))
        ax.plot(lx, ly, "-", lw=2, color=color, zorder=3, solid_capstyle="round")

        for d, old, new in tag_changes(df):
            if axis.is_cut([d])[0]:
                continue
            moves.append((axis.forward([d])[0], color, f" {old} → {new}"))

    # two moves on about the same date get their labels on both sides:
    # the label of the left line goes to the left, and the other label to the right
    moves.sort(key=lambda m: m[0])
    for i, (xd, color, label) in enumerate(moves):
        side = "left" if i > 0 and xd - moves[i - 1][0] < 15 else "right"
        ax.axvline(xd, color=color, lw=1.2, ls=(0, (2, 2)), alpha=0.8, zorder=1)
        ax.text(
            xd,
            0.02,
            label,
            transform=blended_transform_factory(ax.transData, ax.transAxes),
            rotation=90,
            ha=side,
            va="bottom",
            fontsize=9,
            color=TEXT_SECONDARY,
        )

    # changes of the platform and of the SPOT machine, one line and label per date
    top = blended_transform_factory(ax.transData, ax.transAxes)
    changes: dict[pd.Timestamp, list[str]] = {}
    for column in ("platform", "host"):
        for d, value in infrastructure_changes(data, column):
            if value == "unknown":
                continue
            label = value
            if column == "host":
                m = find_machine(machines, value, d)
                label = f"machine {value}" + (
                    f" ({short_cpu(m)}, HS23 {per_core_label(m)})" if m else ""
                )
            same = [c for c in changes if abs((d - c).days) <= 7]
            changes.setdefault(same[0] if same else d, []).append(label)
    for d, labels in sorted(changes.items()):
        # a label at the left edge would be cut off
        if axis.is_cut([d])[0] or (d - start).days < 15:
            continue
        xd = axis.forward([d])[0]
        ax.axvline(xd, color=TEXT_MUTED, lw=0.8, ls="-" if labels[0][0] != "m" else "-.", zorder=1)
        ax.text(
            xd, 0.985, " " + ", ".join(labels), transform=top, rotation=90, ha="right",
            va="top", fontsize=9, color=TEXT_MUTED,
        )

    # y axis from zero, because the reader compares magnitudes; a logarithmic
    # axis for data over more than a factor of about 10
    if args.log_y or config.get("log_y", False):
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:g}" if f"{v:g}"[0] in "25" else "")
        )
        ax.tick_params(axis="y", which="minor", labelsize=10)
        values = pd.concat([df[metric] for df in data.values()])
        ax.set_ylim(values.min() * 0.7, values.max() * 1.6)
    elif metric != "value":
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    if "ylim" in config and metric != "malloc":
        ax.set_ylim(*config["ylim"])
        mark_below_axis(ax, axis, data, metric, config["ylim"][0], args.smooth_window)
    ylabel = config.get("ylabel", METRICS[metric]) if metric == "value" else METRICS[metric]
    ax.set_ylabel(ylabel, fontsize=16)

    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    xmin = axis.forward([start])[0] - 5
    xmax = axis.forward([end])[0] + 5
    ax.set_xlim(xmin, xmax)
    ticks, tick_labels = axis.month_ticks(start, end)
    ax.set_xticks(ticks, tick_labels, fontsize=12)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.tick_params(axis="y", labelsize=12)

    # the gaps of the axis cuts, with a break mark on the x axis
    bottom = blended_transform_factory(ax.transData, ax.transAxes)
    for x0, x1, days in axis.gaps():
        ax.axvspan(x0, x1, color=GAP_FILL, lw=0, zorder=0)
        for xm in (x0, x1):
            ax.plot(
                [xm - 0.6, xm + 0.6], [-0.012, 0.012], color=TEXT_SECONDARY,
                lw=1.2, transform=bottom, clip_on=False, zorder=5,
            )
        months = days / 30.44
        ax.text(
            (x0 + x1) / 2, 0.5, f"{months:.0f} months" if months >= 1.5 else f"{days} days",
            transform=bottom, rotation=90, ha="center", va="center",
            fontsize=8, color=TEXT_MUTED,
        )

    # public results, as a dashed line at the last nightly that they contain
    for r in visible:
        xd = axis.forward([pd.Timestamp(r["date"])])[0]
        ax.axvline(xd, color=TEXT_PRIMARY, lw=1.3, ls=(0, (5, 3)), zorder=4)
        ax.text(
            xd, 0.5, f" {r['name']} ", transform=top, rotation=90, ha="right",
            va="center", fontsize=10, color=TEXT_PRIMARY, fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8),
        )

    # the public results and the SPOT machines with their benchmark, below the x axis
    for i, line in enumerate(captions):
        fig.text(
            left / width,
            (height - h_head - h_plot - h_ticks - 0.22 * i) / height,
            line,
            fontsize=10,
            color=TEXT_SECONDARY,
            va="top",
        )

    # numbered annotations on the line, explained in a list below the plot
    if annotations:
        # a value outside the y axis gets its badge at the edge
        ylo, yhi = ax.get_ylim()
        pad = (yhi - ylo) * 0.03
        for a in annotations:
            if not args.log_y and not config.get("log_y", False):
                a["y"] = min(max(a["y"], ylo + pad), yhi - pad)
        for n, a in enumerate(annotations, start=1):
            ax.plot(a["x"], a["y"], "o", ms=14, color=TEXT_PRIMARY, mec="white", mew=1.5, zorder=6)
            ax.text(
                a["x"], a["y"], str(n), ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold", zorder=7,
            )
        for n, a in enumerate(annotations, start=1):
            column, r = divmod(n - 1, rows)
            text = a["text"].replace("\n", " ")
            fig.text(
                (left + column * (width - left) / columns) / width,
                (height - h_head - h_plot - h_ticks - h_caption - r * h_row) / height,
                f"{n:>2}   {a['date']:%Y-%m-%d}   {text}",
                fontsize=10.5,
                color=TEXT_PRIMARY,
                va="top",
            )

    hep.atlas.text("Simulation Internal", ax=ax, loc=0, fontsize=16)
    subtitle = config.get("title", "")
    ax.text(
        1.0, 1.02,
        r"HL-LHC, $\sqrt{s}$ = 14 TeV, $t\bar{t}$, $\langle\mu\rangle$ = 200"
        + (f"   \u2022   {subtitle}" if subtitle else ""),
        transform=ax.transAxes, fontsize=13, color=TEXT_SECONDARY, ha="right", va="bottom",
    )

    legend = [
        Line2D([], [], color=FLAVOR_STYLE.get(f, FALLBACK_STYLE)[0],
               marker=FLAVOR_STYLE.get(f, FALLBACK_STYLE)[1], lw=2, ms=6,
               label=flavors[f]["label"])
        for f in data
    ]
    ax.legend(
        handles=legend, loc="lower left", bbox_to_anchor=(0.0, 1.09), ncol=len(legend),
        frameon=False, fontsize=12, handlelength=2.2, columnspacing=1.8, borderaxespad=0,
    )

    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"))
    return fig


def matches(annotation: dict, flavor: str, point: str) -> bool:
    """Return True if an annotation of the central file belongs to a plotted line."""
    points = set(annotation.get("points", []))
    return flavor in annotation.get("flavors", []) and (
        point in points or bool(points & RELATED_POINTS.get(point, set()))
    )


def collect_annotations(
    config: dict,
    central: list[dict],
    data: dict[str, pd.DataFrame],
    axis: CompressedAxis,
    metric: str,
    window: int,
) -> list[dict]:
    """Return the annotations that are visible, with their position on the line.

    An annotation that matches more than one line gets one marker, on the first
    matching line in the order of the config.
    """
    result = {}
    for flavor, flavor_config in config["flavors"].items():
        if flavor not in data:
            continue
        df = data[flavor]
        line = smooth(df, metric, window)
        candidates = [a for a in central if matches(a, flavor, flavor_config["points"])]
        candidates += flavor_config.get("annotations", [])
        for a in candidates:
            d = pd.Timestamp(a["date"])
            key = (d, a["text"])
            if key in result:
                continue
            if d < df.index.min() or d > df.index.max() or axis.is_cut([d])[0]:
                continue
            nearest = line.index[line.index.get_indexer([d], method="nearest")[0]]
            result[key] = {
                "date": d,
                "text": a["text"],
                "x": axis.forward([nearest])[0],
                "y": line[nearest],
            }
    return sorted(result.values(), key=lambda a: a["date"])


def plot_relative(
    args, config: dict, flavor: str, data: dict[str, pd.DataFrame], central: list[dict]
):
    """Plot the CPU time at the selected annotations relative to another flavor."""
    point = config["flavors"][flavor]["points"]
    selected_cpu_values = []
    selected_dates = []
    annotations = [a for a in central if matches(a, flavor, point)]
    annotations += config["flavors"][flavor].get("annotations", [])
    for annotation in sorted(annotations, key=lambda a: a["date"]):
        d = pd.Timestamp(annotation["date"])
        if not annotation.get("select", False) or len(data[flavor].loc[d:d]) == 0:
            continue
        selected_cpu_values.append([data[flavor].loc[d:d].cpu.iloc[0], annotation["label"]])
        selected_dates.append(d)
    if not selected_cpu_values:
        return

    with plt.style.context(hep.style.ATLAS):
        compare_to_flavor, compare_to_date = config["flavors"][flavor]["compare_to_other"]
        compare_to_cpu = (
            data[compare_to_flavor].loc[compare_to_date:compare_to_date].cpu.iloc[0]
        )
        cpu_abs = np.array([t for t, _ in selected_cpu_values], dtype=np.float64)
        cpu_labels = [f"{i}: {l}" for i, (_, l) in enumerate(selected_cpu_values, start=1)]
        cpu_rel = cpu_abs / compare_to_cpu
        cpu_rel = np.insert(cpu_rel, 0, cpu_rel[0])
        cpu_rel = np.insert(cpu_rel, -1, cpu_rel[-1])

        compare_fig, compare_ax = plt.subplots(figsize=(10, 5))

        # this gets rid of the vertical lines. we clip in y later anyway
        xrange = np.arange(len(cpu_rel))
        xrange[0] -= 5
        xrange[-1] += 5
        compare_ax.fill_between(
            xrange,
            cpu_rel,
            fc="#6699ff",  # kAzure - 4
            ec="#003399",  # kAzure - 6
            lw=3,
        )

        compare_ax.axhline(1, color="#003399", lw=3, ls="--")

        compare_ax.set_xlim(0, len(cpu_rel) - 1)
        _, ymax = compare_ax.get_ylim()
        compare_ax.set_ylim(0.5, ymax + 0.2)

        compare_ax.set_ylabel("Relative CPU time per event")
        compare_ax.set_xlabel("Work Progress", labelpad=0)

        ticks = np.arange(len(cpu_rel) - 1)
        compare_ax.set_xticks(ticks)
        compare_ax.set_xticklabels([])
        compare_ax.set_xticks(np.arange(len(cpu_rel) - 1) + 0.5, minor=True)

        compare_ax.set_yticks([1, 2, 3, 4, 5, 6])

        compare_ax.set_xticklabels(
            cpu_labels + [""],
            rotation=-22,
            ha="left",
            va="top",
            ma="center",
            fontsize=15,
            minor=True,
        )
        compare_ax.tick_params(axis="x", which="minor", tick1On=False, tick2On=False)

        hep.atlas.text("Internal", fontsize=16, pad=0)

        compare_title = config.get(
            "compare_title",
            f"{config['flavors'][flavor]['label']} vs {config['flavors'][compare_to_flavor]['label']} "
            "track seeding and finding CPU/Event",
        )
        text = rf"""
HL-LHC, $\sqrt{{s}}$ = 14 TeV, $t\bar{{t}}$, $\langle\mu\rangle$ = 200
{compare_title}
Start date: {selected_dates[0]:%d %B %Y}
End date: {selected_dates[-1]:%d %B %Y}
        """
        text = textwrap.dedent(text.strip())

        compare_ax.text(
            0.97,
            0.96,
            s=text,
            transform=compare_ax.transAxes,
            ha="right",
            va="top",
            ma="left",
        )

        compare_fig.tight_layout()

        compare_fig.savefig(args.plots_dir / f"cpu_{flavor}_rel.png")
        compare_fig.savefig(args.plots_dir / f"cpu_{flavor}_rel.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/plot-trackfinding.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="data",
        help="Directory to store the data",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default="plots",
        help="Directory to store the plots",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default="config/annotations.yaml",
        help="Path to the annotations for all plots",
    )
    parser.add_argument(
        "--machines",
        type=Path,
        default="config/machines.yaml",
        help="Path to the CPU and HS23 values of the SPOT machines",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plots",
    )
    parser.add_argument(
        "--hide-annotations",
        action="store_true",
        help="Hide the annotations",
    )
    parser.add_argument(
        "--axis-cuts",
        choices=["config", "auto", "off"],
        default="config",
        help="Cut quiet periods out of the x axis: the cuts in the config, cuts "
        "computed from the data with the threshold below, or no cuts",
    )
    parser.add_argument(
        "--suggest-axis-cuts",
        action="store_true",
        help="Print the axis cuts computed from the data in YAML format and exit",
    )
    parser.add_argument(
        "--cut-threshold",
        type=float,
        default=None,
        help="A period is quiet if no series changes by more than this relative amount "
        "(default: `cut_threshold` of the config, or 0.05)",
    )
    parser.add_argument(
        "--cut-min-days",
        type=int,
        default=60,
        help="The minimum length in days of a quiet period",
    )
    parser.add_argument(
        "--cut-margin-days",
        type=int,
        default=14,
        help="The days that stay visible at both ends of a quiet period",
    )
    parser.add_argument(
        "--gap-days",
        type=float,
        default=14,
        help="The width in days of the gap that replaces a cut",
    )
    parser.add_argument(
        "--line",
        choices=["median", "piecewise"],
        default="median",
        help="Draw a rolling median, or the piecewise constant fits of the config",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help="The number of points of the rolling median",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic y axis (the config can also set `log_y: true`)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    start_date = date.fromisoformat(config["dates"]["start"])
    end_date = date.fromisoformat(config["dates"]["end"])
    data = read_data(args.data_dir, config["flavors"], start_date, end_date)
    machines = []
    if args.machines.exists():
        with open(args.machines, "r") as f:
            machines = yaml.safe_load(f)["machines"]
    central = []
    references = []
    if args.annotations.exists():
        with open(args.annotations, "r") as f:
            loaded = yaml.safe_load(f)
        central = loaded["annotations"]
        references = loaded.get("references", [])

    # the axis cuts do not hide annotations, moves between SPOT jobs, or changes of the infrastructure
    keep = [
        pd.Timestamp(a["date"])
        for flavor, f in config["flavors"].items()
        for a in [a for a in central if matches(a, flavor, f["points"])]
        + f.get("annotations", [])
    ]
    keep += [pd.Timestamp(r["date"]) for r in references]
    keep += [d for df in data.values() for d, _, _ in tag_changes(df)]
    keep += [d for column in ("platform", "host") for d, _ in infrastructure_changes(data, column)]
    cuts = compute_axis_cuts(args, config, data, keep)

    if args.suggest_axis_cuts:
        print(
            f"# quiet periods: threshold {args.cut_threshold or config.get('cut_threshold', 0.05)}, "
            f"at least {args.cut_min_days} days, margin {args.cut_margin_days} days"
        )
        print("axis_cuts:")
        for a, b in cuts:
            print(f'  - {{ start: "{a:%Y-%m-%d}", end: "{b:%Y-%m-%d}" }}')
        return

    axis = CompressedAxis(cuts, args.gap_days)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    metric = config.get("metric", "cpu")
    output = config.get("output", "cpu")
    annotations = (
        []
        if args.hide_annotations
        else collect_annotations(config, central, data, axis, metric, args.smooth_window)
    )
    with plt.style.context(hep.style.ATLAS):
        plot_timeseries(
            args=args, config=config, data=data, axis=axis, metric=metric,
            annotations=[], output=args.plots_dir / output, machines=machines, references=references,
        )
        plot_timeseries(
            args=args, config=config, data=data, axis=axis, metric=metric,
            annotations=annotations, output=args.plots_dir / f"{output}_annotated",
            machines=machines, references=references,
        )
        if metric == "cpu":
            plot_timeseries(
                args=args, config=config, data=data, axis=axis, metric="malloc",
                annotations=[], output=args.plots_dir / "mem", machines=machines, references=references,
            )

    if not args.hide_annotations:
        for flavor, flavor_config in config["flavors"].items():
            if flavor_config.get("compare_to_other") and metric == "cpu":
                plot_relative(args, config, flavor, data, central)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
