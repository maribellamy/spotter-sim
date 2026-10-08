from datetime import date
import argparse
from pathlib import Path

import yaml

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import mplhep as hep

from spotter.axis_cuts import CompressedAxis, find_quiet_periods, periods_to_cuts
from spotter.fitter import piecewise_const
from spotter.machines import (
    find_machine,
    has_scale,
    load_machines,
    normalize_point,
    reference_machine,
)
from spotter.points import PointData, load_fetcher_config, matches, transitive_parts


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

METRICS = {
    "cpu": "CPU time per event [s]",
    "malloc": "Memory allocations per event [MB]",
    "value": "Value",
}


def hs23_unit(machines: list[dict], args) -> float | None:
    """Return the HS23 per core of the reference machine if the plot shows HS23·s, else None.

    HS23·s is the CPU time in s on the reference machine times its HS23 per
    core, the unit of the public plots. It needs the normalization.
    """
    if args.raw or args.unit != "hs23" or not has_scale(machines):
        return None
    return reference_machine(machines).get("per_core")


def read_data(
    data_dir: Path,
    flavors: dict,
    start_date: date,
    end_date: date,
    machines: list[dict] | None = None,
    hs23: float | None = None,
    fetcher_config: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """Read the CSV file of the configured point of each flavor.

    With `machines`, the CPU times are normalized to the reference machine
    (a sum point as the sum of its normalized base points, from
    `fetcher_config`), and with `hs23` they are multiplied by it, to HS23·s.
    """
    data = {}
    for flavor, flavor_config in flavors.items():
        file = data_dir / f"{flavor}_{flavor_config['points']}.csv"
        df = pd.read_csv(file, parse_dates=["date"])
        df.index = df["date"]
        df = df.loc[start_date:end_date]
        if len(df) == 0:
            print(f"{file}: no data from {start_date} to {end_date}")
            continue
        if machines:
            df = normalize_point(
                df, data_dir=data_dir, machines=machines, flavor=flavor,
                point=flavor_config["points"], fetcher_config=fetcher_config,
            )
        if hs23 and "cpu" in df:
            df = df.assign(cpu=df["cpu"] * hs23)
        data[flavor] = df
    return data


def smooth(df: pd.DataFrame, metric: str, window: int) -> pd.Series:
    """Rolling median, which follows the steps and ignores single outliers."""
    return df[metric].rolling(window, center=True, min_periods=1).median()


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


def per_core_label(machine: dict) -> str:
    """Return the HS23 per logical CPU, which is per thread if SMT is on."""
    unit = "thread" if "SMT on" in machine["cores"] else "core"
    return f"{machine['per_core']:.1f} per {unit}"


def machine_captions(
    machines: list[dict], data: dict[str, pd.DataFrame], normalized: bool, hs23: float | None
) -> list[str]:
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
    reference = reference_machine(machines)
    if normalized and reference is not None and any(m.get("scale") for m in seen):
        lines.append(
            f"CPU times normalized to {reference['host']}: the values of the other machines are "
            "divided by the ratio measured per algorithm at the machine changes "
            "(config/machines.yaml), so that a line continues across a machine change"
        )
    if hs23 and reference is not None:
        lines.append(
            f"HS23·s = CPU time in s on {reference['host']} × {hs23:.1f}, its HS23 per core; "
            "the job runs one thread, so the unit is only an estimate of the throughput"
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
    numbers the annotations in a list below the plot, with a marker on every
    line that an annotation affects.
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
    captions += machine_captions(
        machines, data, metric == "cpu" and not args.raw,
        hs23_unit(machines, args) if metric == "cpu" else None,
    )
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
    if metric == "cpu" and hs23_unit(machines, args):
        ylabel = "CPU time per event [HS23·s]"
    elif metric == "cpu" and not args.raw and has_scale(machines):
        ylabel += f", normalized to {reference_machine(machines)['host']}"
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
    top = blended_transform_factory(ax.transData, ax.transAxes)
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
        linear = not args.log_y and not config.get("log_y", False)
        for n, a in enumerate(annotations, start=1):
            for x, y in a["positions"]:
                if linear:
                    y = min(max(y, ylo + pad), yhi - pad)
                ax.plot(x, y, "o", ms=14, color=TEXT_PRIMARY, mec="white", mew=1.5, zorder=6)
                ax.text(
                    x, y, str(n), ha="center", va="center", fontsize=7.5,
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


def collect_annotations(
    config: dict,
    central: list[dict],
    parts: dict,
    point_data: PointData,
    data: dict[str, pd.DataFrame],
    axis: CompressedAxis,
    metric: str,
    window: int,
) -> list[dict]:
    """Return the annotations that are visible, with their positions on the lines.

    An annotation that matches more than one line gets one marker with the
    same number on each of these lines.
    """
    result = {}
    for flavor, flavor_config in config["flavors"].items():
        if flavor not in data:
            continue
        df = data[flavor]
        line = smooth(df, metric, window)
        candidates = [
            a for a in central if matches(a, flavor, flavor_config["points"], parts, point_data)
        ]
        candidates += flavor_config.get("annotations", [])
        for a in candidates:
            d = pd.Timestamp(a["date"])
            if d < df.index.min() or d > df.index.max() or axis.is_cut([d])[0]:
                continue
            nearest = line.index[line.index.get_indexer([d], method="nearest")[0]]
            key = (d, a["text"])
            result.setdefault(key, {"date": d, "text": a["text"], "positions": []})
            result[key]["positions"].append((axis.forward([nearest])[0], line[nearest]))
    return sorted(result.values(), key=lambda a: a["date"])


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
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Plot the measured CPU times, without the normalization to the reference "
        "machine of config/machines.yaml",
    )
    parser.add_argument(
        "--unit",
        choices=["hs23", "s"],
        default="hs23",
        help="The unit of the CPU time: HS23·s (the normalized time times the HS23 per "
        "core of the reference machine, as in the public plots) or seconds",
    )
    parser.add_argument(
        "--fetcher-config",
        type=Path,
        default="config/fetcher.yaml",
        help="Path to the fetcher config, for the sums of the points: an annotation "
        "of a part also belongs to the sums that contain it, and a sum is normalized "
        "as the sum of its normalized base points",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    start_date = date.fromisoformat(config["dates"]["start"])
    end_date = date.fromisoformat(config["dates"]["end"])
    machines = load_machines(args.machines)
    fetcher_config = None
    if args.fetcher_config.exists():
        fetcher_config = load_fetcher_config(args.fetcher_config)
    parts = transitive_parts(fetcher_config) if fetcher_config else {}
    point_data = PointData(args.data_dir)
    data = read_data(
        args.data_dir, config["flavors"], start_date, end_date,
        None if args.raw else machines, hs23_unit(machines, args), fetcher_config,
    )
    central = []
    references = []
    if args.annotations.exists():
        with open(args.annotations, "r") as f:
            loaded = yaml.safe_load(f)
        central = loaded["annotations"]
        references = loaded.get("references", [])

    # the axis cuts do not hide annotations
    keep = [
        pd.Timestamp(a["date"])
        for flavor, f in config["flavors"].items()
        for a in [a for a in central if matches(a, flavor, f["points"], parts, point_data)]
        + f.get("annotations", [])
    ]
    keep += [pd.Timestamp(r["date"]) for r in references]
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
        else collect_annotations(
            config, central, parts, point_data, data, axis, metric, args.smooth_window
        )
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

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
