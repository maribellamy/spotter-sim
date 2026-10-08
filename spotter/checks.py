"""Checks of the point CSVs after a data update.

`steps` finds the changes of the points and their first nightly, groups them
into events, and shows which events have no annotation yet. `gaps` finds dates
inside the lifetime of a point where the point has no value, which is usually
a renamed component. `machine-scale` measures the CPU time of each machine
relative to the reference machine at the machine changes, and prints the
`scale` factors for `config/machines.yaml`.

    python -m spotter.checks steps --since 2026-09-01
    python -m spotter.checks gaps
    python -m spotter.checks machine-scale
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from spotter.machines import find_machine, load_machines, normalize_cpu, reference_machine
from spotter.plotter import matches


def point_names(fetcher_config: dict) -> dict[str, list[str]]:
    """Return the names of all points of each flavor, including validation points."""
    return {
        flavor: [p["name"] for p in c["points"]] + [p["name"] for p in c.get("validation", [])]
        for flavor, c in fetcher_config["flavors"].items()
    }


def read_point(
    data_dir: Path, flavor: str, point: str, machines: list[dict] | None = None
) -> pd.DataFrame | None:
    """Read one point CSV, with the value in the column `y`.

    With `machines`, the CPU times are normalized to the reference machine.
    """
    path = data_dir / f"{flavor}_{point}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    if machines:
        df = normalize_cpu(df, machines, flavor, point)
    df["y"] = df["value"] if "value" in df else df["cpu"]
    return df.reset_index(drop=True)


def machine_label(machine: dict) -> str:
    """Return the host and the period of a machine configuration, for example "aiatlasbm001 (2025-01-01..)"."""
    period = f"{machine.get('start', '')}..{machine.get('end', '')}"
    return f"{machine['host']} ({period})" if period != ".." else machine["host"]


def containing_sums(fetcher_config: dict, flavor: str, point: str) -> list[str]:
    """Return the sum points of the flavor that contain the point, directly or through another sum."""
    sums = {
        p["name"]: p["sum"]
        for p in fetcher_config["flavors"][flavor]["points"]
        if "sum" in p
    }
    result = []
    todo = [point]
    while todo:
        current = todo.pop()
        for name, parts in sums.items():
            if current in parts and name not in result:
                result.append(name)
                todo.append(name)
    return result


def measure_machine_scale(
    df: pd.DataFrame,
    *,
    machines: list[dict],
    annotations: list[dict],
    flavor: str,
    points: list[str],
    window: int,
    min_points: int,
) -> tuple[list[dict], list[str]]:
    """Measure the factor of each machine at the moves between it and the reference.

    For each change of the machine configuration in one series, the function
    compares the median of the last `window` nightlies on the old machine with
    the median of the first `window` nightlies on the new machine. An
    annotation of the series cuts the window, because the code changes there.
    `points` is the point and the sum points that contain it, because an
    annotation of a sum (for example `track_finding`) also changes its parts.
    A move with an annotation on the same day, or with fewer than `min_points`
    nightlies on one side, is skipped. The function returns the measurements
    and the reasons of the skipped moves.
    """
    point = points[0]
    reference = reference_machine(machines)
    df = df[(df["host"] != "unknown") & (df["cpu"] > 0)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    index = [
        next((i for i, m in enumerate(machines) if m is found), -1)
        for found in (find_machine(machines, h, d) for h, d in zip(df["host"], df["date"]))
    ]
    df["machine"] = index
    changes = [
        i for i in range(1, len(df))
        if index[i] != index[i - 1] and index[i] >= 0 and index[i - 1] >= 0
    ]
    dates = sorted({
        pd.Timestamp(a["date"])
        for a in annotations
        if any(matches(a, flavor, p) for p in points)
    })
    rows, skipped = [], []
    for i in changes:
        old, new = machines[index[i - 1]], machines[index[i]]
        if reference not in (old, new):
            continue
        other = new if old is reference else old
        d = df["date"][i]
        name = f"{flavor}.{point} on {d:%Y-%m-%d} ({machine_label(other)})"
        if any(abs((a - d).days) <= 1 for a in dates):
            skipped.append(f"{name}: an annotation on the same day")
            continue
        previous = max([a for a in dates if a < d], default=pd.Timestamp.min)
        following = min([a for a in dates if a > d], default=pd.Timestamp.max)
        # the window stops at an annotation and at a platform change, because the
        # code changes there; a platform change on the move itself stays in the factor
        platforms = (df["platform"][i - 1], df["platform"][i])
        before = df.iloc[max(0, i - window):i]
        before = before[
            (before["machine"] == index[i - 1])
            & (before["date"] > previous)
            & (before["platform"] == platforms[0])
        ]
        after = df.iloc[i:i + window]
        after = after[
            (after["machine"] == index[i])
            & (after["date"] < following)
            & (after["platform"] == platforms[1])
        ]
        if len(before) < min_points or len(after) < min_points:
            skipped.append(f"{name}: only {len(before)} and {len(after)} nightlies")
            continue
        ratio = after["cpu"].median() / before["cpu"].median()
        rows.append({
            "machine": machines.index(other),
            "date": d,
            "flavor": flavor,
            "point": point,
            "n_before": len(before),
            "n_after": len(after),
            "factor": ratio if old is reference else 1 / ratio,
            "platform": "" if platforms[0] == platforms[1] else f"{platforms[0]} -> {platforms[1]}",
        })
    return rows, skipped


def machine_scale_command(args, fetcher_config: dict, annotations: list[dict], machines: list[dict]):
    if reference_machine(machines) is None:
        raise SystemExit(f"{args.machines}: no machine has `reference: true`")
    rows, skipped = [], []
    for flavor, points in point_names(fetcher_config).items():
        if args.flavors and flavor not in args.flavors:
            continue
        for point in points:
            if args.points and point not in args.points:
                continue
            df = read_point(args.data_dir, flavor, point)
            if df is None or "value" in df or "host" not in df:
                continue
            r, s = measure_machine_scale(
                df, machines=machines, annotations=annotations, flavor=flavor,
                points=[point] + containing_sums(fetcher_config, flavor, point),
                window=args.window, min_points=args.min_points,
            )
            rows += r
            skipped += s
    if not rows:
        print("no move between the reference machine and another machine")
        return
    measured = pd.DataFrame(rows).sort_values(["machine", "date", "flavor", "point"])

    print("# measurements: the CPU time on the machine divided by the CPU time on the reference")
    for _, r in measured.iterrows():
        note = f"  (platform {r['platform']})" if r["platform"] else ""
        print(
            f"# {machine_label(machines[r['machine']])}  {r['date']:%Y-%m-%d}  "
            f"{r['flavor']}.{r['point']}: {r['factor']:.3f}  "
            f"({r['n_before']} and {r['n_after']} nightlies){note}"
        )
    for s in skipped:
        print(f"# skipped: {s}")
    print()

    # the factor of a point is the geometric mean over the flavors and the moves;
    # a flavor gets its own factor if it deviates by more than the split threshold
    measured["log"] = np.log(measured["factor"])
    for machine, g in measured.groupby("machine"):
        m = machines[machine]
        by_point = np.exp(g.groupby("point")["log"].mean())
        by_flavor = np.exp(g.groupby(["flavor", "point"])["log"].mean())
        moves = ", ".join(f"{d:%Y-%m-%d}" for d in sorted(g["date"].unique()))
        platforms = sorted({p for p in g["platform"] if p})
        print(f"# {machine_label(m)}: factors from the moves on {moves}")
        if platforms:
            print(f"# the moves include a platform change ({', '.join(platforms)}), so the factors include it")
        print(f"# python -m spotter.checks machine-scale --window {args.window} --min-points {args.min_points}")
        print("    scale:")
        print(f"      default: {by_point.median():.3f}")
        for point, factor in by_point.sort_index().items():
            print(f"      {point}: {factor:.3f}")
        for (flavor, point), factor in by_flavor.sort_index().items():
            if abs(factor / by_point[point] - 1) > args.split_threshold:
                print(f"      {flavor}.{point}: {factor:.3f}")
        print()


def find_steps(
    df: pd.DataFrame, *, window: int, threshold: float, min_abs: float
) -> list[dict]:
    """Find the steps of one series.

    For each date, the function compares the median of the `window` values
    before with the median of the `window` values after. A step is a local
    maximum of the relative change above `threshold` (and above `min_abs`).
    The median window flips before the real step, so the step date is the
    first day that is closer to the level after than to the level before.
    """
    y = df["y"].to_numpy(dtype=float)
    n = len(y)
    score = np.zeros(n)
    before = np.zeros(n)
    after = np.zeros(n)
    for i in range(window, n - window + 1):
        before[i] = np.median(y[i - window:i])
        after[i] = np.median(y[i:i + window])
        if before[i] > 0 and abs(after[i] - before[i]) > min_abs:
            score[i] = abs(after[i] / before[i] - 1)
    steps = {}
    for i in range(window, n - window + 1):
        if score[i] <= threshold or score[i] < score[max(0, i - 5):i + 6].max():
            continue
        b, a = before[i], after[i]
        j = i
        for k in range(max(1, i - 4), min(n, i + 5)):
            if abs(y[k] - a) < abs(y[k] - b) and abs(y[k - 1] - a) >= abs(y[k - 1] - b):
                j = k
                break
        steps[j] = {
            "date": df["date"][j],
            "previous": df["date"][j - 1],
            "before": b,
            "after": a,
            "change": a / b - 1,
            "infrastructure": ", ".join(
                f"{column} {df[column][j - 1]} -> {df[column][j]}"
                for column in ("platform", "host")
                if column in df and df[column][j] != df[column][j - 1]
            ),
        }
    return list(steps.values())


def steps_command(args, fetcher_config: dict, annotations: list[dict], machines: list[dict]):
    rows = []
    for flavor, points in point_names(fetcher_config).items():
        if args.flavors and flavor not in args.flavors:
            continue
        for point in points:
            if args.points and point not in args.points:
                continue
            df = read_point(args.data_dir, flavor, point, None if args.raw else machines)
            if df is None or len(df) < 2 * args.window:
                continue
            # a validation point (for example an efficiency) changes by about 1%
            is_value = "value" in df
            threshold = args.threshold_value if is_value else args.threshold
            min_abs = args.min_abs_value if is_value else args.min_abs_cpu
            for step in find_steps(df, window=args.window, threshold=threshold, min_abs=min_abs):
                d = pd.Timestamp(step["date"])
                known = [
                    a["text"]
                    for a in annotations
                    if abs((pd.Timestamp(a["date"]) - d).days) <= 1 and matches(a, flavor, point)
                ]
                rows.append({"flavor": flavor, "point": point, **step, "annotation": "; ".join(known)})
    if not rows:
        print("no steps")
        return
    steps = pd.DataFrame(rows)
    steps = steps[steps["date"] >= str(args.since)].sort_values(["date", "flavor", "point"])

    # steps within `group_days` of each other are one event
    events = []
    for _, row in steps.iterrows():
        d = pd.Timestamp(row["date"])
        if events and (d - events[-1]["last"]).days <= args.group_days:
            events[-1]["rows"].append(row)
            events[-1]["last"] = d
        else:
            events.append({"first": d, "last": d, "rows": [row]})

    for event in events:
        rows = event["rows"]
        annotated = all(r["annotation"] for r in rows)
        infrastructure = sorted({r["infrastructure"] for r in rows if r["infrastructure"]})
        if args.unannotated and (annotated or infrastructure):
            continue
        status = "annotated" if annotated else ("INFRASTRUCTURE" if infrastructure else "NEW")
        print(f"{event['first']:%Y-%m-%d}  last unchanged {min(r['previous'] for r in rows)}  [{status}]")
        for info in infrastructure:
            print(f"    {info}")
        for r in rows:
            print(
                f"    {r['flavor']}.{r['point']}: {r['before']:.4g} -> {r['after']:.4g} "
                f"({r['change']:+.0%}) on {r['date']}"
                + (f"  | {r['annotation']}" if r["annotation"] else "")
            )


def gaps_command(args, fetcher_config: dict):
    found = False
    for flavor, points in point_names(fetcher_config).items():
        series = {p: read_point(args.data_dir, flavor, p) for p in points}
        series = {p: df for p, df in series.items() if df is not None}
        if not series:
            continue
        all_dates = sorted(set().union(*[set(df["date"]) for df in series.values()]))
        index = {d: i for i, d in enumerate(all_dates)}
        for point, df in series.items():
            have = set(df["date"])
            first, last = min(have), max(have)
            missing = [d for d in all_dates if first <= d <= last and d not in have]
            if not missing:
                continue
            runs = []
            for d in missing:
                if runs and index[d] == index[runs[-1][1]] + 1:
                    runs[-1][1] = d
                else:
                    runs.append([d, d])
            text = ", ".join(a if a == b else f"{a}..{b}" for a, b in runs[:6])
            print(f"{flavor}.{point}: {len(missing)} missing: {text}" + (" ..." if len(runs) > 6 else ""))
            found = True
    if not found:
        print("no gaps")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default="config/fetcher.yaml")
    parser.add_argument("--annotations", type=Path, default="config/annotations.yaml")
    parser.add_argument("--data-dir", type=Path, default="data")
    parser.add_argument("--machines", type=Path, default="config/machines.yaml")
    parser.add_argument(
        "--raw", action="store_true",
        help="use the measured CPU times, without the normalization to the reference machine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    steps = sub.add_parser("steps", help="find the steps of the points and group them into events")
    steps.add_argument("--since", type=date.fromisoformat, default=date.min, help="only events from this date")
    steps.add_argument("--flavors", nargs="+", help="only these flavors")
    steps.add_argument("--points", nargs="+", help="only these points")
    steps.add_argument("--threshold", type=float, default=0.08, help="minimum relative change (default 0.08)")
    steps.add_argument("--threshold-value", type=float, default=0.006, help="minimum relative change of a validation point (default 0.006)")
    steps.add_argument("--min-abs-cpu", type=float, default=0.02, help="minimum change of a CPU point in s")
    steps.add_argument("--min-abs-value", type=float, default=0.005, help="minimum change of a validation point")
    steps.add_argument("--window", type=int, default=7, help="number of values before and after a step")
    steps.add_argument("--group-days", type=int, default=2, help="steps within these days are one event")
    steps.add_argument("--unannotated", action="store_true", help="only events without an annotation")

    sub.add_parser("gaps", help="find dates without a value inside the lifetime of a point")

    scale = sub.add_parser(
        "machine-scale",
        help="measure the CPU time of each machine relative to the reference machine, "
        "and print the `scale` factors for the machines config",
    )
    scale.add_argument("--flavors", nargs="+", help="only these flavors")
    scale.add_argument("--points", nargs="+", help="only these points")
    scale.add_argument("--window", type=int, default=7, help="number of nightlies before and after a move (default 7)")
    scale.add_argument("--min-points", type=int, default=3, help="minimum number of nightlies on each side of a move (default 3)")
    scale.add_argument(
        "--split-threshold", type=float, default=0.03,
        help="a flavor gets its own factor if it deviates from the factor of the point by more than this (default 0.03)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        fetcher_config = yaml.safe_load(f)
    annotations = []
    if args.annotations.exists():
        with open(args.annotations) as f:
            annotations = yaml.safe_load(f)["annotations"]
    machines = load_machines(args.machines)
    if args.command == "steps":
        steps_command(args, fetcher_config, annotations, machines)
    elif args.command == "machine-scale":
        machine_scale_command(args, fetcher_config, annotations, machines)
    else:
        gaps_command(args, fetcher_config)


if __name__ == "__main__":
    main()
