"""Checks of the point CSVs after a data update.

`steps` finds the changes of the points and their first nightly, groups them
into events, and shows which events have no annotation yet. `gaps` finds dates
inside the lifetime of a point where the point has no value, which is usually
a renamed component.

    python -m spotter.checks steps --since 2026-09-01
    python -m spotter.checks gaps
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from spotter.plotter import matches


def point_names(fetcher_config: dict) -> dict[str, list[str]]:
    """Return the names of all points of each flavor, including validation points."""
    return {
        flavor: [p["name"] for p in c["points"]] + [p["name"] for p in c.get("validation", [])]
        for flavor, c in fetcher_config["flavors"].items()
    }


def read_point(data_dir: Path, flavor: str, point: str) -> pd.DataFrame | None:
    """Read one point CSV, with the value in the column `y`."""
    path = data_dir / f"{flavor}_{point}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    df["y"] = df["value"] if "value" in df else df["cpu"]
    return df.reset_index(drop=True)


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


def steps_command(args, fetcher_config: dict, annotations: list[dict]):
    rows = []
    for flavor, points in point_names(fetcher_config).items():
        if args.flavors and flavor not in args.flavors:
            continue
        for point in points:
            if args.points and point not in args.points:
                continue
            df = read_point(args.data_dir, flavor, point)
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
    args = parser.parse_args()

    with open(args.config) as f:
        fetcher_config = yaml.safe_load(f)
    if args.command == "steps":
        annotations = []
        if args.annotations.exists():
            with open(args.annotations) as f:
                annotations = yaml.safe_load(f)["annotations"]
        steps_command(args, fetcher_config, annotations)
    else:
        gaps_command(args, fetcher_config)


if __name__ == "__main__":
    main()
