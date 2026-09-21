"""Show the values around a date, to check a step or a rename.

Points of a flavor (from the point CSVs):

    python -m spotter.peek 2025-07-09 --flavor acts_fast --points ckf pixel_seeding

Components of a SPOT job (from the raw tables, CPU per event in s):

    python -m spotter.peek 2025-07-09 --tag actstracking --components "TrackFinding|Seeding"

Rows of the validation tables (for example the reco efficiency):

    python -m spotter.peek 2025-07-09 --tag actstracking --validation reco_efficiency
"""

import argparse
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from spotter.fetcher import raw_paths, read_table


def window(df: pd.DataFrame, center: date, days: int) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"]).dt.date
    return df[(dates >= center - timedelta(days=days)) & (dates <= center + timedelta(days=days))]


def logs_info(raw_dir: Path, tag: str) -> pd.DataFrame:
    """Return the platform and the host of each log of a SPOT job."""
    _, logs_path, _ = raw_paths(raw_dir, tag)
    logs = read_table(logs_path, [])
    logs["platform"] = logs["url"].str.extract(r"-main-x86_64-(\w+-gcc\d+)-opt-")
    logs["host"] = logs["host"].str.replace(".cern.ch", "", regex=False)
    return logs[["date", "platform", "host"]]


def show_points(args):
    columns = {}
    info = None
    for point in args.points:
        path = args.data_dir / f"{args.flavor}_{point}.csv"
        if not path.exists():
            print(f"{path}: no such file")
            continue
        df = window(pd.read_csv(path), args.date, args.days)
        columns[point] = df.set_index("date")["value" if "value" in df else "cpu"]
        extra = [c for c in ("tag", "platform", "host") if c in df]
        info = df.set_index("date")[extra] if info is None else info.combine_first(df.set_index("date")[extra])
    table = pd.DataFrame(columns)
    if info is not None:
        table = table.join(info)
    print(table.sort_index().to_string(float_format=lambda v: f"{v:.4g}"))


def show_components(args):
    raw_path, _, _ = raw_paths(args.data_dir / "raw", args.tag)
    raw = window(read_table(raw_path, []), args.date, args.days)
    with warnings.catch_warnings():
        # a regex with groups is fine here; only the match counts
        warnings.simplefilter("ignore", UserWarning)
        selected = raw["component"].str.contains(args.components)
    raw = raw[(raw["step"] == args.step) & selected]
    raw = raw.assign(cpu=raw["cpu"] / raw["count"] * 1e-3)
    table = raw.pivot_table(index="component", columns="date", values="cpu")
    print(table.to_string(float_format=lambda v: f"{v:.4g}", na_rep="-"))
    info = logs_info(args.data_dir / "raw", args.tag)
    print(window(info, args.date, args.days).set_index("date").T.to_string())


def show_validation(args):
    _, _, validation_path = raw_paths(args.data_dir / "raw", args.tag)
    v = window(read_table(validation_path, []), args.date, args.days)
    v = v[(v["table"] == args.validation) & (v["eta"] == args.eta) & (v["pt"].fillna("") == args.pt)]
    table = v.pivot_table(index="alg", columns="date", values="value")
    print(f"{args.validation}, eta: {args.eta}, pt: {args.pt}")
    print(table.to_string(float_format=lambda x: f"{x:.4g}", na_rep="-"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("date", type=date.fromisoformat, help="the center date")
    parser.add_argument("--days", type=int, default=5, help="days before and after the date (default 5)")
    parser.add_argument("--data-dir", type=Path, default="data")
    parser.add_argument("--flavor", help="show points of this flavor")
    parser.add_argument(
        "--points", nargs="+",
        default=["full_track_chain", "clusterization", "space_point_formation", "pixel_seeding", "ckf", "ambiguity"],
        help="the points to show with --flavor",
    )
    parser.add_argument("--tag", help="show raw data of this SPOT job, for example actstracking")
    parser.add_argument("--components", default=".", help="regex on the component names, with --tag")
    parser.add_argument("--step", default="Execute", help="the PerfMon step, with --tag")
    parser.add_argument("--validation", help="a validation table, for example reco_efficiency, with --tag")
    parser.add_argument("--eta", default="All eta", help="the eta bin, with --validation")
    parser.add_argument("--pt", default="All pt", help="the pT bin, with --validation")
    args = parser.parse_args()

    if args.flavor:
        show_points(args)
    elif args.tag and args.validation:
        show_validation(args)
    elif args.tag:
        show_components(args)
    else:
        parser.error("give --flavor, or --tag (with --components or --validation)")


if __name__ == "__main__":
    main()
