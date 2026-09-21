import re
from datetime import date, datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import itertools
import urllib.request
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
import argparse
from pathlib import Path
import yaml
import pandas as pd
from spotter.parser import parse_spot, extract_points_per_algorithm, is_masked
from spotter.utilities import date_range


BASE_URL = "https://atlaspmb.web.cern.ch/atlaspmb"
LOG_DIRS = ["logs", "logs/archive_centos7"]


def get_log_index(*, tag: str) -> dict[date, str]:
    """Map each date to the URL of the `main` RAWtoALL log for a SPOT tag.

    The platform and the compiler in the log name change over time
    (centos7-gcc11, el9-gcc13, el9-gcc14, el9-gcc15). If a date has more than
    one log, the log with el9 and the newest compiler wins.
    """
    assert isinstance(tag, str)
    pattern = re.compile(
        rf"^(\d\d-\d\d-\d\d\d\d)-main-x86_64-(\w+)-gcc(\d+)-opt-phase2_recoonly_{tag}-RAWtoALL$"
    )
    candidates: dict[date, tuple[tuple[bool, int], str]] = {}
    for log_dir in LOG_DIRS:
        url = f"{BASE_URL}/spot-mon-phase2_recoonly_{tag}/pages/{log_dir}/"
        try:
            html = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
        except HTTPError as e:
            print(url, e)
            continue
        for name in re.findall(r'href="([^"?/]+)"', html):
            m = pattern.match(name)
            if not m:
                continue
            d = datetime.strptime(m.group(1), "%d-%m-%Y").date()
            rank = (m.group(2) == "el9", int(m.group(3)))
            if d not in candidates or rank > candidates[d][0]:
                candidates[d] = (rank, f"{url}{name}")
    return {d: u for d, (_, u) in candidates.items()}


def load_log(url: str, retries: int) -> str | None:
    assert isinstance(url, str)
    assert isinstance(retries, int)
    for _ in range(retries):
        try:
            return urllib.request.urlopen(url, timeout=120).read().decode("utf-8")
        except (URLError, TimeoutError, IncompleteRead) as e:
            print(url, e)
    return None


def raw_paths(raw_dir: Path, tag: str) -> tuple[Path, Path]:
    """Return the paths of the raw table and of the log table for a SPOT tag.

    The raw table has one row for each component of each downloaded log.
    The log table has one row for each downloaded log, also if the log has no
    PerfMonMTSvc summary, so that the next run does not download it again.
    """
    return raw_dir / f"{tag}.csv", raw_dir / f"{tag}_logs.csv"


def download(
    *, tag: str, dates: list[date], step: str, raw_dir: Path, retries: int, refetch: bool
) -> None:
    """Download the logs of a SPOT tag that are not in the raw table yet."""
    raw_path, logs_path = raw_paths(raw_dir, tag)
    if raw_path.exists() and logs_path.exists() and not refetch:
        raw = pd.read_csv(raw_path, parse_dates=["date"])
        logs = pd.read_csv(logs_path, parse_dates=["date"])
        raw["date"] = raw["date"].dt.date
        logs["date"] = logs["date"].dt.date
    else:
        raw = pd.DataFrame(columns=["date", "step", "component", "count", "cpu", "vmem", "malloc"])
        logs = pd.DataFrame(columns=["date", "url", "components"])

    index = get_log_index(tag=tag)
    missing = [(d, index[d]) for d in dates if d in index and d not in set(logs["date"])]
    print(f"{tag}: {len(index)} logs available, {len(missing)} to download")
    if len(missing) == 0:
        return

    def work(job: tuple[date, str]) -> tuple[date, str, pd.DataFrame] | None:
        d, url = job
        txt = load_log(url, retries)
        if txt is None:
            return None
        components = parse_spot(txt)
        components = components[components["step"] == step]
        components.insert(0, "date", d)
        return d, url, components

    new_raw = []
    new_logs = []
    with ThreadPoolExecutor(16) as tp:
        for i, result in enumerate(tp.map(work, missing), start=1):
            if result is not None:
                d, url, components = result
                new_raw.append(components)
                new_logs.append({"date": d, "url": url, "components": len(components)})
            if i % 100 == 0 or i == len(missing):
                print(f"{tag}: {i}/{len(missing)} downloaded")

    failed = len(missing) - len(new_logs)
    if failed > 0:
        print(f"{tag}: {failed} logs failed, the next run tries them again")

    raw = pd.concat([df for df in [raw, *new_raw] if len(df) > 0], ignore_index=True)
    logs = pd.concat([logs, pd.DataFrame(new_logs)], ignore_index=True)
    raw = raw.sort_values(["date", "step", "component"], kind="stable")
    logs = logs.sort_values("date")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)
    logs.to_csv(logs_path, index=False)


def get_sources(flavor_config: dict) -> list[dict]:
    """Return the SPOT tags of a flavor with their date ranges.

    A flavor has either one `tag` for all dates, or a list of `sources`. Each
    source has a `tag` and an optional `start` and `end` (both inclusive).
    With sources, a flavor can follow one configuration when it moves from
    one SPOT job to another.
    """
    if "tag" in flavor_config:
        assert "sources" not in flavor_config
        return [{"tag": flavor_config["tag"]}]
    sources = flavor_config["sources"]
    for a, b in itertools.combinations(sources, 2):
        a_start = a.get("start", "0000-00-00")
        a_end = a.get("end", "9999-99-99")
        b_start = b.get("start", "0000-00-00")
        b_end = b.get("end", "9999-99-99")
        if a_start <= b_end and b_start <= a_end:
            raise ValueError(f"Sources overlap: {a} and {b}")
    return sources


def extract(
    *, flavor: str, flavor_config: dict, dates: list[date], step: str, raw_dir: Path
) -> dict[str, pd.DataFrame]:
    """Compute the points of a flavor from the raw tables of its SPOT tags."""
    raws = []
    for source in get_sources(flavor_config):
        raw_path, _ = raw_paths(raw_dir, source["tag"])
        raw = pd.read_csv(raw_path, parse_dates=["date"])
        raw["date"] = raw["date"].dt.date
        in_range = [d for d in dates if is_masked(d, [source])]
        raw = raw[(raw["step"] == step) & raw["date"].isin(set(in_range))]
        raw.insert(1, "tag", source["tag"])
        raws.append(raw)
    raw = pd.concat(raws, ignore_index=True)

    points_per_algorithm = defaultdict(list)
    for (d, tag), components in raw.groupby(["date", "tag"]):
        if is_masked(d, flavor_config.get("mask", [])):
            continue
        points = extract_points_per_algorithm(d, components, flavor_config["points"])
        for algorithm, point in points.items():
            points_per_algorithm[algorithm].append({**point, "tag": tag})

    result = {}
    for algorithm, points in points_per_algorithm.items():
        result[algorithm] = pd.DataFrame(points).dropna()
    print(f"{flavor}: {raw['date'].nunique()} dates, {len(result)} points")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/fetcher.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="data",
        help="Directory to store the data",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download logs, only compute the points from the raw tables",
    )
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="Download all logs again, also the logs that are in the raw tables",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    start_date = date.fromisoformat(config["dates"]["start"])
    end_date = date.fromisoformat(config["dates"]["end"])
    dates = date_range(start_date, end_date)
    step: str = config["step"]
    raw_dir: Path = args.data_dir / "raw"

    if not args.skip_download:
        tags = sorted(
            {s["tag"] for c in config["flavors"].values() for s in get_sources(c)}
        )
        for tag in tags:
            download(
                tag=tag,
                dates=dates,
                step=step,
                raw_dir=raw_dir,
                retries=3,
                refetch=args.refetch,
            )

    args.data_dir.mkdir(parents=True, exist_ok=True)
    for flavor, flavor_config in config["flavors"].items():
        points_per_algorithm = extract(
            flavor=flavor,
            flavor_config=flavor_config,
            dates=dates,
            step=step,
            raw_dir=raw_dir,
        )
        for algorithm, data in points_per_algorithm.items():
            data.to_csv(args.data_dir / f"{flavor}_{algorithm}.csv", index=False)


if __name__ == "__main__":
    main()
