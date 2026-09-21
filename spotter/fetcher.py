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
from spotter.parser import (
    parse_spot,
    parse_validation,
    extract_points_per_algorithm,
    is_masked,
)
from spotter.utilities import date_range


BASE_URL = "https://atlaspmb.web.cern.ch/atlaspmb"

# the version of the information that the download step reads from a log; a log
# from an older version is downloaded again (1: host, 2: validation tables and
# negative memory values)
FETCH_VERSION = 2
LOG_DIRS = ["logs", "logs/archive_centos7"]


def get_log_index(*, tag: str, el9_from: date = date.min) -> dict[date, str]:
    """Map each date to the URL of the `main` RAWtoALL log for a SPOT tag.

    The platform and the compiler in the log name change over time
    (centos7-gcc11, el9-gcc13, el9-gcc14, el9-gcc15). If a date has more than
    one log, the log with the newest compiler wins. Before `el9_from`, a
    centos7 log wins over an el9 log, because el9 is faster on the same code
    and most dates before the switch only have a centos7 log.
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
            prefer_el9 = d >= el9_from
            rank = ((m.group(2) == "el9") == prefer_el9, int(m.group(3)))
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


def raw_paths(raw_dir: Path, tag: str) -> tuple[Path, Path, Path]:
    """Return the paths of the raw tables and of the log table for a SPOT tag.

    The raw table has one row for each component of each downloaded log. The
    validation table has the summary tables of the track finding validation.
    The log table has one row for each downloaded log, also if the log has no
    PerfMonMTSvc summary, so that the next run does not download it again.
    """
    return (
        raw_dir / f"{tag}.csv",
        raw_dir / f"{tag}_logs.csv",
        raw_dir / f"{tag}_validation.csv",
    )


def read_table(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a raw table, with the date as a date, or return an empty table."""
    if not path.exists():
        return pd.DataFrame(columns=columns)
    table = pd.read_csv(path, keep_default_na=False, na_values=[""])
    table["date"] = pd.to_datetime(table["date"]).dt.date
    return table


def download(
    *,
    tag: str,
    dates: list[date],
    step: str,
    raw_dir: Path,
    retries: int,
    refetch: bool,
    el9_from: date,
) -> None:
    """Download the logs of a SPOT tag that are not in the raw table yet.

    The function also downloads a date again if the selected log of the date
    changed, for example after a change of `el9_from`.
    """
    raw_path, logs_path, validation_path = raw_paths(raw_dir, tag)
    raw_columns = ["date", "step", "count", "cpu", "vmem", "malloc", "component"]
    logs_columns = ["date", "url", "host", "components", "version"]
    validation_columns = ["date", "alg", "table", "eta", "pt", "value"]
    if refetch:
        raw = pd.DataFrame(columns=raw_columns)
        logs = pd.DataFrame(columns=logs_columns)
        validation = pd.DataFrame(columns=validation_columns)
    else:
        raw = read_table(raw_path, raw_columns)
        logs = read_table(logs_path, logs_columns)
        validation = read_table(validation_path, validation_columns)
    if "version" not in logs:
        logs["version"] = 0

    index = get_log_index(tag=tag, el9_from=el9_from)
    known = dict(zip(logs["date"], logs["url"]))
    old = set(logs.loc[logs["version"] < FETCH_VERSION, "date"])
    missing = [
        (d, index[d])
        for d in dates
        if d in index and (known.get(d) != index[d] or d in old)
    ]
    print(f"{tag}: {len(index)} logs available, {len(missing)} to download")
    if len(missing) == 0:
        return

    def work(
        job: tuple[date, str],
    ) -> tuple[date, str, str, pd.DataFrame, pd.DataFrame] | None:
        d, url = job
        txt = load_log(url, retries)
        if txt is None:
            return None
        # the machine that ran the job, because a new machine changes all timings
        host = re.search(r"running on (\S+) on ", txt)
        components = parse_spot(txt)
        components = components[components["step"] == step]
        components.insert(0, "date", d)
        tables = parse_validation(txt)
        tables.insert(0, "date", d)
        return d, url, host.group(1) if host else "unknown", components, tables

    new_raw = []
    new_logs = []
    new_validation = []
    with ThreadPoolExecutor(16) as tp:
        for i, result in enumerate(tp.map(work, missing), start=1):
            if result is not None:
                d, url, host, components, tables = result
                new_raw.append(components)
                new_validation.append(tables)
                new_logs.append(
                    {
                        "date": d,
                        "url": url,
                        "host": host,
                        "components": len(components),
                        "version": FETCH_VERSION,
                    }
                )
            if i % 100 == 0 or i == len(missing):
                print(f"{tag}: {i}/{len(missing)} downloaded")

    failed = len(missing) - len(new_logs)
    if failed > 0:
        print(f"{tag}: {failed} logs failed, the next run tries them again")

    # replace the rows of the dates that were downloaded again
    replaced = {log["date"] for log in new_logs}
    raw = raw[~raw["date"].isin(replaced)]
    logs = logs[~logs["date"].isin(replaced)]
    validation = validation[~validation["date"].isin(replaced)]
    raw = pd.concat([df for df in [raw, *new_raw] if len(df) > 0], ignore_index=True)
    logs = pd.concat([df for df in [logs, pd.DataFrame(new_logs)] if len(df) > 0], ignore_index=True)
    # the legacy jobs have no validation tables, so the list can be empty
    tables = [df for df in [validation, *new_validation] if len(df) > 0]
    validation = pd.concat(tables, ignore_index=True) if tables else validation
    raw = raw.sort_values(["date", "step", "component"], kind="stable")
    logs = logs.sort_values("date")
    if len(validation) > 0:
        validation = validation.sort_values(["date", "alg", "table"], kind="stable")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)
    logs.to_csv(logs_path, index=False)
    validation.to_csv(validation_path, index=False)


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
    *,
    flavor: str,
    flavor_config: dict,
    dates: list[date],
    step: str,
    raw_dir: Path,
    masks: list[dict],
) -> dict[str, pd.DataFrame]:
    """Compute the points of a flavor from the raw tables of its SPOT tags.

    `masks` apply in addition to the masks of the flavor.
    """
    raws = []
    validations = []
    for source in get_sources(flavor_config):
        raw_path, logs_path, validation_path = raw_paths(raw_dir, source["tag"])
        in_range = set(d for d in dates if is_masked(d, [source]))

        # the platform of the log, for example el9-gcc14, and the SPOT machine
        logs = read_table(logs_path, [])
        logs["platform"] = logs["url"].str.extract(r"-main-x86_64-(\w+-gcc\d+)-opt-")
        logs["host"] = logs["host"].str.replace(".cern.ch", "", regex=False)
        logs = logs[["date", "platform", "host"]]

        raw = read_table(raw_path, [])
        raw = raw[(raw["step"] == step) & raw["date"].isin(in_range)]
        raw.insert(1, "tag", source["tag"])
        raws.append(raw.merge(logs, on="date", how="left"))

        validation = read_table(validation_path, ["date", "alg", "table", "eta", "pt", "value"])
        validation = validation[validation["date"].isin(in_range)]
        validation.insert(1, "tag", source["tag"])
        validations.append(validation.merge(logs, on="date", how="left"))
    raw = pd.concat([df for df in raws if len(df) > 0] or raws[:1], ignore_index=True)
    validation = pd.concat(
        [df for df in validations if len(df) > 0] or validations[:1], ignore_index=True
    )

    masks = masks + flavor_config.get("mask", [])
    points_per_algorithm = defaultdict(list)
    for (d, tag, platform, host), components in raw.groupby(
        ["date", "tag", "platform", "host"]
    ):
        if is_masked(d, masks):
            continue
        points = extract_points_per_algorithm(d, components, flavor_config["points"])
        for algorithm, point in points.items():
            points_per_algorithm[algorithm].append(
                {**point, "tag": tag, "platform": platform, "host": host}
            )

    result = {}
    for algorithm, points in points_per_algorithm.items():
        result[algorithm] = pd.DataFrame(points).dropna()

    # the points from the validation tables, for example the reco efficiency
    for point in flavor_config.get("validation", []):
        rows = validation[
            validation["alg"].str.contains(point["alg"])
            & (validation["table"] == point["table"])
            & (validation["eta"] == point.get("eta", "All eta"))
            & (validation["pt"].fillna("") == point.get("pt", "All pt"))
        ]
        point_masks = masks + point.get("mask", [])
        rows = rows[~rows["date"].map(lambda d: is_masked(d, point_masks)).astype(bool)]
        duplicated = rows[rows["date"].duplicated(keep=False)]
        if len(duplicated) > 0:
            raise ValueError(
                f"{flavor}: '{point['alg']}' matches more than one algorithm: "
                f"{sorted(duplicated['alg'].unique())}"
            )
        result[point["name"]] = rows[["date", "value", "tag", "platform", "host"]]

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
                el9_from=date.fromisoformat(config.get("el9_from", "0001-01-01")),
            )

    args.data_dir.mkdir(parents=True, exist_ok=True)
    for flavor, flavor_config in config["flavors"].items():
        points_per_algorithm = extract(
            flavor=flavor,
            flavor_config=flavor_config,
            dates=dates,
            step=step,
            raw_dir=raw_dir,
            masks=config.get("mask", []),
        )
        for algorithm, data in points_per_algorithm.items():
            data.to_csv(args.data_dir / f"{flavor}_{algorithm}.csv", index=False)


if __name__ == "__main__":
    main()
