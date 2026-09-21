import re
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor
import itertools
import urllib.request
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
import argparse
from pathlib import Path
import yaml
import pandas as pd
from spotter.parser import extract_all_points
from spotter.utilities import date_range
from spotter.spot_data_config import SpotConfig, SpotData


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


def do_fetch(data: SpotData, retries: int, indices: dict[str, dict[date, str]]) -> None:
    assert isinstance(data, SpotData)
    assert isinstance(retries, int)

    def load(u: str, retries: int) -> None | str:
        assert isinstance(u, str)
        assert isinstance(retries, int)
        for _ in range(retries):
            try:
                return urllib.request.urlopen(u, timeout=120).read().decode("utf-8")
            except (URLError, TimeoutError, IncompleteRead) as e:
                print(d, u, e)
        return None

    d: date = data.date
    for flavor, config in data.spot_configs.items():
        url = indices[config.tag].get(d)
        data.add_data(flavor=flavor, data=None if url is None else load(url, retries))


def do_fetch_all(*, data: list[SpotData], retries: int) -> None:
    assert isinstance(data, list)
    assert isinstance(retries, int)
    tags = {c.tag for sd in data for c in sd.spot_configs.values()}
    indices = {tag: get_log_index(tag=tag) for tag in tags}
    for tag, index in indices.items():
        print(f"{tag}: {len(index)} logs available")
    with ThreadPoolExecutor() as tp:
        # consume the results so that exceptions in the workers are raised
        list(tp.map(do_fetch, data, itertools.repeat(retries), itertools.repeat(indices)))


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
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    start_date = date.fromisoformat(config["dates"]["start"])
    end_date = date.fromisoformat(config["dates"]["end"])
    date_list = date_range(start_date, end_date)

    flavors: list[str] = list(config["flavors"].keys())
    step: str = config["step"]

    spot_configs: list[SpotConfig] = []
    for flavor in flavors:
        spot_configs.append(
            SpotConfig(
                flavor,
                config["flavors"][flavor]["tag"],
                config["flavors"][flavor]["points"],
            )
        )

    spot_data: list[SpotData] = []
    for d in date_list:
        data = SpotData(d, step)
        for conf in spot_configs:
            data.add_spot_config(spot_config=conf)
        spot_data.append(data)

    # Fetch the log files
    do_fetch_all(data=spot_data, retries=3)
    data_per_algorithm_per_flavor: dict[
        str, dict[str, dict[str, date | pd.DataFrame]]
    ] = extract_all_points(spot_data)

    args.data_dir.mkdir(parents=True, exist_ok=True)

    for flavor, data_per_algorithm in data_per_algorithm_per_flavor.items():
        for algorithm, data in data_per_algorithm.items():
            data.to_csv(f"{args.data_dir}/{flavor}_{algorithm}.csv", index=False)


if __name__ == "__main__":
    main()
