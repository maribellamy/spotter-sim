from datetime import date
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from urllib.error import HTTPError
import argparse
from pathlib import Path
import yaml
import pandas as pd
from spotter.parser import extract_all_points
from spotter.utilities import date_range
from spotter.spot_data_config import spot_config, spot_data


def get_url(*,
            d: date,
            flavor: str) -> str:
    assert isinstance(d, date)
    assert isinstance(flavor, str)
    date_str = d.strftime("%d-%m-%Y")
    base = f"https://atlaspmb.web.cern.ch/atlaspmb/spot-mon-phase2_recoonly_{flavor}/pages/logs"
    binary = "main-x86_64-el9-gcc13-opt"
    job = f"phase2_recoonly_{flavor}-RAWtoALL"
    if d < date.fromisoformat("2024-03-01"):
        base += "/archive_centos7"
        binary = "main-x86_64-centos7-gcc11-opt"
    return f"{base}/{date_str}-{binary}-{job}"


def do_fetch(data: spot_data) -> None:
    assert isinstance(data, spot_data)

    def load(u: str) -> None | str:
        assert isinstance(u, str)
        try:
            return urllib.request.urlopen(u).read().decode("utf-8")
        except HTTPError as e:
            print(d, u, e)
        return None

    d: date = data.date
    for (flavor, config) in data.spot_configs.items():
        data.addData(flavor=flavor,
                     data=load(get_url(d=d, flavor=config.tag)))


def do_fetch_all(*,
                 data: list[spot_data]) -> None:
    assert isinstance(data, list)
    with ThreadPoolExecutor() as tp:
        tp.map(do_fetch, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
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
    step: str = config['step']

    spotConfs: list[spot_config] = []
    for flavor in flavors:
        spotConfs.append(spot_config(flavor,
                                     config['flavors'][flavor]['tag'],
                                     config['flavors'][flavor]['components']))

    spotData: list[spot_data] = []
    for d in date_list:
        data = spot_data(d, step)
        for conf in spotConfs:
            data.addSpotConfiguration(spotConfig=conf)
        spotData.append(data)

    # Fetch the log files
    do_fetch_all(data=spotData)
    data_per_algorithm_per_flavor: dict[str, dict[str, dict[str, date | pd.core.frame.DataFrame]]] = extract_all_points(spotData)

    for flavor, data_per_algorithm in data_per_algorithm_per_flavor.items():
        for algorithm, data in data_per_algorithm.items():
            data.to_csv(f"{args.data_dir}/{flavor}_{algorithm}.csv", index=False)


if __name__ == "__main__":
    main()
