import re
from datetime import date
from collections import defaultdict
from spotter.spot_data_config import spot_data

import pandas as pd


def parse_spot(txt: str) -> pd.core.frame.DataFrame:
    assert isinstance(txt, str)
    p = re.compile(
        r"\d\d:\d\d:\d\d PerfMonMTSvc\s+INFO (\w+?)\s+(\d+)\s+(\d+(?:\.\d*)?)\s+(\d+(?:\.\d*)?)\s+(\d+(?:\.\d*)?)\s+(.+)"
    )
    columns = ["step", "count", "cpu", "vmem", "malloc", "component"]
    data: list[tuple[str]] = []

    for line in txt.splitlines():
        m = p.match(line)
        if not m:
            continue
        data.append(m.groups())

    result = pd.DataFrame(data, columns=columns)
    result[["count"]] = result[["count"]].astype(int)
    result[["cpu", "vmem", "malloc"]] = result[["cpu", "vmem", "malloc"]].astype(float)
    return result


def sum_two_points(
    a: dict[str, date | float], b: dict[str, date | float]
) -> dict[str, date | float]:
    assert isinstance(a, dict)
    assert isinstance(b, dict)
    assert a["date"] == b["date"]
    return {
        "date": a["date"],
        "cpu": a["cpu"] + b["cpu"],
        "vmem": a["vmem"] + b["vmem"],
        "malloc": a["malloc"] + b["malloc"],
    }


def convert_point(d: date, p: pd.core.frame.DataFrame) -> dict[str, date | float]:
    assert isinstance(d, date)
    assert isinstance(p, pd.core.frame.DataFrame)

    points = []
    for s in p.iloc:
        toAdd = {
            "date": d,
            "cpu": s["cpu"] / s["count"] * 1e-3,
            "vmem": s["vmem"] * 1e-3,
            "malloc": s["malloc"] / s["count"] * 1e-3,
        }
        points.append(toAdd)

    assert len(points) > 0
    for i in range(1, len(points)):
        points[0] = sum_two_points(points[0], points[i])
    return points[0]


def null_point(d: date) -> dict[str, date | None]:
    assert isinstance(d, date)
    return {
        "date": d,
        "cpu": None,
        "vmem": None,
        "malloc": None,
    }


def extract_components(
    data: pd.core.frame.DataFrame, components: list[str]
) -> pd.core.frame.DataFrame:
    assert isinstance(data, pd.core.frame.DataFrame)
    assert isinstance(components, list)
    assert len(components) != 0 and isinstance(components[0], str)

    output = data[(data["component"].isin(components))]
    if len(output) == 0:
        raise Exception(f"elements not found with compoennts in {components}:\n{data}")
    return output


def extract(
    d: date, data: pd.core.frame.DataFrame, components: list[str]
) -> dict[str, dict[str, date | float | None]]:
    assert isinstance(d, date)
    assert isinstance(data, pd.core.frame.DataFrame)
    assert isinstance(components, list)
    assert len(components) != 0 and isinstance(components[0], str)

    track_point: dict[str, date | float] = convert_point(
        d, extract_components(data, components)
    )
    return {
        "track_finding": track_point,
    }


def extract_points(data: spot_data) -> dict[str, str | None]:
    assert isinstance(data, spot_data)

    result = dict()
    for flavor, logText in data.data.items():
        if logText is None:
            continue
        # Take the summary
        summary_algos: pd.core.frame.DataFrame = parse_spot(logText)
        if len(summary_algos) == 0:
            continue

        # Take the step we need
        spot_exec_data: pd.core.frame.DataFrame = summary_algos[
            summary_algos["step"] == data.step
        ]
        if len(spot_exec_data) == 0:
            continue

        # Take the algo we need
        points = extract(data.date, spot_exec_data, data.getComponents(flavor=flavor))
        result[flavor] = points
    return result


def extract_all_points(
    data: list[spot_data],
) -> dict[str, dict[str, dict[str, date | pd.core.frame.DataFrame]]]:
    assert isinstance(data, list)

    flavors = data[0].data.keys()
    data_per_algorithm_per_flavor = {flavor: defaultdict(list) for flavor in flavors}

    for sd in data:
        assert isinstance(sd, spot_data)
        points_per_algorithm_per_flavor = extract_points(sd)

        for flavor, points_per_flavor in points_per_algorithm_per_flavor.items():
            for algorithm, point in points_per_flavor.items():
                data_per_algorithm_per_flavor[flavor][algorithm].append(point)

    for flavor, data_per_algorithm in data_per_algorithm_per_flavor.items():
        for algorithm in data_per_algorithm:
            data_per_algorithm[algorithm] = pd.DataFrame(
                data_per_algorithm[algorithm]
            ).dropna()
            d = data_per_algorithm[algorithm]

        d["date"] = pd.to_datetime(d["date"])
        d.index = d["date"]

    return data_per_algorithm_per_flavor
