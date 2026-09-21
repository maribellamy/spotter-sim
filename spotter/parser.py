import re
from datetime import date
from collections import defaultdict
from spotter.spot_data_config import SpotData

import pandas as pd


def parse_spot(txt: str) -> pd.DataFrame:
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


def zero_point(d: date) -> dict[str, date | float]:
    assert isinstance(d, date)
    return {
        "date": d,
        "cpu": 0,
        "vmem": 0,
        "malloc": 0,
    }


def null_point(d: date) -> dict[str, date | float]:
    assert isinstance(d, date)
    return {
        "date": d,
        "cpu": float("nan"),
        "vmem": float("nan"),
        "malloc": float("nan"),
    }


def convert_point(d: date, p: pd.DataFrame) -> dict[str, date | float]:
    assert isinstance(d, date)
    assert isinstance(p, pd.DataFrame)
    assert len(p) == 1
    one = p.iloc[0]
    return {
        "date": d,
        "cpu": one["cpu"] / one["count"] * 1e-3,
        "vmem": one["vmem"] * 1e-3,
        "malloc": one["malloc"] / one["count"] * 1e-3,
    }


def sum_points(*points: dict[str, date | float]) -> dict[str, date | float]:
    for point in points:
        assert isinstance(point, dict)
        assert isinstance(point, dict)
        assert points[0]["date"] == point["date"]
    return {
        "date": points[0]["date"],
        "cpu": sum(point["cpu"] for point in points),
        "vmem": sum(point["vmem"] for point in points),
        "malloc": sum(point["malloc"] for point in points),
    }


def extract_points_per_algorithm(
    d: date, data: pd.DataFrame, points_mapping: list
) -> dict[str, dict[str, date | float]]:
    assert isinstance(d, date)
    assert isinstance(data, pd.DataFrame)
    assert isinstance(points_mapping, list)

    def extract_one(d: date, data: pd.DataFrame, name: str):
        r = data[data["component"].str.contains(name)]
        if len(r) == 0:
            return null_point(d)
        if len(r) > 1:
            raise ValueError(
                f"{d}: '{name}' matches more than one component: {list(r['component'])}"
            )
        return convert_point(d, r)

    # an optional component does not run on all dates, so a sum counts it as zero there
    optional = {p["name"] for p in points_mapping if p.get("optional", False)}

    points_per_algorithm = {}
    for point_mapping in points_mapping:
        name = point_mapping["name"]

        if "map" in point_mapping:
            point = extract_one(d, data, point_mapping["map"])
        elif "sum" in point_mapping:
            points = []
            for n in point_mapping["sum"]:
                if n in points_per_algorithm:
                    point = points_per_algorithm[n]
                else:
                    point = extract_one(d, data, n)
                if n in optional and pd.isna(point["cpu"]):
                    point = zero_point(d)
                points.append(point)
            point = sum_points(*points)
        else:
            raise ValueError(f"Unknown mapping: {point_mapping}")

        points_per_algorithm[name] = point

    return points_per_algorithm


def extract_points_per_algorithm_per_flavor(
    data: SpotData,
) -> dict[str, dict[str, date | float | None]]:
    assert isinstance(data, SpotData)

    data_per_algorithm_per_flavor = {}
    for flavor, log_text in data.data.items():
        if log_text is None:
            continue
        # Take the summary
        summary_algos: pd.DataFrame = parse_spot(log_text)
        if len(summary_algos) == 0:
            continue

        # Take the step we need
        spot_exec_data: pd.DataFrame = summary_algos[summary_algos["step"] == data.step]
        if len(spot_exec_data) == 0:
            continue

        # Take the algo we need
        points_per_algorithm = extract_points_per_algorithm(
            data.date, spot_exec_data, data.get_points(flavor=flavor)
        )
        data_per_algorithm_per_flavor[flavor] = points_per_algorithm
    return data_per_algorithm_per_flavor


def extract_all_points(
    data: list[SpotData],
) -> dict[str, dict[str, dict[str, date | pd.DataFrame]]]:
    assert isinstance(data, list)

    flavors = data[0].data.keys()
    data_per_algorithm_per_flavor = {flavor: defaultdict(list) for flavor in flavors}

    for sd in data:
        assert isinstance(sd, SpotData)
        points_per_algorithm_per_flavor = extract_points_per_algorithm_per_flavor(sd)

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
