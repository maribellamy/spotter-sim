import re
from datetime import date
from collections import defaultdict

import pandas as pd


def parse_spot(txt: str) -> pd.DataFrame:
    p = re.compile(
        r"\d\d:\d\d:\d\d PerfMonMTSvc\s+INFO (\w+?)\s+(\d+)\s+(\d+(?:\.\d*)?)\s+(\d+(?:\.\d*)?)\s+(\d+(?:\.\d*)?)\s+(.+)"
    )
    columns = ["step", "count", "cpu", "vmem", "malloc", "component"]
    data = []

    for line in txt.splitlines():
        m = p.match(line)
        if not m:
            continue
        data.append(m.groups())

    result = pd.DataFrame(data, columns=columns)
    result[["count"]] = result[["count"]].astype(int)
    result[["cpu", "vmem", "malloc"]] = result[["cpu", "vmem", "malloc"]].astype(float)
    return result


def convert_point(d: date, p: dict) -> dict:
    return {
        "date": d,
        "cpu": p["cpu"] / p["count"] * 1e-3,
        "vmem": p["vmem"] * 1e-3,
        "malloc": p["malloc"] / p["count"] * 1e-3,
    }


def sum_two_points(a: dict, b: dict) -> dict:
    if a["date"] != b["date"]:
        raise Exception("mixed dates")
    return {
        "date": a["date"],
        "cpu": a["cpu"] + b["cpu"],
        "vmem": a["vmem"] + b["vmem"],
        "malloc": a["malloc"] + b["malloc"],
    }


def null_point(d: date) -> dict:
    return {
        "date": d,
        "cpu": None,
        "vmem": None,
        "malloc": None,
    }


def extract_athena_track_finding_point(data: pd.DataFrame):
    data = data[(data["component"] == "ITkSiSpTrackFinder")]
    if len(data) == 0:
        raise Exception("not found")
    return data.iloc[0]


def extract_acts_pixel_seeding_point(data: pd.DataFrame):
    data = data[
        (
            (data["component"] == "ActsPixelSeedingAlg")
            | (data["component"] == "ActsBenchmarkSpotPixelSeedingAlg")
        )
    ]
    if len(data) == 0:
        raise Exception("not found")
    return data.iloc[0]


def extract_acts_strip_seeding_point(data: pd.DataFrame):
    data = data[
        (
            (data["component"] == "ActsStripSeedingAlg")
            | (data["component"] == "ActsBenchmarkSpotStripSeedingAlg")
        )
    ]
    if len(data) == 0:
        raise Exception("not found")
    return data.iloc[0]


def extract_acts_ckf_point(data: pd.DataFrame):
    data = data[
        (
            (data["component"] == "ActsTrackFindingAlg")
            | (data["component"] == "ActsBenchmarkSpotTrackFindingAlg")
        )
    ]
    if len(data) == 0:
        raise Exception("not found")
    return data.iloc[0]


def extract_athena_points(d: date, data: pd.DataFrame):
    null = null_point(d)

    try:
        track_finding_point = convert_point(d, extract_athena_track_finding_point(data))
    except Exception as e:
        track_finding_point = null
        print(d, "no track finding data")

    return {
        "track_finding": track_finding_point,
    }


def extract_acts_default_points(d: date, data: pd.DataFrame) -> dict:
    null = null_point(d)

    try:
        pixel_seeding_point = convert_point(d, extract_acts_pixel_seeding_point(data))
    except Exception as e:
        pixel_seeding_point = null
        print(d, "no pixel seeding data")

    try:
        strip_seeding_point = convert_point(d, extract_acts_strip_seeding_point(data))
    except Exception as e:
        strip_seeding_point = null
        print(d, "no strip seeding data")

    if pixel_seeding_point is not null and strip_seeding_point is not null:
        seeding_point = sum_two_points(pixel_seeding_point, strip_seeding_point)
    else:
        seeding_point = null

    try:
        ckf_point = convert_point(d, extract_acts_ckf_point(data))
    except Exception as e:
        ckf_point = null
        print(d, "no ckf data")

    if seeding_point is not null and ckf_point is not null:
        track_finding_point = sum_two_points(seeding_point, ckf_point)
    else:
        track_finding_point = null

    return {
        "pixel_seeding": pixel_seeding_point,
        "strip_seeding": strip_seeding_point,
        "seeding": seeding_point,
        "ckf": ckf_point,
        "track_finding": track_finding_point,
    }


def extract_acts_fast_points(d: date, data: pd.DataFrame) -> dict:
    null = null_point(d)

    try:
        seeding_point = convert_point(d, extract_acts_pixel_seeding_point(data))
    except Exception as e:
        seeding_point = null
        print(d, "no seeding data")

    try:
        ckf_point = convert_point(d, extract_acts_ckf_point(data))
    except Exception as e:
        ckf_point = null
        print(d, "no ckf data")

    if seeding_point is not null and ckf_point is not null:
        track_finding_point = sum_two_points(seeding_point, ckf_point)
    else:
        track_finding_point = null

    return {
        "seeding": seeding_point,
        "ckf": ckf_point,
        "track_finding": track_finding_point,
    }


def extract_points(d: date, txt: str, flavor: str) -> pd.DataFrame:
    if flavor == "legacy_default":
        legacy_default_spot_data = parse_spot(txt)
        legacy_default_spot_exec_data = legacy_default_spot_data[
            legacy_default_spot_data["step"] == "Execute"
        ]

        return extract_athena_points(d, legacy_default_spot_exec_data)

    if flavor == "legacy_fast":
        legacy_fast_spot_data = parse_spot(txt)
        legacy_fast_spot_exec_data = legacy_fast_spot_data[
            legacy_fast_spot_data["step"] == "Execute"
        ]

        legacy_fast_points = extract_athena_points(d, legacy_fast_spot_exec_data)

        return legacy_fast_points

    if flavor == "acts_default":
        acts_default_spot_data = parse_spot(txt)
        acts_default_spot_exec_data = acts_default_spot_data[
            acts_default_spot_data["step"] == "Execute"
        ]

        acts_default_points = extract_acts_default_points(
            d, acts_default_spot_exec_data
        )

        return acts_default_points

    if flavor == "acts_fast":
        acts_fast_spot_data = parse_spot(txt)
        acts_fast_spot_exec_data = acts_fast_spot_data[
            acts_fast_spot_data["step"] == "Execute"
        ]

        acts_fast_points = extract_acts_fast_points(d, acts_fast_spot_exec_data)

        return acts_fast_points

    raise Exception(f"unknown flavor: {flavor}")


def extract_all_points(fetch_results: list[tuple[date, str]], flavor: str) -> dict:
    data_per_algorithm = defaultdict(list)

    for d, results in fetch_results:
        points_per_algorithm = extract_points(d, results, flavor)
        for algorithm, point in points_per_algorithm.items():
            data_per_algorithm[algorithm].append(point)

    for algorithm in data_per_algorithm:
        data_per_algorithm[algorithm] = pd.DataFrame(
            data_per_algorithm[algorithm]
        ).dropna()
        d = data_per_algorithm[algorithm]

        d["date"] = pd.to_datetime(d["date"])
        d.index = d["date"]

    return data_per_algorithm
