import re
from datetime import date

import pandas as pd


def parse_spot(txt: str) -> pd.DataFrame:
    assert isinstance(txt, str)
    p = re.compile(
        # vmem and malloc can be negative, for example if an algorithm frees memory
        r"\d\d:\d\d:\d\d PerfMonMTSvc\s+INFO (\w+?)\s+(\d+)\s+(\d+(?:\.\d*)?)\s+(-?\d+(?:\.\d*)?)\s+(-?\d+(?:\.\d*)?)\s+(.+)"
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


VALIDATION_TABLES = {
    "Hit efficiency": "hit_efficiency",
    "Hit purity": "hit_purity",
    "Match probability": "match_probability",
    "Reco efficiency": "reco_efficiency",
}


def parse_validation(txt: str) -> pd.DataFrame:
    """Read the summary tables of the ACTS track finding validation algorithms.

    Each algorithm (for example ActsTracksTrackFindingValidationAlg) prints
    tables of the hit efficiency, the hit purity, the match probability and the
    reco efficiency in bins of eta and pT, and a table of counts. The result
    keeps the "All pt" column, the "All eta" row and the counts, with the
    columns alg, table, eta, pt and value.
    """
    assert isinstance(txt, str)
    time = re.compile(r"^\d\d:\d\d:\d\d ")
    alg_line = re.compile(r"^(\w+ValidationAlg)\s+INFO")
    rows = []
    alg = table = None
    header: list[str] = []
    for line in txt.splitlines():
        line = time.sub("", line)
        m = alg_line.match(line)
        if m:
            alg, table = m.group(1), None
            continue
        if alg is None:
            continue
        title = line.split(" :")[0].strip()
        if title in VALIDATION_TABLES:
            table = VALIDATION_TABLES[title]
            continue
        if not line.startswith("|") or line.startswith("|-"):
            if line.strip() and not line.startswith("|"):
                # any other text ends the output of the algorithm
                alg = table = None
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0].startswith("eta"):
            header = cells[1:]
        elif len(cells) == 2:
            try:
                rows.append((alg, "counts", cells[0], "", float(cells[1])))
            except ValueError:
                pass
        elif table is not None and len(cells) == len(header) + 1:
            eta = cells[0]
            for pt, value in zip(header, cells[1:]):
                if eta == "All eta" or pt == "All pt":
                    rows.append((alg, table, eta, pt, float(value)))
    return pd.DataFrame(rows, columns=["alg", "table", "eta", "pt", "value"])


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


def is_masked(d: date, masks: list[dict]) -> bool:
    """Return True if a mask covers the date.

    A mask has an optional `start` and an optional `end` (both inclusive),
    and a `reason`. A missing `start` or `end` leaves that side open.
    """
    assert isinstance(d, date)
    for mask in masks:
        start = date.fromisoformat(mask["start"]) if "start" in mask else date.min
        end = date.fromisoformat(mask["end"]) if "end" in mask else date.max
        if start <= d <= end:
            return True
    return False


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

    # an optional point does not exist on all dates (or has a mask), so a sum counts it as zero there
    optional = {p["name"] for p in points_mapping if p.get("optional", False)}

    points_per_algorithm = {}
    for point_mapping in points_mapping:
        name = point_mapping["name"]

        if is_masked(d, point_mapping.get("mask", [])):
            point = null_point(d)
        elif "map" in point_mapping:
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
