"""The points of the fetcher config: base points, sums and annotation matching."""

from pathlib import Path

import pandas as pd
import yaml


def load_fetcher_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def sums_of(fetcher_config: dict, flavor: str) -> dict[str, list[str]]:
    """Return the parts of each sum point of the flavor, as in the config."""
    return {p["name"]: p["sum"] for p in fetcher_config["flavors"][flavor]["points"] if "sum" in p}


def base_points(fetcher_config: dict, flavor: str, point: str) -> list[str]:
    """Return the base points (`map` points) of a sum, through nested sums; a base point returns itself."""
    sums = sums_of(fetcher_config, flavor)
    if point not in sums:
        return [point]
    return [b for part in sums[point] for b in base_points(fetcher_config, flavor, part)]


def transitive_parts(fetcher_config: dict) -> dict[str, dict[str, set[str]]]:
    """Return the parts of each sum point of each flavor, through nested sums."""
    result = {}
    for flavor in fetcher_config["flavors"]:
        sums = sums_of(fetcher_config, flavor)
        parts = {}
        for name in sums:
            seen: set[str] = set()
            todo = list(sums[name])
            while todo:
                part = todo.pop()
                if part in seen:
                    continue
                seen.add(part)
                todo += sums.get(part, [])
            parts[name] = seen
        result[flavor] = parts
    return result


class PointData:
    """The dates with a value of each point, read on demand from the point CSVs."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.dates: dict[tuple[str, str], set] = {}

    def has_value(self, flavor: str, point: str, date, days: int = 3) -> bool:
        """Return True if the point of the flavor has a value within `days` of the date."""
        key = (flavor, point)
        if key not in self.dates:
            path = self.data_dir / f"{flavor}_{point}.csv"
            self.dates[key] = set(pd.to_datetime(pd.read_csv(path)["date"])) if path.exists() else set()
        d = pd.Timestamp(date)
        return any(abs((x - d).days) <= days for x in self.dates[key])


def matches(
    annotation: dict,
    flavor: str,
    point: str,
    parts: dict | None = None,
    point_data: PointData | None = None,
) -> bool:
    """Return True if an annotation belongs to the point of the flavor.

    The annotation lists the flavor, and the point or one of its parts
    (`parts` from `transitive_parts`). With `point_data`, a part counts only
    if it has a value in the flavor around the date of the annotation.
    """
    if flavor not in annotation.get("flavors", []):
        return False
    points = set(annotation.get("points", []))
    if point in points:
        return True
    related = points & (parts or {}).get(flavor, {}).get(point, set())
    if point_data is None:
        return bool(related)
    return any(point_data.has_value(flavor, p, annotation["date"]) for p in related)
