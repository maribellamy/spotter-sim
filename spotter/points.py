"""The points of the fetcher config, and which sums contain which parts.

A sum point (for example `track_finding`) adds its parts (`seeding` and
`ckf`), and a part can be a sum itself. An annotation of a part also belongs
to every sum that contains it, so that one annotation of the CKF shows on the
track finding plot and on the full chain plot.
"""

from pathlib import Path

import pandas as pd
import yaml


def load_fetcher_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def transitive_parts(fetcher_config: dict) -> dict[str, dict[str, set[str]]]:
    """Return the parts of each sum point of each flavor, through nested sums."""
    result = {}
    for flavor, config in fetcher_config["flavors"].items():
        sums = {p["name"]: p["sum"] for p in config["points"] if "sum" in p}
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
    """The dates with a value of each point of each flavor, read on demand from the point CSVs."""

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
    """Return True if an annotation of the central file belongs to a plotted line.

    The annotation belongs to the line if it lists the flavor, and the point
    or one of the parts of the point (`parts` from `transitive_parts`). With
    `point_data`, a part counts only if it has a value in the flavor around
    the date of the annotation: the strip seeding is a part of the ACTS
    default track finding, but it does not run in the fast configuration.
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
