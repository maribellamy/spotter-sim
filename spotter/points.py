"""The points of the fetcher config, and which sums contain which parts.

A sum point (for example `track_finding`) adds its parts (`seeding` and
`ckf`), and a part can be a sum itself. An annotation of a part also belongs
to every sum that contains it, so that one annotation of the CKF shows on the
track finding plot and on the full chain plot.
"""

from pathlib import Path

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


def matches(annotation: dict, flavor: str, point: str, parts: dict | None = None) -> bool:
    """Return True if an annotation of the central file belongs to a plotted line.

    The annotation belongs to the line if it lists the flavor, and the point
    or one of the parts of the point (`parts` from `transitive_parts`).
    """
    points = set(annotation.get("points", []))
    related = {point} | (parts or {}).get(flavor, {}).get(point, set())
    return flavor in annotation.get("flavors", []) and bool(points & related)
