"""The SPOT machines of config/machines.yaml, and the normalization of the CPU times.

One machine period is the reference (`reference: true`). The others have a
`scale` map: the CPU time on that machine divided by the CPU time on the
reference for the same code, with the keys `default`, `<point>` and
`<flavor>.<point>`.
"""

from pathlib import Path

import pandas as pd
import yaml

from spotter.points import base_points


def load_machines(path: Path) -> list[dict]:
    """Read the machine periods, or return an empty list if the file does not exist."""
    if not path.exists():
        return []
    with open(path, "r") as f:
        return yaml.safe_load(f)["machines"]


def find_machine(machines: list[dict], host: str, d: pd.Timestamp) -> dict | None:
    """Return the configuration of a SPOT machine on a date, from config/machines.yaml."""
    for m in machines:
        start = pd.Timestamp(m.get("start", "1900-01-01"))
        end = pd.Timestamp(m.get("end", "2100-01-01"))
        if m["host"] == host and start <= d <= end:
            return m
    return None


def reference_machine(machines: list[dict]) -> dict | None:
    """Return the machine period that defines the scale of the CPU times."""
    return next((m for m in machines if m.get("reference")), None)


def has_scale(machines: list[dict]) -> bool:
    """Return True if a reference exists and at least one machine period has factors."""
    return reference_machine(machines) is not None and any(m.get("scale") for m in machines)


def scale_factor(machine: dict | None, flavor: str, point: str) -> float:
    """Return the factor of a point: `flavor.point`, else `point`, else `default`, else 1."""
    scale = (machine or {}).get("scale") or {}
    for key in (f"{flavor}.{point}", point, "default"):
        if key in scale:
            return float(scale[key])
    return 1.0


def normalize_cpu(
    df: pd.DataFrame, machines: list[dict], flavor: str, point: str
) -> pd.DataFrame:
    """Divide the `cpu` column by the factor of the machine of each row.

    A row with the host `unknown` takes the host of the previous row.
    """
    if "cpu" not in df or "host" not in df or not has_scale(machines):
        return df
    hosts = df["host"].mask(df["host"] == "unknown").ffill().bfill()
    dates = pd.to_datetime(df["date"]) if "date" in df else df.index
    factors = [
        scale_factor(find_machine(machines, h, d), flavor, point) for h, d in zip(hosts, dates)
    ]
    df = df.copy()
    df["cpu"] = df["cpu"] / factors
    return df


def normalize_point(
    df: pd.DataFrame,
    *,
    data_dir: Path,
    machines: list[dict],
    flavor: str,
    point: str,
    fetcher_config: dict | None,
) -> pd.DataFrame:
    """Normalize the `cpu` column of a point to the reference machine.

    A sum point is the sum of its normalized base points (from `data_dir`, a
    missing base point counts as zero), because the factor of a sum depends
    on the mix of its parts.
    """
    if "cpu" not in df or not has_scale(machines):
        return df
    bases = base_points(fetcher_config, flavor, point) if fetcher_config else [point]
    if bases == [point]:
        return normalize_cpu(df, machines, flavor, point)
    dates = pd.to_datetime(df["date"]) if "date" in df else df.index
    total = pd.Series(0.0, index=pd.DatetimeIndex(dates))
    for base in bases:
        path = data_dir / f"{flavor}_{base}.csv"
        if not path.exists():
            continue
        part = normalize_cpu(pd.read_csv(path, parse_dates=["date"]), machines, flavor, base)
        total += part.set_index("date")["cpu"].reindex(total.index).fillna(0.0)
    df = df.copy()
    df["cpu"] = total.to_numpy()
    return df
