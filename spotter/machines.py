"""The SPOT machines, and the normalization of the CPU times to one reference machine.

`config/machines.yaml` has one entry per machine period (a host with an
optional `start` and `end`). One entry is the reference (`reference: true`).
The other entries can have a `scale` map: the CPU time on that machine divided
by the CPU time on the reference machine for the same code, per point. The
plotter and the checks divide the CPU times by these factors, so that a series
continues across a machine change.

    scale:
      default: 1.19                     # points without their own factor
      ckf: 1.30                         # one point of every flavor
      legacy_default.space_point_formation: 1.27   # one point of one flavor
"""

from pathlib import Path

import pandas as pd
import yaml


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
    """Return the CPU time on the machine divided by the CPU time on the reference machine.

    The factor of `flavor.point` wins over the factor of `point`, which wins
    over `default`. A machine without factors, or an unknown machine, has the
    factor 1.
    """
    scale = (machine or {}).get("scale") or {}
    for key in (f"{flavor}.{point}", point, "default"):
        if key in scale:
            return float(scale[key])
    return 1.0


def normalize_cpu(
    df: pd.DataFrame, machines: list[dict], flavor: str, point: str
) -> pd.DataFrame:
    """Divide the `cpu` column by the factor of the machine of each row.

    A row with the host `unknown` takes the host of the previous row, because
    the SPOT job stays on its machine. The other columns do not change.
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
