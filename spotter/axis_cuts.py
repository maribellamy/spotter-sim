"""Cut quiet periods out of the x axis of a time series plot.

A quiet period is a date range in which no plotted series changes by more than
a relative threshold. The plotter replaces each cut with a narrow gap, so that
the plot shows the dates where something happens.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
from matplotlib.dates import date2num, num2date


def find_quiet_periods(
    series: list[pd.Series],
    *,
    threshold: float,
    min_days: int,
    window: int = 7,
    edge_days: int = 7,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Find the date ranges in which all series are about constant.

    `series` holds the values of each plotted series, indexed by date. The
    function smooths each series with a rolling median of `window` points,
    because single outliers are not a change. A range is quiet if the smoothed
    values of each series stay within `threshold` (relative to their minimum),
    and if no series starts or ends inside the range (with a tolerance of
    `edge_days`). Only ranges of at least `min_days` count.
    """
    smooth = [
        s.sort_index().rolling(window, center=True, min_periods=1).median()
        for s in series
        if len(s) > 0
    ]
    if len(smooth) == 0:
        return []
    table = pd.concat(smooth, axis=1).sort_index()
    dates = list(table.index)
    values = table.to_numpy()
    first = [s.index.min() for s in smooth]
    last = [s.index.max() for s in smooth]
    edge = timedelta(days=edge_days)

    def covers(k: int, a: pd.Timestamp, b: pd.Timestamp) -> bool:
        # a series that starts or ends inside the range is a change
        inside = a <= last[k] and first[k] <= b
        return not inside or (first[k] <= a + edge and last[k] >= b - edge)

    periods = []
    i = 0
    while i < len(dates):
        lo = np.full(len(smooth), np.inf)
        hi = np.full(len(smooth), -np.inf)
        end = i
        for j in range(i, len(dates)):
            row = values[j]
            present = ~np.isnan(row)
            new_lo = np.where(present, np.minimum(lo, row), lo)
            new_hi = np.where(present, np.maximum(hi, row), hi)
            valid = np.isfinite(new_lo) & (new_lo > 0)
            spread = (new_hi[valid] - new_lo[valid]) / new_lo[valid]
            if np.any(spread > threshold):
                break
            if not all(covers(k, dates[i], dates[j]) for k in range(len(smooth))):
                break
            lo, hi, end = new_lo, new_hi, j
        if (dates[end] - dates[i]).days >= min_days:
            periods.append((dates[i], dates[end]))
            i = end + 1
        else:
            i += 1
    return periods


def periods_to_cuts(
    periods: list[tuple[pd.Timestamp, pd.Timestamp]],
    *,
    margin_days: int,
    min_cut_days: int,
    keep: list[pd.Timestamp] = (),
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Convert quiet periods to axis cuts.

    Each cut leaves `margin_days` of the quiet period visible at both ends, so
    that the level before and after the cut stays visible. The dates in `keep`
    (for example annotations) stay visible with the same margin.
    """
    margin = timedelta(days=margin_days)
    cuts = []
    for a, b in periods:
        pieces = [(a + margin, b - margin)]
        for d in sorted(keep):
            split = []
            for x, y in pieces:
                if x < d + margin and d - margin < y:
                    split += [(x, d - margin), (d + margin, y)]
                else:
                    split.append((x, y))
            pieces = split
        cuts += [(x, y) for x, y in pieces if (y - x).days >= min_cut_days]
    return cuts


class CompressedAxis:
    """Map dates to x values in which each cut shrinks to a gap of `gap_days`."""

    def __init__(self, cuts: list[tuple[pd.Timestamp, pd.Timestamp]], gap_days: float):
        self.cuts = sorted((date2num(a), date2num(b)) for a, b in cuts)
        self.gap = gap_days

    def forward(self, dates) -> np.ndarray:
        x = np.atleast_1d(date2num(dates)).astype(float)
        result = x.copy()
        for a, b in self.cuts:
            inside = np.clip(x - a, 0, b - a)
            result -= inside - inside / (b - a) * self.gap
        return result

    def is_cut(self, dates) -> np.ndarray:
        x = np.atleast_1d(date2num(dates)).astype(float)
        result = np.zeros(len(x), dtype=bool)
        for a, b in self.cuts:
            result |= (x > a) & (x < b)
        return result

    def split_lines(self, x_dates: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Return x and y for a line plot, without points inside the cuts.

        The line has a break (NaN) where it crosses a cut, so that it does not
        draw a slope over the hidden dates.
        """
        keep = ~self.is_cut(x_dates)
        dates = pd.to_datetime(pd.Series(x_dates)[keep]).reset_index(drop=True)
        values = pd.Series(y)[keep].reset_index(drop=True).to_numpy(dtype=float)
        x = self.forward(dates)
        if len(x) == 0:
            return x, values
        raw = date2num(dates)
        breaks = [
            i
            for i in range(1, len(raw))
            if any(raw[i - 1] <= a and b <= raw[i] for a, b in self.cuts)
        ]
        x = np.insert(x, breaks, np.nan)
        values = np.insert(values, breaks, np.nan)
        return x, values

    def month_ticks(
        self, start, end, max_ticks: int = 14, min_distance: float = 40
    ) -> tuple[list[float], list[str]]:
        """Return tick positions and labels at the first day of visible months.

        A tick closer than `min_distance` (in compressed days) to the previous
        tick is dropped, so that the labels at a gap do not overlap.
        """
        months = pd.date_range(
            pd.Timestamp(start).to_period("M").to_timestamp(), end, freq="MS"
        )
        visible = [m for m in months if not self.is_cut([m])[0]]
        step = 1
        while len(visible) / step > max_ticks and step < 12:
            step = {1: 2, 2: 3, 3: 6, 6: 12}[step]
        visible = [m for m in visible if (m.month - 1) % step == 0]
        kept = []
        for m in visible:
            if not kept or self.forward([m])[0] - self.forward([kept[-1]])[0] >= min_distance:
                kept.append(m)
        visible = kept
        labels = []
        previous_year = None
        for m in visible:
            if m.year != previous_year:
                labels.append(f"{m:%b}\n{m:%Y}")
                previous_year = m.year
            else:
                labels.append(f"{m:%b}")
        return list(self.forward(visible)), labels

    def gaps(self) -> list[tuple[float, float, int]]:
        """Return the x range of each gap and the number of hidden days."""
        result = []
        for a, b in self.cuts:
            x0 = self.forward([num2date(a)])[0]
            result.append((x0, x0 + self.gap, int(round(b - a))))
        return result
