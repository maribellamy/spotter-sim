from datetime import timedelta, date


def date_range(start: date, end: date, step=1) -> list[date]:
    result = [start]
    while result[-1] < end:
        result.append(result[-1] + timedelta(days=step))
    return result
