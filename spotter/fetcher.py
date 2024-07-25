from datetime import date
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from urllib.error import HTTPError
import itertools


def get_url(d: date, flavor: str) -> str:
    assert isinstance(d, date)
    assert isinstance(flavor, str)
    date_str = d.strftime("%d-%m-%Y")
    base = f"https://atlaspmb.web.cern.ch/atlaspmb/spot-mon-phase2_recoonly_{flavor}/pages/logs"
    binary = "main-x86_64-el9-gcc13-opt"
    job = f"phase2_recoonly_{flavor}-RAWtoALL"
    if d < date.fromisoformat("2024-03-01"):
        base += "/archive_centos7"
        binary = "main-x86_64-centos7-gcc11-opt"
    return f"{base}/{date_str}-{binary}-{job}"


def do_fetch(d: date, flavor: str) -> tuple[date, list[None | str]]:
    assert isinstance(flavor, str)
    url: str = get_url(d, flavor)

    try:
        return urllib.request.urlopen(url).read().decode("utf-8")
    except HTTPError as e:
        print(d, url, e)
        return None


def do_fetch_all(
    dates: list[date], flavors: dict[str, str]
) -> dict[str, list[tuple[date, str]]]:
    assert isinstance(dates, list)
    assert isinstance(flavors, dict)

    def fetch(d_flavor: tuple[date, tuple[str, str]]) -> tuple[date, str, None | str]:
        d, (flavor, flavor_tag) = d_flavor
        return d, flavor, do_fetch(d, flavor_tag)

    with ThreadPoolExecutor() as tp:
        results = list(tp.map(fetch, itertools.product(dates, flavors.items())))
    return {
        flavor: [
            (d, result)
            for d, f, result in results
            if f == flavor and result is not None
        ]
        for flavor in flavors
    }
