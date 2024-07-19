from datetime import date
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from urllib.error import HTTPError

def get_url(d: date,
            flavor: str) -> str:
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


def do_fetch(d: date,
             flavors: list[str]) -> tuple[date, list[None | str]]:
    assert isinstance(flavors, list)
    urls: list[str] = [get_url(d, flavor) for flavor in flavors]

    def load(u: str) -> None | str:
        try:
            return urllib.request.urlopen(u).read().decode("utf-8")
        except HTTPError as e:
            print(d, u, e)
            return None

    results: list[None | str] = list(map(load, urls))
    return d, results


def do_fetch_all(*,
                 dates: list[date],
                 flavors: list[str]) -> list[tuple[date, list[None | str]]]:
    assert isinstance(dates, list)
    assert isinstance(flavors, list)
    with ThreadPoolExecutor() as tp:
        return list(tp.map(do_fetch, dates, [flavors] * len(dates)))
