from datetime import date
import argparse

import yaml

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

from spotter.parser import extract_all_points
from spotter.fetcher import do_fetch_all
from spotter.fitter import piecewise_const
from spotter.utilities import date_range


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    start_date = date.fromisoformat(config["dates"]["start"])
    end_date = date.fromisoformat(config["dates"]["end"])
    date_list = date_range(start_date, end_date)
    flavors = {
        flavor: config["flavors"][flavor]["tag"] for flavor in config["flavors"].keys()
    }
    labels = [config["flavors"][flavor]["label"] for flavor in flavors]
    annotations_per_flavor = {
        flavor: config["flavors"][flavor]["annotations"] for flavor in flavors
    }

    piecewise_edges = {
        flavor: list(
            map(
                lambda x: tuple(
                    pd.to_datetime(date.fromisoformat(xi))
                    for xi in [x[k] for k in ["start", "end"]]
                ),
                config["flavors"][flavor]["piecewise_const"],
            )
        )
        for flavor in flavors
    }

    results_per_flavor = do_fetch_all(dates=date_list, flavors=flavors)
    data_per_flavor = {
        flavor: extract_all_points(results, flavor)
        for flavor, results in results_per_flavor.items()
    }

    track_finding_data = {
        flavor: data["track_finding"] for flavor, data in data_per_flavor.items()
    }

    track_finding_data_fit = {
        flavor: pd.DataFrame(
            {
                "date": piecewise_const(
                    track_finding_data[flavor]["date"],
                    track_finding_data[flavor]["cpu"],
                    piecewise_edges[flavor],
                )[0],
                "cpu": piecewise_const(
                    track_finding_data[flavor]["date"],
                    track_finding_data[flavor]["cpu"],
                    piecewise_edges[flavor],
                )[1],
            }
        )
        for flavor in flavors
    }

    # plot cpu

    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = fig.subplots(1, 1)

    for i, (flavor, label) in enumerate(zip(flavors, labels)):
        color = f"C{i+1}"

        ax.plot(
            track_finding_data[flavor]["date"],
            track_finding_data[flavor]["cpu"],
            ".",
            color=color,
            alpha=0.5,
        )
        ax.plot(
            track_finding_data_fit[flavor]["date"],
            track_finding_data_fit[flavor]["cpu"],
            ".--",
            color=color,
            label=label,
        )

    # ax.plot(acts_seeding_data["date"], acts_seeding_data["cpu"], ".--", label="Acts-based Seeding")
    # ax.plot(acts_ckf_data["date"], acts_ckf_data["cpu"], ".--", label="Acts-based CKF")

    # ax.set_title("Track Finding CPU")
    ax.set_ylabel("CPU time per event [s]")
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.03)

    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.1)

    ax.legend()

    fig.savefig("cpu.png")
    fig.savefig("cpu.pdf")

    for flavor, annotations in annotations_per_flavor.items():
        for annotation in annotations:
            d, s, ha, txy = [annotation[k] for k in ["date", "text", "alignment", "offset"]]
            d = date.fromisoformat(d)

            if d not in date_list:
                continue

            y = track_finding_data[flavor].loc[d:d].cpu.iloc[0]
            t = f"{d:%Y-%m-%d}"
            ax.annotate(
                f"{t}\n{s}",
                xy=(date2num(d), y),
                xytext=txy,
                ha=ha,
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-|>"),
            )

    fig.savefig("cpu_annotated.png")
    fig.savefig("cpu_annotated.pdf")

    # plot memory allocation

    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = fig.subplots(1, 1)

    for i, (flavor, label) in enumerate(zip(flavors, labels)):
        color = f"C{i+1}"

        ax.plot(
            track_finding_data[flavor]["date"],
            track_finding_data[flavor]["malloc"],
            ".--",
            color=color,
            label=label,
        )

    # ax.set_title("Track Finding Memory")
    ax.set_ylabel("Memory allocations [mb/event]")
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")

    ax.legend()

    fig.savefig("mem.png")
    fig.savefig("mem.pdf")


if __name__ == "__main__":
    main()
