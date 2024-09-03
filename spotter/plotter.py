from datetime import date
import argparse
from pathlib import Path
import re
import textwrap

import yaml

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import mplhep as hep

from spotter.fitter import piecewise_const
from spotter.utilities import date_range


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/plot-trackfinding.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="data",
        help="Directory to store the data",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default="plots",
        help="Directory to store the plots",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    start_date = date.fromisoformat(config["dates"]["start"])
    end_date = date.fromisoformat(config["dates"]["end"])
    date_list = date_range(start_date, end_date)
    flavors = list(config["flavors"].keys())
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

    data_per_algorithm_per_flavor = {flavor: {} for flavor in flavors}
    for flavor in flavors:
        for file in args.data_dir.glob(f"{flavor}_*.csv"):
            algorithm = re.search(f"{flavor}_(.*).csv", file.name).group(1)
            df = pd.read_csv(file, parse_dates=["date"])
            df["date"] = pd.to_datetime(df["date"])
            df.index = df["date"]
            df = df.loc[start_date:end_date]
            data_per_algorithm_per_flavor[flavor][algorithm] = df

    track_finding_data = {
        flavor: data[config["flavors"][flavor]["points"]]
        for flavor, data in data_per_algorithm_per_flavor.items()
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

    args.plots_dir.mkdir(parents=True, exist_ok=True)

    # plot cpu

    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = fig.subplots(1, 1)

    for i, (flavor, label) in enumerate(zip(flavors, labels)):
        color = f"C{i + 1}"

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

    fig.savefig(args.plots_dir / "cpu.png")
    fig.savefig(args.plots_dir / "cpu.pdf")

    for flavor, annotations in annotations_per_flavor.items():
        selected_cpu_values = []
        for annotation in annotations:
            d, s, ha, txy, select, label = [
                annotation[k]
                for k in ["date", "text", "alignment", "offset", "select", "label"]
            ]
            d = date.fromisoformat(d)

            if d not in date_list:
                continue
            if len(track_finding_data[flavor].loc[d:d]) == 0:
                continue
            y = track_finding_data[flavor].loc[d:d].cpu.iloc[0]

            if select:
                selected_cpu_values.append([y, label])

            t = f"{d:%Y-%m-%d}"
            ax.annotate(
                f"{t}\n{s}",
                xy=(date2num(d), y),
                xytext=txy,
                ha=ha,
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-|>", color="k"),
            )

        if selected_cpu_values:
            with plt.style.context(hep.style.ATLAS):
                compare_to_flavor, compare_to_date = config["flavors"][flavor][
                    "compare_to_other"
                ]
                compare_to_cpu = (
                    track_finding_data[compare_to_flavor]
                    .loc[compare_to_date:compare_to_date]
                    .cpu.iloc[0]
                )
                cpu_abs = np.array(
                    [t for t, _ in selected_cpu_values], dtype=np.float64
                )
                cpu_labels = [
                    f"{i}: {l}" for i, (_, l) in enumerate(selected_cpu_values, start=1)
                ]
                cpu_rel = cpu_abs / compare_to_cpu
                cpu_rel = np.insert(cpu_rel, 0, cpu_rel[0])
                cpu_rel = np.insert(cpu_rel, -1, cpu_rel[-1])

                compare_fig, compare_ax = plt.subplots(figsize=(10, 5))

                # this gets rid of the vertical lines. we clip in y later anyway
                xrange = np.arange(len(cpu_rel))
                xrange[0] -= 5
                xrange[-1] += 5
                compare_ax.fill_between(
                    xrange,
                    cpu_rel,
                    fc="#6699ff",  # kAzure - 4
                    ec="#003399",  # kAzure - 6
                    lw=3,
                )

                compare_ax.axhline(1, color="#003399", lw=3, ls="--")

                compare_ax.set_xlim(0, len(cpu_rel) - 1)
                _, ymax = compare_ax.get_ylim()
                compare_ax.set_ylim(0.5, ymax + 0.2)

                compare_ax.set_ylabel("Relative CPU time per event")
                compare_ax.set_xlabel("Work Progress", labelpad=0)

                ticks = np.arange(len(cpu_rel) - 1)
                compare_ax.set_xticks(ticks)
                compare_ax.set_xticklabels([])
                compare_ax.set_xticks(np.arange(len(cpu_rel) - 1) + 0.5, minor=True)

                compare_ax.set_yticks([1, 2, 3, 4, 5, 6])

                compare_ax.set_xticklabels(
                    cpu_labels + [""],
                    rotation=-22,
                    ha="left",
                    va="top",
                    ma="center",
                    fontsize=10,
                    minor=True,
                )
                compare_ax.tick_params(
                    axis="x", which="minor", tick1On=False, tick2On=False
                )

                hep.atlas.text("Internal", fontsize=16, pad=0)

                text = rf"""
HL-LHC, $\sqrt{{s}}$ = 14 TeV, $t\bar{{t}}$, $\langle\mu\rangle$ = 200
{config["flavors"][flavor]["label"]} vs {config["flavors"][compare_to_flavor]["label"]} track seeding and finding CPU/Event
Start date: 15 February 2024, Athena 24.0.24 (Acts v32.0.2)
End date: 30 July 2024, Athena 25.0.11 (Acts v36.0.0)
                """
                text = textwrap.dedent(text.strip())

                compare_ax.text(
                    0.97,
                    0.96,
                    s=text,
                    transform=compare_ax.transAxes,
                    ha="right",
                    va="top",
                    ma="left",
                )

                compare_fig.tight_layout()

                compare_fig.savefig(args.plots_dir / f"cpu_{flavor}_rel.png")
                compare_fig.savefig(args.plots_dir / f"cpu_{flavor}_rel.pdf")

    fig.savefig(args.plots_dir / "cpu_annotated.png")
    fig.savefig(args.plots_dir / "cpu_annotated.pdf")

    # plot memory allocation

    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = fig.subplots(1, 1)

    for i, (flavor, label) in enumerate(zip(flavors, labels)):
        color = f"C{i + 1}"

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

    fig.savefig(args.plots_dir / "mem.png")
    fig.savefig(args.plots_dir / "mem.pdf")


if __name__ == "__main__":
    main()
