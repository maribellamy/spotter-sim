from datetime import date
import argparse
from pathlib import Path
import re

import yaml

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

from spotter.fitter import piecewise_const
from spotter.utilities import date_range
import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
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

    data_per_algorithm_per_flavor = {flavor: {} for flavor in flavors}
    for flavor in flavors:
        for file in args.data_dir.glob(f"{flavor}_*.csv"):
            algorithm = re.search(f"{flavor}_(.*).csv", file.name).group(1)
            df = pd.read_csv(file, parse_dates=["date"])
            df["date"] = pd.to_datetime(df["date"])
            df.index = df["date"]
            data_per_algorithm_per_flavor[flavor][algorithm] = df

    track_finding_data = {
        flavor: data["track_finding"]
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

    ROOT.gROOT.SetStyle("ATLAS")
    canvas = ROOT.TCanvas("canvas", "canvas", 1600, 800)
    canvas.SetBottomMargin(0.25)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.05)

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
                arrowprops=dict(arrowstyle="-|>"),
            )

        if selected_cpu_values:
            plot = ROOT.TH1D(
                f"plot_{flavor}",
                "",
                len(selected_cpu_values),
                -0.5,
                len(selected_cpu_values) - 0.5,
            )
            plot.SetStats(0)
            compare_to_flavor, compare_to_date = config["flavors"][flavor][
                "compare_to_other"
            ]
            compare_to_cpu = (
                track_finding_data[compare_to_flavor]
                .loc[compare_to_date:compare_to_date]
                .cpu.iloc[0]
            )

            graph = ROOT.TGraph(len(selected_cpu_values) + 4)

            for bin, (cpu, label) in enumerate(selected_cpu_values):
                plot.GetXaxis().SetBinLabel(plot.GetXaxis().FindBin(bin), label)

                graph.SetPoint(bin + 2, bin + 0.5, cpu / compare_to_cpu)

            graph.SetPoint(0, -0.5, 0)
            graph.SetPoint(1, -0.5, graph.GetPointY(2))
            graph.SetPoint(
                len(selected_cpu_values) + 2,
                len(selected_cpu_values) + 0.5,
                graph.GetPointY(len(selected_cpu_values) + 1),
            )
            graph.SetPoint(
                len(selected_cpu_values) + 3, len(selected_cpu_values) + 0.5, 0
            )
            plot.GetXaxis().SetTitle("Work Progress")
            plot.GetXaxis().SetTitleOffset(2.7)
            plot.GetXaxis().SetLabelOffset(0.022)
            plot.GetYaxis().SetTitleOffset(0.9)
            plot.GetYaxis().SetTitleSize(0.035)

            plot.GetYaxis().SetRangeUser(0.5, 6.5)
            plot.GetXaxis().SetRangeUser(0.0, len(selected_cpu_values) + 1)
            plot.GetYaxis().SetTitle(
                "#frac{"
                + config["flavors"][flavor]["label"]
                + " CPU time per event}{"
                + config["flavors"][compare_to_flavor]["label"]
                + " CPU time per event}"
            )

            graph.SetFillStyle(1001)
            # graph.SetFillColor(ROOT.kPink + 1)
            # graph.SetLineColor(ROOT.kPink + 5)
            graph.SetFillColor(ROOT.kAzure - 4)
            graph.SetLineColor(ROOT.kAzure - 6)
            graph.SetLineWidth(2)

            plot.Draw("")

            graph.Draw("LF")

            ROOT.gPad.RedrawAxis()

            atlas = ROOT.TLatex(0.50, 0.85, "ATLAS")
            atlas.SetNDC()
            atlas.SetTextFont(72)
            atlas.SetLineWidth(2)
            atlas.Draw()

            type_of_plot = ROOT.TLatex(0.60, 0.85, "Internal")
            type_of_plot.SetNDC()
            type_of_plot.SetTextFont(42)
            type_of_plot.SetLineWidth(2)
            type_of_plot.Draw()

            sample = ROOT.TLatex(
                0.50, 0.78, "HL-LHC, #sqrt{s} = 14 TeV, t#bar{t}, #LT#mu#GT = 200"
            )
            sample.SetNDC()
            sample.SetTextFont(42)
            sample.SetTextSize(0.04)
            sample.SetLineWidth(2)
            sample.Draw()

            text_on_plot = [
                "Start date: 15 February 2024, Athena 24.0.24 (Acts v32.0.2)",
                "End date: 30 July 2024, Athena 25.0.11 (Acts v36.0.0)",
            ]
            latex_text = []
            x, y = 0.50, 0.72
            y_offset = 0.06
            for line, text in enumerate(text_on_plot):
                ltext = ROOT.TLatex(x, y - line * y_offset, text)
                ltext.SetNDC()
                ltext.SetTextFont(42)
                ltext.SetTextSize(0.035)
                ltext.SetLineWidth(2)
                latex_text.append(ltext)
                ltext.Draw("same")

            line = ROOT.TLine(-0.5, 1, len(selected_cpu_values) + 0.5, 1.0)
            line.SetLineColor(ROOT.kBlack)
            line.SetLineStyle(3)
            line.SetLineWidth(2)
            line.Draw()

            canvas.SaveAs(str(args.plots_dir / "CPU_acts_athena_progress.png"))

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
