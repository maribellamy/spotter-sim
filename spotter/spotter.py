
def date_range(start, end, step=1):
    from datetime import timedelta
    result = [start]
    while result[-1] < end:
        result.append(result[-1] + timedelta(days=step))
    return result

def main():
    from datetime import date
    
    from spotter.parser import extract_all_points
    from spotter.fetcher import do_fetch_all
    from spotter.fitter import piecewise_const
    
    import yaml

    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.dates import date2num
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    start_date = date.fromisoformat(config["dates"]["start"])
    end_date = date.fromisoformat(config["dates"]["end"])
    date_list = date_range(start_date, end_date)
    flavors = list(config["flavors"].keys())
    flavor_tags = [config["flavors"][flavor]["tag"] for flavor in flavors]
    
    acts_default_edges = list(
        map(
            lambda x: tuple(
                pd.to_datetime(date.fromisoformat(xi))
                for xi in [x[k] for k in ["start", "end"]]
            ),
            config["flavors"]["acts_default"]["piecewise_const"],
        )
    )

    fetch_results = do_fetch_all(dates=date_list, flavors=flavor_tags)
    data_per_flavor = extract_all_points(fetch_results, flavors)

    legacy_default_track_finding_data = data_per_flavor["legacy_default"]["track_finding"]
    legacy_fast_track_finding_data = data_per_flavor["legacy_fast"]["track_finding"]
    acts_default_track_finding_data = data_per_flavor["acts_default"]["track_finding"]
    acts_fast_track_finding_data = data_per_flavor["acts_fast"]["track_finding"]

    acts_default_track_finding_data_fit = pd.DataFrame(
        {
            "date": piecewise_const(
                acts_default_track_finding_data["date"],
                acts_default_track_finding_data["cpu"],
                acts_default_edges,
            )[0],
            "cpu": piecewise_const(
                acts_default_track_finding_data["date"],
                acts_default_track_finding_data["cpu"],
                acts_default_edges,
            )[1],
        }
    )
    
    # plot cpu

    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = fig.subplots(1, 1)

    ax.plot(
        legacy_default_track_finding_data["date"],
        legacy_default_track_finding_data["cpu"],
        ".--",
        color="C1",
        label="Athena Default Track Finding",
    )
    ax.plot(
        legacy_fast_track_finding_data["date"],
        legacy_fast_track_finding_data["cpu"],
        ".--",
        color="C2",
        label="Athena Fast Track Finding",
    )
    ax.plot(
        acts_default_track_finding_data["date"],
        acts_default_track_finding_data["cpu"],
        ".",
        color="C3",
        alpha=0.5,
    )
    ax.plot(
        acts_default_track_finding_data_fit["date"],
        acts_default_track_finding_data_fit["cpu"],
        ".--",
        color="C3",
        label="Acts-based Default Track Finding",
    )
    ax.plot(
        acts_fast_track_finding_data["date"],
        acts_fast_track_finding_data["cpu"],
        ".--",
        color="C4",
        label="Acts-based Fast Track Finding",
    )
    
    # ax.plot(acts_seeding_data["date"], acts_seeding_data["cpu"], ".--", label="Acts-based Seeding")
    # ax.plot(acts_ckf_data["date"], acts_ckf_data["cpu"], ".--", label="Acts-based CKF")

    # ax.set_title("Track Finding CPU")
    ax.set_ylabel("CPU time per event [s]")
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")
    
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.03)
    
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.1)
    
    ax.legend()
    
    fig.savefig("cpu.png")
    fig.savefig("cpu.pdf")
    
    for annotation in config["flavors"]["acts_default"]["annotations"]:
        d, s, ha, txy = [annotation[k] for k in ["date", "text", "alignment", "offset"]]

        d = date.fromisoformat(d)
        y = acts_default_track_finding_data.loc[d:d].cpu.iloc[0]
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
    
    ax.plot(
        legacy_default_track_finding_data["date"],
        legacy_default_track_finding_data["malloc"],
        ".--",
        color="C1",
        label="Athena Default Track Finding",
    )
    ax.plot(
        legacy_fast_track_finding_data["date"],
        legacy_fast_track_finding_data["malloc"],
        ".--",
        color="C2",
        label="Athena Fast Track Finding",
    )
    ax.plot(
        acts_default_track_finding_data["date"],
        acts_default_track_finding_data["malloc"],
        ".--",
        color="C3",
        label="Acts-based Default Track Finding",
    )
    ax.plot(
        acts_fast_track_finding_data["date"],
        acts_fast_track_finding_data["malloc"],
        ".--",
        color="C4",
        label="Acts-based Default Track Finding",
    )
    
    # ax.set_title("Track Finding Memory")
    ax.set_ylabel("Memory allocations [mb/event]")
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")
    
    ax.legend()
    
    fig.savefig("mem.png")
    fig.savefig("mem.pdf")
    
if __name__ == "__main__":
    main()
