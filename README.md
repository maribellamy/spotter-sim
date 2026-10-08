# SPOTTER

Generates plots from SPOT (https://atlaspmb.web.cern.ch/atlaspmb/) which help to keep track of the developments for reconstruction chains and their impact on CPU and memory consumption.

Originally from https://github.com/andiwand/cern-scripts/blob/858eb78de0548021836913b861e0aa2c50bf51e1/notebooks/2024-05-28_spot-graph.ipynb.

Installing the dependencies (use a virtual environment)
```bash
pip install -r requirements.txt
```

How to run it:
```bash
# run the fetching
python -m spotter.fetcher

# run the plotting
python -m spotter.plotter
```

Afterwards you should find a couple of PNG and PDF files in the current working directory.

The fetcher has two steps:

1. The download step reads the log index of each SPOT job and downloads the logs that are not in `data/raw/` yet. `data/raw/<tag>.csv` has one row for each component of each log, and `data/raw/<tag>_logs.csv` has one row for each downloaded log.
2. The extract step computes the points of each flavor from the raw tables and writes `data/<flavor>_<point>.csv`.

```bash
# only compute the points again, for example after a change to a mask
python -m spotter.fetcher --skip-download

# download all logs again, for example after a change to the step
python -m spotter.fetcher --refetch
```

`config/fetcher.yaml` explains the `map`, `sum`, `optional`, `mask` and `sources` options.

The plotter can cut quiet periods out of the x axis. A quiet period is a date range in which no plotted series changes by more than a relative threshold. The cuts never hide an annotation, a platform change or a move of a flavor between SPOT jobs.

`config/annotations.yaml` holds the annotations for all plots. Each annotation lists the flavors and the points that it affects, so a change appears on every plot that shows one of them. An annotation of a part also belongs to every sum that contains it (the `sum` points of `config/fetcher.yaml`): an annotation of `ckf` shows on the track finding plot and on the full chain plot too. The plots also mark a move of a flavor from one SPOT job to another, with a dotted line in the color of the flavor.

`config/machines.yaml` holds the CPU model and the HS23 (HEPScore23) benchmark of each SPOT machine. The plots name the machine, its CPU and its HS23 below the x axis and at each machine change.

One machine is the reference (`reference: true`). The other machine periods have `scale` factors: the CPU time on that machine divided by the CPU time on the reference machine for the same code, per point, and per flavor and point where the flavors differ. The plotter and the checks divide the CPU times by these factors, so that a series continues across a machine change. The plots show the CPU time in HS23·s: the normalized time in s times the HS23 per core of the reference machine, as in the public plots. The job runs one thread, so the unit is only an estimate of the throughput. `--unit s` gives the normalized seconds, and `--raw` the measured seconds.

HS23 does not give these factors. It measures the throughput of the whole machine, so the per-core value with SMT on is wrong by about 45% for a single thread, and the ratio between two machines differs between the algorithms by tens of percent (for example 1.15 for the clusterization and 1.29 for the CKF between aiatlasbm001 with SMT on and aiatlasbm006).

A plot config can set `log_y: true` for data that changes by more than a factor of about 10.

```bash
# use the axis cuts of the config (the default)
python -m spotter.plotter --config config/plot-trackfinding.yaml

# plot without axis cuts
python -m spotter.plotter --config config/plot-trackfinding.yaml --axis-cuts off

# compute the cuts from the data, and print them for the `axis_cuts` key of the config
python -m spotter.plotter --config config/plot-trackfinding.yaml --axis-cuts auto --suggest-axis-cuts --cut-threshold 0.05 --cut-min-days 60
```

## After a data update

Three tools help to check new data and to add annotations:

```bash
# list the steps in the points, grouped into events; show only the events
# without an annotation and without a platform or machine change
python -m spotter.checks steps --since 2026-09-01 --unannotated

# list the dates inside the lifetime of a point without a value
# (a gap is usually a renamed component; add the new name to the regex)
python -m spotter.checks gaps

# show the values around a date: points of a flavor, components of a SPOT
# job, or a validation table
python -m spotter.peek 2025-07-09 --flavor acts_fast --points ckf pixel_seeding reco_efficiency
python -m spotter.peek 2026-05-28 --tag actstracking --components "TrackFindingAlg"
python -m spotter.peek 2024-05-27 --tag actstracking --validation reco_efficiency

# after a machine change: measure the factors of the machine at its moves to and
# from the reference machine, and print the `scale` block for config/machines.yaml
python -m spotter.checks machine-scale

# write one standalone HTML page with all plot configs (about 16 MB)
python -m spotter.gallery --output gallery.html
```

The step date is the first nightly with the new value. Nightly D contains the merges into athena main up to about 21:00 Zurich time on day D.

Example:
![CPU annotated](plots/cpu_annotated.png)
