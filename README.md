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

Example:
![CPU annotated](plots/cpu_annotated.png)
