# setpiece-corner-analytics

Reproducible corner-kick analytics on StatsBomb Open Data. The pipeline fetches the open data at a pinned revision, ingests and quality-checks it, and builds the corner-kick analyses and report assets from a single command-line entry point, so every stage can be rerun from a clean checkout.

## Setup

Windows:

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

macOS / Linux:

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

## Run

```
python -m setpiece --help
```

## Test

```
python -m pytest
```

## Attribution

Data: StatsBomb Open Data, https://github.com/statsbomb/open-data
