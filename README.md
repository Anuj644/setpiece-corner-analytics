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

## RQ3 central-box positioning

Run `fetch`, `ingest`, `qa` and `rq1` first to build the pinned corner inputs. Then run:

```
python -m pytest tests/rq3
python -m pytest -m data tests/data/test_rq3_audit.py
python -m setpiece rq3
python -m setpiece review-export rq3
```

The RQ3 tests cover boundary-inclusive geometry, mirrored left and right corners, null
counts in obscured zones, profile filters, whole-match stability refits, match-bootstrap
team intervals, and the CLI exports. A run writes eight `rq3_*.parquet` tables to
`data/processed/derived/`. The tables include the team-season eligibility funnel, model
selection, assignments, centroid descriptions, same-k sensitivity comparisons, and
camera coverage by cluster. Cluster-profile and example-corner figures go to
`reports/figures/rq3/`.

With the default configuration and pinned core slices, 507 of 940 matched 360 snapshots
passed screening and showed all four central zones. The model selected three clusters
from `k = 2` through `k = 5` by ranking silhouette scores among fits that passed the
whole-match stability check.
Team cluster shares use each team's eligible snapshots as the denominator. Use the
eligibility funnel to see how many corners had a snapshot and how many passed screening.

The data test checks 569 World Cup corners, 300 frames, 286 frames with at least 16
players, and 140 frames showing all four areas. For the documented grid, planning
decision D-41 verifies 366 screened, 211 central-zone-visible, and 48
full-box-visible snapshots. See `reports/rq3/audit_reproduction.csv` and the RQ3
manifest for the counts and status.

The G4 export writes three representative images per cluster when available and a
seeded sample of up to 30 eligible snapshots under `reports/review/rq3/`. Reviewers
fill in a copy of `review.csv` to record camera alignment, orientation, and zone-count
checks. The manual review of the initial RQ3 run is pending.

## Attribution

Data: StatsBomb Open Data, https://github.com/statsbomb/open-data
