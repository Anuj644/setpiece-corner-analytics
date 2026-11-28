# Decisions

## G4 RQ3 snapshot review

The review sample contains 39 images, with three representative snapshots from each
of the three clusters and 30 randomly sampled eligible snapshots.

Anuj marked camera-outline alignment, orientation, and zone counts as correct for
all 39 images. The automated cross-check confirmed orientation, full visibility of
all four central zones, and agreement with the stored zone counts for all 39 images.

The automated check found all player markers inside the camera outline in 37 images.
It found 18 of 19 inside in `snapshot_023.png` and 20 of 21 inside in
`snapshot_024.png` (both round to 95%). Both snapshots still passed full zone
visibility and zone-count checks. The review available so far identified no
systematic issue and did not trigger a change to the visibility or counting rules.

G4 remains open pending the independent Claude review in `review_claude.csv`.

Evidence is recorded in [Anuj's review](reports/review/rq3/review_anuj.csv)
and the [automated check](reports/review/rq3/automated_check.csv).
