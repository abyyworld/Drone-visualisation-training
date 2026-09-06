# Dataset audit — `/home/user/Drone-visualisation-training/datasets/turbine_v2`

## Verdict

| Check | Result | Detail |
|---|---|---|
| No source-family shortcut | PASS | families share a class: True; scene labels: none; never co-occurring: none |
| No cross-split source leakage | PASS | 0 sources span splits |
| No near-duplicate frame leakage | PASS | 0.0% of held-out images match an adjacent train frame at IoU>0.5 |
| Box scales comparable across classes | PASS | 1.9x spread between class median box areas |
| Test split thick enough to measure | PASS | thin classes: none |
| Label format consistent | PASS | 0 polygon files, 2043 box files |
| No duplicate images | PASS | 0 duplicate files |

## Distribution

| Split | Images | Backgrounds | Boxes | Per class |
|---|---|---|---|---|
| train | 2283 | 551 | 2580 | corrosion 418, crack 1479, surface_peeling 683 |
| valid | 198 | 43 | 249 | corrosion 44, crack 145, surface_peeling 60 |
| test | 197 | 41 | 204 | corrosion 34, crack 108, surface_peeling 62 |

## Source families

| Family | Files | Boxes | Purity | Classes |
|---|---|---|---|---|
| `<numeric>` | 2043 | 3033 | 57% | corrosion 496, crack 1732, surface_peeling 805 |

## Split leakage

| Split | Images | Adjacent train frame | Confirmed IoU>0.5 |
|---|---|---|---|
| valid | 198 | 1 (0.5%) | 0 (0.0%) |
| test | 197 | 0 (0.0%) | 0 (0.0%) |

## Box geometry

| Class | n | Median area | <1% area | >50% area |
|---|---|---|---|---|
| corrosion | 496 | 0.0874 | 5.2% | 3.8% |
| crack | 1732 | 0.0671 | 9.6% | 1.4% |
| surface_peeling | 805 | 0.0458 | 22.7% | 0.4% |
