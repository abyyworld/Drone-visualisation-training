# Dataset audit — `/home/user/Drone-visualisation-training`

## Verdict

| Check | Result | Detail |
|---|---|---|
| No source-family shortcut | FAIL | families share a class: True; scene labels: ['healthy']; never co-occurring: ['corrosion + healthy', 'crack + healthy', 'healthy + surface_peeling'] |
| No cross-split source leakage | PASS | 0 sources span splits |
| No near-duplicate frame leakage | FAIL | 32.5% of held-out images match an adjacent train frame at IoU>0.5 |
| Box scales comparable across classes | FAIL | 9.2x spread between class median box areas |
| Test split thick enough to measure | FAIL | thin classes: {'corrosion': 16, 'surface_peeling': 29} |
| Label format consistent | FAIL | 4927 polygon files, 2484 box files |
| No duplicate images | FAIL | 5 duplicate files |

## Distribution

| Split | Images | Backgrounds | Boxes | Per class |
|---|---|---|---|---|
| train | 6581 | 98 | 8484 | corrosion 539, crack 1790, healthy 5284, surface_peeling 871 |
| valid | 627 | 8 | 831 | corrosion 62, crack 188, healthy 506, surface_peeling 75 |
| test | 312 | 3 | 409 | corrosion 16, crack 103, healthy 261, surface_peeling 29 |

## Source families

| Family | Files | Boxes | Purity | Classes |
|---|---|---|---|---|
| `<numeric>` | 2484 | 3673 | 57% | corrosion 617, crack 2081, surface_peeling 975 |
| `Areial_Healthy` | 1692 | 1881 | 100% | healthy 1881 |
| `Healthy_Train` | 3344 | 4170 | 100% | healthy 4170 |

## Split leakage

| Split | Images | Adjacent train frame | Confirmed IoU>0.5 |
|---|---|---|---|
| valid | 627 | 548 (87.4%) | 204 (32.5%) |
| test | 312 | 267 (85.6%) | 87 (27.9%) |

## Box geometry

| Class | n | Median area | <1% area | >50% area |
|---|---|---|---|---|
| corrosion | 617 | 0.0875 | 5.3% | 3.6% |
| crack | 2081 | 0.0670 | 9.3% | 1.5% |
| healthy | 6051 | 0.4418 | 0.4% | 42.7% |
| surface_peeling | 975 | 0.0481 | 21.6% | 0.5% |
