# Datasets: what exists, what I verified, and what does not exist

Written after actually going and looking, from an environment that can reach
GitHub but not Zenodo, Hugging Face, Kaggle or IEEE DataPort. Where I could
download and check something, it says so. Where I could not, it says that too —
the difference matters, because dataset descriptions are frequently optimistic.

## The short answer to "one dataset with both fire and people"

**There isn't one, publicly, in RGB.** I looked. No public aerial RGB dataset
annotates fire/smoke *and* people in the same frames. If you find one, it
supersedes most of this page.

The closest thing that exists is thermal, and that turns out to be the more
interesting finding.

## What I verified by downloading it

### HIT-UAV — aerial thermal, people ✅ verified

`github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset` · **CC BY 4.0**

| | |
|---|---|
| Images | 2,898 aerial thermal (from 43,470 frames) |
| Person boxes | **12,312** |
| Other classes | Car 7,311 · Bicycle 4,980 · OtherVehicle 148 |
| Altitude | 60–130 m |
| Conditions | day **and night**, camera 30–90° |
| Median person box | **0.067% of frame area** |
| Tiny (<0.1% of frame) | **73% of person boxes** |

That last row is why this dataset is worth more than its size suggests. Most
person datasets are dominated by large, close, upright people; a model trained
on those scores well and finds nobody from a drone. Here, three quarters of the
targets are in the regime that actually decides whether this class works.

No fire in it. Cloned in about a minute over plain git.

**Gotcha, already handled:** its COCO file keys the box list `annotation`
(singular) and filenames `filename`. Read strictly, you get zero boxes, every
image is filed as a genuine negative, and training proceeds cheerfully on 2,898
pictures of nothing. `prepare_datasets.py` now accepts both spellings and
*refuses* a file with no recognisable box list rather than returning empty.

### WIT-UAS — people over real fire ⚠️ found, could not download

`github.com/castacks/WIT-UAS-Dataset` · CMU AirLab ·
paper: `arxiv.org/pdf/2312.09159.pdf`

**This is the closest public dataset to your exact use case**: "A Wildland-fire
Infrared Thermal Dataset to Detect Crew Assets From Aerial Views", captured
during **prescribed burns**, bounding-box annotated. Classes, read from the repo:
`person, car, bicycle, othervehicle, dontcare`.

So it is people *at a real fire*, from the air — which is what you asked for.
It is thermal, and its download server (`airlab-share-02.andrew.cmu.edu:9000`,
via `pip install minio && python scripts/download_data.py`) is blocked from
here. **You can almost certainly get it.** Do that before anything else on this
page.

## The finding worth acting on: thermal is the right sensor for people

Look at a thermal frame from altitude and the reason is obvious. A person is a
bright signature against cool ground. In RGB at 100 m they are a few ambiguous
pixels the colour of a rock — and at night, nothing at all.

For the person class specifically, thermal is not a nice-to-have:

* **It works at night.** Half of a wildfire's dangerous hours.
* **It sees through smoke far better than RGB**, which is precisely the
  condition where a crew most needs to be located.
* **It does not care about camouflage or ground colour.** PPE colour, terrain
  and shadow all stop mattering.

This is a hardware decision, not a software one, and it should be made
deliberately: does the airframe carry a thermal payload? Many enterprise drones
do (dual RGB + radiometric), and if yours does, the honest architecture is
**two models on two streams** — fire/smoke from RGB, people from thermal —
rather than one model asked to do both from one sensor.

If the airframe is RGB-only, say so out loud when describing the person class,
because RGB person detection from altitude is materially weaker and the crew
should know which of the two things they are looking at.

## RGB person-from-air, if you must stay RGB

Not verified by download — the hosts are blocked here — so treat sizes and
terms as claims to check:

| Dataset | Where | Note |
|---|---|---|
| **HERIDAL** | search "HERIDAL database" (Univ. of Split) | wilderness search-and-rescue, RGB aerial. The closest RGB match to searching for someone in wildland. |
| **SARD** | search "SARD search and rescue dataset" | people in varied poses — lying, seated, obscured. Posture matters more than volume: a casualty is rarely standing. |
| **TinyPerson** | `github.com/ucas-vg/TinyBenchmark` | maritime/beach, wrong background, right object scale. |
| **VisDrone** | `github.com/VisDrone/VisDrone-Dataset` | drone imagery, urban and traffic-heavy. Research use only — check before any deployment. |

## Fire and smoke

| Dataset | Where | Note |
|---|---|---|
| **D-Fire** | `github.com/gaiasd/DFireDataset` ✅ reachable | ~21k, ground-level. Weighted 0.25: wrong viewpoint for a drone. |
| **FASDD** | Zenodo / Science Data Bank | 122k with ~52k hard negatives. The negatives are the valuable part. |
| **FLAME** | IEEE DataPort (free account) | genuine UAV pile burns. |
| **Boreal Forest Fire** | via the *Scientific Data* (2025) paper | genuine UAV, boxes + segmentation. |

## The gap you will still have

Even with all of the above, **no image contains a person next to a fire**. That
combination has to come from your own footage, and it is the case that matters
most: a crew working a flame front is exactly the frame where both classes
appear together and where scene priors learned from separate datasets fall
apart. `training/hard_negatives.md` has the mining workflow;
`station/incidentlog` already records every frame, which is the raw material.

Budget for labelling a few hundred frames of your own. It will do more for
real-world performance than another 50,000 public images.
