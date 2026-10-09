# archery-tag-stat-tracker

Automatically count per-player stats from Archery Tag league footage. The first stat is **shots taken per player**. Later stats are hits, times hit, and catches.

The input is the league's single fixed wide-angle camera: 3490×1400, 60 fps, mounted at one end of the neutral zone and looking down the field. Each team holds one half (left and right of the neutral zone).

## Status

_Last updated: 2026-10-08_

| Stage | State |
|---|---|
| Player detection + pose | ✅ Working. Field crop at high res, plus 2× zoomed tiles of the far end |
| Tracking | ⚠️ Works, but IDs fragment (one player becomes many track IDs) |
| Shot detection (pose) | ⚠️ 63% of real shots found, 44% of counted shots real (v3) |
| Shot detection (arrows) | ⚠️ 49% found, **68%** of counted shots real; 34% found / 86% real in a stricter setting (A4, see [Arrow detection](#arrow-detection-long-exposure)) |
| Shooter attribution from arrows | ⚠️ With lens correction: 64% of credited shots right (at 26% of all shots found, strict arrow setting); was 55% / 17% without |
| Lens correction + re-stitch | ✅ Two GoPro halves corrected and re-joined: far baseline straight, center pillar aligned. GoPro lens values fixed; each video gets its own stitch profile (see [Lens correction](#lens-correction-and-re-stitching-lenspy-)) |
| Who's who (player identity) | ⏳ Not started |
| Hits / catches | ⏳ Not started |

## Results

> **New: [lens-variant report](docs/lens_variants_report.md).** It compares five lens corrections (plus a resampling control) by the arrow flights they produce:
> - **T4** gives the cleanest gravity arcs (1.6‰ vs 2.3–3.1‰) and the longest flights.
> - Detection should still run on the original pixels, because resampling alone drops ~25% of arrows.


Ground truth: every shot in the first 60 s of the sample clip, hand-labeled with the [labeling page](#labeling-ground-truth). That's **35 shots** ([`data/labels_0-60s.json`](data/labels_0-60s.json)).

A detected shot counts as correct if it is within **±0.6 s** of a labeled release **and** the label click falls inside the shooter's box (grown by 30%). Each label can be matched to at most one detection.

| Version | Detector | Shot rule | Recall | Precision | Notes |
|---|---|---|---|---|---|
| v1 | field crop | "draw pose": one hand at the face, the other held out | 23% | 22% | Fails on near players seen from the front or back |
| v2 | field crop | "aim": both wrists at head height, then drop | 54% | 54% | Tuned on the same 35 labels, so optimistic |
| v3 | field crop + far-end tiles | "aim" | **63%** | 44% | Labeled shooters with no detection drop from 8 to 4. But people per frame go from 7.4 to 13.8: the far tiles also pick up spectators and refs behind the field, which adds false shots |

| A1 | arrows: frame-by-frame linking | | 14% | 9% | The shooter's body movement stole the arrow's blobs |
| A2 | arrows: three-blob seeding + player masking | | 26% | 26% | "Must start outside the zone" threw away real arrows |
| A3 | arrows: speed in body-heights per frame + zone-crossing test | | 69% | 21% | Fake flights from players hidden behind the far-right bunker |
| A4 | arrows: A3 + at least 5 points + merge duplicates | | 49% | **68%** | Best F1 so far (0.57); tuned on the same labels |

Arrow rows (A1–A4) are scored on time and direction only: a flight must start between 0.05 s before and 0.6 s after a labeled release, and fly away from the shooter's half. They don't yet say *who* shot.

- **Recall:** the share of real shots that were found.
- **Precision:** the share of counted shots that were real.

## How it works

```
video ──► crop field + far tiles ──► YOLO11-pose ──► NMS merge ──► ByteTrack ──► per-track keypoints (.npz)
                                                                                          │
                     hand labels ◄── labeling page                                       ▼
                          │                                      shot rule (aim → release) ──► shots per track
                          └──────────────► evaluate / tune ◄─────────────────────────────┘
```

### 1. Detection and pose: [`track.py`](src/archery_tag_stat_tracker/track.py)

- Model: `yolo11s-pose` (17 COCO keypoints), run on Apple Silicon via MPS.
- **Field crop:** inference runs on `x 300–3300, y 0–1000` instead of the whole frame. Cutting out the ceiling and floor gives players more pixels at the same model input size (`imgsz=3008`).
- **Far-end tiles:** players at the far end are 70–100 px tall and were often missed mid-draw. The far strip (`y 0–500`) is also run as two overlapping tiles, each upscaled about 2×. In a spot check this recovered 6 of the 10 missed far shooters with confidence ≥ 0.29 (see below).
- The results are merged with NMS (IoU 0.5).

The test frame with the field crop at high resolution. Nearly every player is found:

![detection on cropped field](docs/img/detection_crop.jpg)

Labeled shooters the field-crop-only detector missed. They're all at the far end and mid-draw:

![missed far players](docs/img/far_misses.jpg)

### 2. Tracking

- [ByteTrack](https://github.com/ifzhang/ByteTrack), called directly on the merged detections.
- Lost tracks are kept for about 2 s, because players duck behind bunkers.
- Every second frame is processed (30 fps).
- **Known issue:** a 60 s clip yields over 100 track IDs for about 10 players. Fragments have to be re-joined into players before per-player counts mean anything.

### 3. Shot rule: [`shots.py`](src/archery_tag_stat_tracker/shots.py)

A shot is an **aim** that lasts at least 0.3 s and then ends. Aiming means both wrists are within 0.2 × box-height below the head. At release, the draw hand snaps away and the bow arm drops.

Why wrist height and not the textbook draw pose: near players are seen from the front or back. Their extended bow arm points almost straight at the lens, so in 2D it lands next to the draw hand. Only side-on (far) players show the arm extended. Wrist height works for both views.

Parameters were picked by grid search with [`tune.py`](src/archery_tag_stat_tracker/tune.py). With only 35 labels this overfits, so we need more labeled minutes before trusting the numbers.

### 4. Labeling ground truth: [`label.py`](src/archery_tag_stat_tracker/label.py)

A local web page for marking shots. Play the clip, pause at each release, and click the shooter. Labels autosave to `labels.json`. Keys:

- `Space` plays and pauses.
- `,` and `.` step one frame.
- `←` and `→` jump 1 s.
- `[` and `]` change speed.
- `Z` undoes the last label.

![labeling page](docs/img/labeler.png)

![labeling page stepping through two labeled releases](docs/img/labeler.gif)

### 5. Evaluation and tuning: [`evaluate.py`](src/archery_tag_stat_tracker/evaluate.py), [`tune.py`](src/archery_tag_stat_tracker/tune.py)

See [Results](#results) for the matching criteria.

### 6. Arrow detection (long exposure): [`arrows.py`](src/archery_tag_stat_tracker/arrows.py)

**Why:** pose-based shot detection breaks when a shooter is crouched, behind a bunker, or seen from the front. But every shot has to fly across the **neutral zone**: plain turf, empty during play, and filmed by a fixed camera. So instead of watching the shooter, we watch the middle.

**The long-exposure idea:** stack a short window (about 0.8 s) of frames and keep, for each pixel, the largest change from the background. A flying arrow leaves a streak across the zone. At 60 fps the arrow moves 20–160 px per frame, so the streak is **dotted**, one blob per frame. Coloring each pixel by the frame it first changed in shows the direction of travel.

Each example below has three panels:
1. The frame at release, with the labeled shooter circled in red.
2. The long exposure.
3. The time-colored first-change map.

Cyan arrows are the flights the detector found. White lines are the neutral-zone edges.

**Mid-field, right → left (5.36 s).** The arrow crosses the whole zone in about 5 frames. A second flight (left → right) is the return shot labeled at 5.66 s.

![long exposure 5.36s](docs/img/le_536.jpg)

**Near-left shooter, low and fast (32.25 s).** About 140 px per frame. The flight is only picked up right of the stitch seam (the brightness change mid-image). The part near the shooter is hidden by their own body.

![long exposure 32.25s](docs/img/le_3225.jpg)

**Far end (13.97 s).** Far arrows are small and slow on screen (20–55 px per frame), and the center pillar splits their flights. Speed is therefore measured in player-heights per frame, and flights count if their straight-line extension crosses the zone.

![long exposure 13.97s](docs/img/le_1397.jpg)

**Failure case: false flight (25.6 s).** Players crouched behind the far-right bunker aren't detected as people, so their moving limbs form straight, fast blob chains. This is the main remaining source of false shots.

![false positive 25.6s](docs/img/le_false_2560.jpg)

**Pipeline:**
1. **Find fast-moving blobs.** In a band around the neutral zone, compare each frame with the frames 2 before and 2 after, and keep the pixels that differ from both. This leaves only fast movers.
2. **Mask out players.** Ignore blobs inside tracked player boxes.
3. **Start a flight only from a convincing triplet.** That means three blobs in consecutive frames that are evenly spaced, in a straight line, and fast (15–260 px per frame). Random clutter almost never forms one. The flight is then extended both ways, allowing 1 missed frame.
4. **Decide whether a flight is a shot.** It must have at least 5 points, move at least 0.25 player-heights per frame, fit a straight line (within 8 px or 15% of its speed), and have a straight-line path that crosses the neutral zone.
5. **Merge duplicates.** Merge flights going the same way within 0.15 s of each other. One arrow can split into several flights at the seam, at the pillar, or because of blur.

Runs at about 3× real time on an M1 Pro (18 s for 60 s of video).

**Experiment log (first minute, 35 labels):**

| Step | Change | Recall | Precision | What we learned |
|---|---|---|---|---|
| A1 | Frame-by-frame nearest-blob linking | 14% | 9% | Arrows move up to 160 px per frame, and the shooter's moving body steals their blobs |
| A2 | Start flights from 3-blob triplets; mask players; must start outside the zone and enter it | 26% | 26% | Clean arrows found. But arrows often *first appear* inside the zone (hidden by the shooter's body until then), so "starts outside" rejected them |
| A3 | Speed in player-heights per frame; straight-line extension must cross the zone | 69% | 21% | Far arrows recovered. False flights come from players hidden behind the far-right bunker (see the failure case) |
| A4 | At least 5 points; merge duplicates | 49% | 68% | Best balance. Setting the minimum to 6 points gives 34% / 86%: a high-precision signal |

**Crediting a shooter** ([`attribute.py`](src/archery_tag_stat_tracker/attribute.py)): fit a line through the flight and extend it back into the shooter's half. Then pick the tracked player whose shoulders are closest to that line, within 0.8 player-heights, in the 0.6 s before the arrow first appears. Scored strictly (right time **and** the label click inside that player's box):

| Arrow setting | Arrows found | Right player at the right time (raw frame), recall / precision | Same, **lens-corrected** |
|---|---|---|---|
| ≥ 4 points | 63% / 27% | 29% / 16% | 37% / 19% |
| ≥ 5 points | 49% / 68% | 20% / 35% | 29% / 42% |
| ≥ 6 points | 34% / 86% | 17% / 55% | **26% / 64%** |

In the raw frame, even real arrows were credited to the wrong player about 45% of the time: a straight line extended over hundreds of fisheye-bent pixels drifts off the shooter. Doing the geometry in [lens-corrected](#lens-correction-and-re-stitching-lenspy-) coordinates raised both numbers by about 9 points. The remaining misses are mostly shooters who aren't tracked at the release moment, or several players close to the line.

## Experiments

### Lens correction and re-stitching: [`lens.py`](src/archery_tag_stat_tracker/lens.py) ✅

**Why:** crediting an arrow to its shooter means extending the flight's line back into the shooter's half. The fisheye bends straight flights, so long extensions drift off the shooter. Corrected coordinates fixed a good part of that (see [Crediting a shooter](#6-arrow-detection-long-exposure-arrowspy)).

![before / after](docs/img/lens_before_after.jpg)

**The setup:** two GoPros, one per half, stitched side by side with a **hard cut** (the seam) through the neutral zone. In this video the seam is at x = 1822. The far pillar visibly jumps there because each camera sees it from a slightly different position.

![seam zoom](docs/img/seam_zoom.jpg)

**The final model** (variant "T4", chosen by eye), applied to each half:
1. **Radial lens correction**, k1 = **+0.2**, with each lens center placed **on the seam**, at the height of the far baseline (y = 190). Radial correction keeps lines through the lens center straight, so the seam column stays a straight vertical line in both halves. With the same settings on both sides, the halves still meet with **no black gap in the middle**.
2. **Zoom**, so the seam column fills the full height.
3. **Far-baseline alignment.** In the source, the far baseline is two straight segments forming a "^" with a ~12px step at the seam: slopes −0.14 on the left and +0.14 on the right. Steps:
   - Shear each half by 0.60 of its slope.
   - Stretch each half vertically about the seam's bottom so the segments meet.
   - Shear again so both segments lie on one line.

   Result: one level line, slope −0.001, with points 2.4px RMS off it.
4. **Pillar seam warp.** The two cameras see the far center pillar with parallax: the pad top and base were 7px apart across the seam, the right edge leaned, and the logo showed twice. A local warp of the right half near the seam fixes this:
   - a vertical stretch that matches the pad top and base;
   - a sideways shift of −23px at the top to −9px at the base, which removes the duplicated strip and makes the edges vertical.

   It fades out over 400px into the right half and below y≈300–600, so the rest of the frame is untouched.

![pillar: original / lens+baseline / final](docs/img/lens_pillar.jpg)

The corrected 2-minute clip is [`data/sample_2min_corrected.mp4`](data/sample_2min_corrected.mp4). It renders in about real time. The outer corners stay black. That's expected with radial correction, and nothing is lost there. `Lens.points()` corrects coordinates (arrow blobs, keypoints, boxes) without warping pixels. `Lens.frame()` warps whole frames.

**How we got there.** Automatic fitting failed, so I rendered candidates and you picked between them:

| Round | Candidates | Outcome |
|---|---|---|
| Auto-fit | Fisheye model fitted to snapped "straight" lines (4 variants) | ❌ Degenerate fits: they made lines look straight by distorting the image wrongly. Partly because the stitcher had already over-dewarped each half, so the needed correction is the *opposite* sign from a raw GoPro fisheye |
| 1 | Radial k1 −0.45 … +0.2; fisheye f 700–1400 | k1 = **+0.2** "very close" |
| 2–3 | Rulers on the lines; lens centers on the seam to close the middle gap | Middle gap closed, but the far baseline was still kinked |
| 4–6 | Shear vs rotate each half to level the baseline | Rotation opens a wedge. Full shear looked wrong, because the automatic baseline points had snapped to bunker edges and turf texture |
| 7 | Partial shears | 0.60 shear (N1) best |
| 8–9 | **Hand-picked** baseline points from zoomed full-res crops; align plus make collinear | Q3/Q4 "perfect" |
| 10–11 | Local pillar warp strengths | T4 "perfect" |

![baseline variants (zoomed far end)](docs/img/lens_baseline_options.jpg)

Lessons:
- The stitched halves aren't raw GoPro frames.
- Automatic line snapping is unreliable on this footage; hand-pick reference points from zoomed crops.
- Leveling a line isn't the same as straightening it. Each camera is turned away from the center, so some slope is real perspective.

**Equipment vs per-video settings.** `EQUIPMENT` in `lens.py` holds the GoPro lens values (k1 and focal lengths), which stay fixed while the cameras are the same. Everything tied to how a particular video was stitched lives in a **per-video profile**, [`data/stitch/<video>.json`](data/stitch/va-S1pJyK5U.json): the seam, baseline points, shear and pillar warp.
- `lens seam <video>` detects the seam automatically (it found 1823 here, against 1822 by hand).
- `lens build <profile> --video <video>` refuses to build if the seam doesn't match the profile.

**Redoing this for a new video:** camera angles and the cut can change between nights, so expect to redo the alignment. The lens values should carry over.
1. Detect the seam with `lens seam`.
2. Make a people-free background frame (the median of about 60 frames).
3. Hand-pick far-baseline points on both sides from zoomed crops.
4. Render shear and pillar variants, and pick by eye.
5. Save a new profile, then run `lens build`.

## Usage

```bash
uv sync
mkdir -p models && curl -sL -o models/yolo11s-pose.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt

# 1. detect + track (≈10 min per video-minute on an M1 Pro)
uv run python -m archery_tag_stat_tracker.track data/sample_2min.mp4 --duration 60 --out out/tracks.npz

# 2. shots + overlay video
uv run python -m archery_tag_stat_tracker.shots out/tracks.npz --video data/sample_2min.mp4 --out out/overlay.mp4

# 3. score against labels / grid-search the rule
uv run python -m archery_tag_stat_tracker.evaluate out/tracks.npz data/labels_0-60s.json -v
uv run python -m archery_tag_stat_tracker.tune out/tracks.npz data/labels_0-60s.json

# arrows in flight (uses tracks to mask out players) + long-exposure images
uv run python -m archery_tag_stat_tracker.arrows data/sample_2min.mp4 --duration 60 --tracks out/tracks.npz --out out/arrows.json
uv run python -m archery_tag_stat_tracker.longexposure data/sample_2min.mp4 5.36 --crop 1100,150,2800,650 \
  --arrows out/arrows.json --labels data/labels_0-60s.json --out out/le_536.jpg

# label more footage (expects clip.mp4 in the folder; 1746×700 is fine)
uv run python -m archery_tag_stat_tracker.label out/label   # → http://127.0.0.1:8765/
```

- `data/sample_2min.mp4`: the first 2 minutes of the full-resolution source ([YouTube](https://www.youtube.com/watch?v=va-S1pJyK5U)). Label coordinates are in source pixels (3490×1400).
- `data/sample_2min_corrected.mp4`: the same 2 minutes, lens-corrected and re-stitched (3490×1400, 60 fps). Made with:

```bash
uv run python -m archery_tag_stat_tracker.lens build data/stitch/va-S1pJyK5U.json --video data/sample_2min.mp4
uv run python -m archery_tag_stat_tracker.lens video data/sample_2min.mp4 --out data/sample_2min_corrected.mp4 --crf 27
```

## Roadmap

1. **Next:** credit each arrow flight to a shooter: trace it back, then choose among nearby players using timing and pose. Score *who shot* against the labels.
2. Fix false arrow flights from players hidden behind bunkers (for example, ignore blob chains that start and end inside a player's half without reaching the zone).
3. Mask out the area off the field. The far tiles roughly doubled the number of people detected, many of them spectators and refs.
4. Label 2–3 more minutes from other games, so tuning and testing use different data.
5. Re-join track fragments into players using appearance and side, plus a quick naming step.
6. If the wrist-height rule plateaus, train a small classifier on keypoint sequences (or short player-crop clips), using the labels as training data.
7. Hits, times hit, and catches (the arrow flights are the starting point).
