# LABELING_QUEUE

What to hand-label first, in order of value per hour. Definitions and formats: [docs/LABELING_GUIDE.md](docs/LABELING_GUIDE.md).

The queue is generated per match on HiPerGator from the pipeline's own predictions, so labeling effort goes where the models are unsure or where a label is needed for evaluation:

```
python -m vbcv all -c configs/<match>.yaml
python scripts/make_labeling_queue.py -c configs/<match>.yaml --export-images
# -> runs/<match>/labeling_queue/clips.csv, ball_frames.csv, images/
```

A demo queue built from the public clip (`configs/public_gtu.yaml`) has 59 clip rows (2 rally boundaries, 36 touches, 21 jumps) and 120 ball frames (40 rejected_by_tracker, 40 missed, 20 low_conf, 20 kept). Source: `scripts/make_labeling_queue.py` on `runs/public_gtu` (see docs/WORKLOG.md).

## Priority 0 (before any labeling): choose held-out matches
Pick 3 of the first 10 UF matches as test matches and write them in `configs/splits.yaml`. Never label-and-tune on them; they are only for the numbers in reports.

## Priority 1: rally boundaries (all rallies, every match)
- Why: rally/serve segmentation is output #1 and every later stage is scoped by rallies. Public data only gives frame-level rally/no-rally (frame classifier F1 0.833 on held-out public matches, results/eval_rally_frames.json); no rally boundary metric exists yet.
- Queue rows: `task=rally_boundary`. Label every rally in the match anyway; the predicted boundaries are only a starting point.
- If a DataVolley/Hudl export exists, align it first: it already has rally count and order.

## Priority 2: court points (once per camera setup)
- Why: all court positions, speeds and side assignment depend on it. Add holdout points so calibration error is measured, not assumed.

## Priority 3: touches in 30 full rallies per match
- Queue rows: `task=touch`, highest priority for predicted `serve/attack/block`.
- Why: the touch labeler is rule-based and unvalidated; 30 rallies x ~5 touches gives ~150 touches per match to measure it and later train a temporal classifier.

## Priority 4: ball points (300 frames per match)
- Queue rows: `ball_frames.csv`, weighted to frames the detector missed or the tracker rejected (hard cases).
- Why: fine-tuning VballNet / the heatmap model on UF gym lighting and backgrounds; the public held-out recall is 0.50-0.62 for the D-FINE baseline.

## Priority 5: jumps (50 per match)
- Queue rows: `task=jump`.
- Why: jump height from flight time is unvalidated; takeoff/landing frames measure the timing error directly. Add any known jump-reach measurements.

## Priority 6: player boxes + jersey IDs (100 frames)
- Why: needed to evaluate player detection/tracking (no end-line player ground truth exists publicly) and to attribute touches to players (the hardest step in the closest published study, see SURVEY.md).
