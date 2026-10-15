# HANDOFF

Status as of the end of the agent run. Numbers here are copied from `EVAL.md` and point to files in `results/`. Nothing was run on UF footage or on HiPerGator.

## What works (tested on public footage, CPU)
- `python -m vbcv all -c configs/public_gtu.yaml` runs all six stages end to end on a public 33 s end-line clip (detect 425.9 s on CPU, every other stage under 2 s; `docs/WORKLOG.md`). Stages checkpoint (`_done.json`), skip when nothing changed, retry once, and log failures.
- Court calibration from clicked points (`scripts/click_court.py` -> `configs/courts/*.yaml` -> `calibrate`), with fit / leave-one-out / held-out error and an overlay image. Optional radial distortion (`k1`) from traced lines.
- Ball: VballNet ONNX (MIT, temporal, 9 grayscale frames) in `detect`; tracker cleanup (jump gating, static false-positive removal, gap interpolation).
- Players: D-FINE detector (Apache-2.0 model card) + ByteTrack (supervision, MIT), feet projected to court meters, near/far side.
- Ball model trained here (`scripts/train_ball_heatmap.py`, small single-frame heatmap net, 35 min CPU): held-out F1 0.796 vs 0.744 for pretrained D-FINE (thr 0.2) on the same 167 frames. On GPU, train longer and at full resolution.
- Rally frame classifier on public data: held-out acc 0.822 / F1 0.833.
- Jump height from flight time and a ballistic 3D serve-speed fit, both checked on synthetic data only.
- Labeling queue generator, labeling guide, footage request.
- 10 unit tests (`python -m pytest -q tests`), ruff clean.

## What does not work / is not validated
- **Rally segmentation from ball visibility** is untested on real rallies (the public clip is one continuous drill). On UF film, start by combining the rally frame classifier (`scripts/rally_frame_classifier.py`) with whistle audio.
- **Touch labels** (serve/pass/set/attack/block) come from simple rules on ball direction changes + nearest player. Treat them as a labeling aid only.
- **Player identity** across a match (jersey numbers) is not implemented. Track IDs switch often (70 IDs in 33 s of public video).
- **Serve speed** needs a full camera matrix. It is recovered from the court homography assuming square pixels and a centered principal point; that assumption is unchecked. The old ground-plane proxy (`avg_ground_speed_kmh`) is off by more than +200% and should not be reported.
- **Jump height** assumes the feet bbox bottom is tracked every frame; with real bbox jitter the error will be larger than the synthetic numbers.
- **D-FINE tiled inference** from its model card is not implemented (full-frame only), which likely costs small-ball recall.
- **SLURM scripts** were written from the UF RC documentation but never submitted. `--gpus=1` with no partition gets an L4 per the docs; change it if your group uses another partition.
- `scripts/trace_court_lines.py` (automatic line tracing) failed on the public clip; manual clicking is the supported path.

## Rerun everything
```
pip install -r requirements.txt && pip install -e . && bash scripts/fetch_models.sh
python -m vbcv all -c configs/public_gtu.yaml                     # e2e
python scripts/eval_calibration.py --points configs/courts/public_gtu.yaml
python scripts/eval_ball_frames.py --split <ball_ds>/test           # D-FINE held-out m17+m18
python scripts/train_ball_heatmap.py --data <ball_ds> --steps 1500  # resumable, writes runs/train_ball_heatmap/
python scripts/rally_frame_classifier.py --data <rally_ds> --out runs/eval_rally_frames
python scripts/eval_kinematics.py
python scripts/make_labeling_queue.py -c configs/public_gtu.yaml --export-images
```
Public data: Hugging Face `GYdevy/volleyball-big-ds` (ball, MIT) and `GYdevy/volleyball_rally_no_rally` (Apache-2.0); see SURVEY.md. Ball dataset caveat: test matches 8 and 13 also appear in train; only matches 17 and 18 are held out.

## New UF match on HiPerGator
1. Copy video to `/blue/<group>/<user>/film/`. Make `configs/<match>.yaml` (copy `configs/public_gtu.yaml`, change `name`, `video`, `court.points_file`).
2. Click court points once per camera setup (`scripts/click_court.py`, use `--holdout`).
3. `STAGE=all CONFIG=configs/<match>.yaml sbatch slurm/stage.sbatch`.
4. Check `runs/<match>/calibrate/overlay.jpg` and `python scripts/render_overlay.py -c configs/<match>.yaml` before trusting any stat.

## Ranked next steps
1. Get film per FOOTAGE_REQUEST.md; pick 3 held-out matches up front.
2. Label rally boundaries for 2 matches; measure rally segmentation; add a whistle detector and the rally frame classifier as signals.
3. Click court points with holdout points; record the held-out error.
4. Label 300 ball frames per match; fine-tune VballNet (or the heatmap model here) on UF + public data with `slurm/train_ball.sbatch`; report held-out-match F1.
5. Label player boxes (100 frames) to measure detection/tracking; add tiled D-FINE inference; try RF-DETR.
6. Touches: label 30 full rallies/match, measure the rules, then train a temporal classifier on ball+player features.
7. Jersey-number recognition / identity assignment (the hardest step in the closest published study).
8. Validate serve speed with a radar gun and jump height with a jump-reach test from the match camera position.
9. Scouting aggregates (`stats`) only after 2-6 are measured.

## Open risks
- Camera position or zoom changing between matches breaks calibration reuse.
- Licenses: Ultralytics and BoxMOT are AGPL-3.0; they are optional only. Check every new model's weight license (SURVEY.md).
- The repo `eodorfman111/uf-volleyball-CV` is public; keep UF data, frames, and labels out of it.
- Leo's beach volleyball tool was never accessible to the agent, so nothing was reused from it directly.
