# EVAL: per-stage metrics, baselines, failure cases

Every number here comes from a JSON file in `results/` (copied from `runs/`), produced by the script named next to it. Nothing here was measured on UF footage. "Synthetic" means simulated data with a known answer, which is an upper bound on real accuracy.

| Stage | Metric | Data | Baseline number | Source |
|---|---|---|---|---|
| Court homography | fit RMSE / leave-one-out RMSE (m) | public end-line clip, 6 hand-read points | 0.326 m / 1.246 m | `results/calibrate_public_gtu.json` (`python -m vbcv calibrate -c configs/public_gtu.yaml`) |
| Court homography | median court error on a 0.5 m grid vs click noise | synthetic, same camera | 2 px noise: 0.074 m near half, 0.245 m far half; 4 px: 0.165 / 0.515 m | `results/eval_calibration_synthetic.json` (`scripts/eval_calibration.py`) |
| Ball (single frame) | P / R / F1, match if center within max(10 px, box) | public held-out matches 17+18 (167 frames, 167 balls) | D-FINE thr 0.2: P 0.936 R 0.617 F1 0.744; thr 0.5: P 0.977 R 0.497 F1 0.659 | `results/eval_ball_frames_dfine.json` (`scripts/eval_ball_frames.py`) |
| Ball (trained here) | same | same | see "Ball model trained here" below | `results/train_ball_heatmap.json` |
| Ball (temporal, VballNet) | coverage / runtime only | public clip, 991 frames, no labels | 880 frames with a ball, 9.39 s on CPU | `docs/WORKLOG.md` |
| Players | none (no public end-line player ground truth) | public clip | 11.6 detections/frame mean, 70 track IDs over 33 s (ID switches are frequent) | `runs/public_gtu/track/players.csv` |
| Rally (frame level) | accuracy / F1 | public rally/no-rally, held-out matches 4, 13, 15 (n=2328, majority baseline 0.5) | acc 0.822, F1 0.833 (raw); 5-frame smoothing hurts: acc 0.661 | `results/eval_rally_frames.json` (`scripts/rally_frame_classifier.py`) |
| Rally (segments) | none | no public end-line rally-boundary labels | not measured | - |
| Touch labels | none | no public end-line touch labels | not measured; rules only | - |
| Jump height | estimated vs true cm | synthetic foot tracks, noise-free | 30 fps: 40->39.4, 50->49.2, 80->78.5, 100->99.3 cm; jumps of 20 cm not detected | `results/eval_kinematics_synthetic.json` (`scripts/eval_kinematics.py`) |
| Serve speed | estimated vs true km/h | synthetic serves through the public-clip camera, 2 px noise | ground-plane proxy: +239% to +278% error (unusable); ballistic 3D fit: median within 1%, p90 abs error 1.4-8.0% | same |

## Failure cases (public data)
1. **Static ball false positive.** VballNet locked onto a person standing by the right wall for most of the public clip. The rally detector then saw "ball visible" all 33 s. Fix in `track.drop_static` (cells hit in more than 3% of frames are dropped): ball rows 949 -> 862, contacts 58 -> 36. See `runs/public_gtu/viz/overlay_0_400.mp4`, frames 21 and 350 (`python scripts/render_overlay.py -c configs/public_gtu.yaml --start 0 --end 400 --stills 21 350`).
2. **Drill clip is one "rally".** The public clip is a continuous drill, so the pipeline reports 1 rally, and its "serve" is a held ball (serve speed 21.7 km/h ballistic, 15.9 km/h proxy). This is not a rally-segmentation result; it is a clip with no real serves.
3. **Superwide lens.** The public clip is strongly distorted; hand-read points have 1.25 m leave-one-out error. Automatic line tracing (`scripts/trace_court_lines.py`) recovered only 1-2 points per line on this clip and is not used.
4. **Far-court foreshortening.** The far half of the court spans about 44 px vertically in the public clip (image y 684 -> 640 along the center line, from the fitted homography), so 4 px of click error is ~0.5 m there (synthetic study). Far-side player positions and far-side jumps will be the least accurate.
5. **Missed small balls.** D-FINE misses 38-50% of held-out balls; held-out ball boxes are 13.5 px (p10) to 19.2 px (median) at 960x960 (`results/ball_size_stats.txt`). Missed examples are in `runs/eval_ball_frames/predictions.json`.
6. **Off-court people.** Referees, coaches and bench players near the court are tracked as players; the court mask (1.5 m margin) keeps people standing right next to the sideline.

## What could not be validated without UF footage
- Any accuracy number on UF gyms, lighting, uniforms, or camera height.
- Rally boundaries, touch labels, player identity, jersey numbers: no public ground truth for this view.
- Real jump height and serve speed error: the synthetic tests assume a perfect camera model and noise-free player boxes. Real numbers need a radar gun / jump-reach measurement (see FOOTAGE_REQUEST.md).
- HiPerGator SLURM scripts: written from UF RC documentation, never submitted (no credentials on the dev VM).
