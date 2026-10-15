# Worklog (agent checkpoints)

- Env: Devin VM, 8 CPU, 31 GB RAM, no GPU, no HiPerGator creds. venv at ~/venv (torch cpu 2.14.1, ultralytics 8.4.177, onnxruntime 1.23.2, rtmlib 0.0.16, opencv 5.0.0.93).
- Reference clones (not vendored): ~/ref/fvti (asigatchov/fast-volleyball-tracking-inference, MIT), ~/ref/bvv (ddecks/beach-volley-vision, MIT), ~/ref/wasb (nttcom/WASB-SBDT, MIT).
- Public end-line test clip: ~/ref/fvti/examples/gtu_20250316_002.mp4 (1920x1080, 29.97 fps, 991 frames, MIT). Beach clip: beach_st_lenina_20250622_g1_005.mp4.
- VballNet ONNX on CPU over gtu clip: V4c 9.39 s, 880/991 visible; FastV1 8.37 s, 792/991 visible (no GT).
- Public end-line data (HF): GYdevy/volleyball-big-ds (MIT, YOLO ball, 41210/249/182, names matchN_{rally,non_rally}_frame), GYdevy/volleyball_rally_no_rally (Apache-2.0, 5767+5767 frames, 13 matches, 1280x720), GYdevy/volleyball-whistles (MIT, audio). Downloads -> ~/data/public/{ball_ds.zip,rally_ds.zip}.
- Ball ds split check (2026-10-15): train match-named imgs per match: m1 542, m2 166, m7 180, m8 238, m9 818, m10 250, m11 12, m13 80, m14 526, m15 178, m16 260 (+37960 non-match-named Roboflow imgs). valid = m4 66, m11 79, m13 14, m15 90. test = m8 1, m13 14, m17 84, m18 83. Clean held-out = test m17+m18 (167 imgs).
- GitHub: `gh repo create` fails ("Resource not accessible by integration"); asked Leo to create repo. Working in local git.
- gtu clip calibration: wide-angle view; court lines read by hand from a median background of 45 frames (players removed). Points in configs/courts/public_gtu.yaml. Left post bottom ~(535,673), right post ~(1291,691) (not used: post offset from sideline unknown, 0.5-1.0 m).
- Rally dataset: 11532 frames (5766 rally / 5766 non_rally), 13 matches (1,2,3,4,7,8,9,10,11,13,14,15,16), names matchN_frameidx. Frame classifier eval: scripts/rally_frame_classifier.py, test matches 4,13,15.
- Calibration (commit after 58ab29f, fit_homography = least squares when <10 pts): public_gtu 6 hand-read points fit RMSE 0.326 m, LOO RMSE 1.246 m (results/calibrate_public_gtu.json). Synthetic click-noise study (scripts/eval_calibration.py, 200 trials, seed 0): 6 pts @2px median err 0.074 m near half / 0.245 m far half (results/eval_calibration_synthetic.json).
