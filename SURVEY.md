# SURVEY — computer vision for indoor volleyball from a fixed end-line camera

Compiled 2026-10-15 on a Devin VM (no GPU, no HiPerGator access). Every version came from the PyPI JSON API on that date. Every license came from the GitHub API (`license.spdx_id`), a LICENSE file, or a Hugging Face card. The raw search logs are in `docs/arxiv.txt` and `docs/arxiv_abstracts.txt`. Upstream accuracy numbers are **quoted from the authors**. We did not reproduce them unless the "Ours" column says so.

## TL;DR — what I'd bet on

| Stage | Bet | Why | License |
|---|---|---|---|
| Ball | **Multi-frame heatmap net (TrackNet family: VballNet / WASB / TrackNetV3)**, plus a physics/Kalman trajectory filter | A volleyball seen from the end line is about 8–25 px across and blurs at contact. Every published comparison (WASB, TrackNetV3) shows that temporal heatmap models beat single-frame detectors on small, fast balls. VballNet already ships end-line indoor ONNX weights that run on CPU. | MIT (VballNet, WASB, TrackNetV3) |
| Players | Permissive detector (RF-DETR or D-FINE, Apache-2.0) **or** YOLO (AGPL) + ByteTrack/OC-SORT + court-polygon filter | Players are large and easy to detect. The hard part is keeping identities through occlusion at the net, so association (motion + court constraints) is where effort pays off. | Apache / MIT (YOLO = AGPL) |
| Court | **Manual 4–8-point homography, once per camera setup**. Later, optionally, a court-keypoint model. | The camera is fixed, so you click once per venue/session and get a sub-pixel calibration. Automatic court detection is only needed if the camera moves. | OpenCV Apache-2.0 |
| Pose / jump | RTMPose via `rtmlib` (no mmcv) for ankles/hips. Jump height from flight time (h = g·t²/8) **and** from pixel displacement scaled by the homography. | Flight-time jump height needs no metric calibration, only fps. *Gravity as a reference* (arXiv 1909.02211) reports 3.9 cm MAE on jumps from monocular video. | Apache-2.0 |
| Serve speed | Fit a ballistic arc to the ball track between toss contact and the first opponent touch / net crossing. Use the court plane plus net height as a 3D constraint. | This is the same idea as `fast-volleyball-tracking-inference/src/calc_speed.py` (physics least-squares fit). Plain pixel displacement through a ground-plane homography is **wrong** for an airborne ball (see below). | MIT |
| Rally / serve segmentation | Ball-trajectory state machine first. Add a frame classifier trained on the public end-line rally/no-rally set. Add whistle audio if UF film keeps audio. | The public data for exactly this task exists (Volleylitics rally/no-rally, whistles). | Apache-2.0 / MIT |
| Touch / event | Contact spotting = ball-direction-change candidates × nearest player, then classify the touch with **volleyball rules** (serve→pass→set→attack, ≤3 touches/side). Learned precise-event-spotting (E2E-Spot / T-DEED) once UF labels exist. | arXiv 2609.28049 (66 amateur matches, 46,648 labelled contacts) found a small trained temporal model beats VLM prompting at contact spotting, and "the sport's own rules recover rally outcomes the pixels cannot". | E2E-Spot BSD-3, T-DEED **GPL-3.0** |

**Biggest risk.** Identity (which UF player did it). Jersey numbers are rarely readable from the end line at 1080p. arXiv 2609.28049 finds identity is where every automated approach struggles. Plan for a human-in-the-loop roster assignment per rally.

---

## 1. Ball detection and tracking (small-object focus)

### Why it is hard here
From behind the end line the far-court ball is ~8–15 px at 1080p. It is motion-blurred at serves and attacks, often against bright ceiling lights. The Volleylitics end-line test labels I downloaded have boxes of 0.8–1.2 % of image width, i.e. ~10–15 px at 1280 wide (`ref/samples/ball/test/labels`). The D-FINE volleyball card reports **ceiling-light fixtures as the dominant false positive** on end-line footage (77 of 136 candidate boxes). Single-frame detectors cannot separate a light from a ball. Motion can.

### Options found

| Project | What | License | Maturity / evidence | Reusable? |
|---|---|---|---|---|
| **asigatchov/fast-volleyball-tracking-inference** (71★, pushed 2026-10) | ONNX/OpenVINO VballNet ball models (9-frame grayscale stacks, 288×512), rally extraction, court-aware touch detection, physics speed fit. **Example clip is an indoor end-line view.** | MIT | Authors report F1 0.902 (VballNetV4c) / 0.799 (FastV1) on *beach-test-raw* (3 clips, 886 frames). **We ran both models on CPU**: 991-frame 1080p clip in 9.4 s / 8.4 s (≈110 fps). Ball reported in 880/991 and 792/991 frames (no GT, so no accuracy number). | **Yes: baseline ball model, used directly.** |
| asigatchov/vball-net, vball-net-pytorch | Training code for VballNet (TF and PyTorch), grid-head variants. | MIT | Active (2026-10). Training data not published. | Training code reference |
| **nttcom/WASB-SBDT** (BMVC 2023) | HRNet-style heatmap baseline evaluated on 5 sports **including volleyball** (Volleyball dataset + new ball annotations). Weights on Google Drive. | MIT | Paper reports SOTA vs 6 methods on all 5 sports. Not maintained since 2023-11. | Strong second baseline; GPU recommended |
| qaz812345/TrackNetV3 | Heatmap + background estimate + inpainting rectification (shuttlecock). | MIT (LICENSE file; GitHub API shows NOASSERTION) | Authors: F1 98.56 % on shuttlecock test set. | Architecture reference |
| TrackNetV4/TrackNetV4 (arXiv 2409.14543) | Motion-attention maps fused into TrackNetV2/V3. | MIT | Tennis/shuttle only. | VballNet is built on it |
| ddecks/beach-volley-vision + HF `deadfast/beach-volley-vision-models` | Beach pipeline: fine-tuned TrackNetV3 (~18k visible-ball frames), rally FSM, RF-DETR players, homography, serve detector. | MIT | 1★, active. **Closest public analogue to Leo's beach tool.** | Pipeline design reference |
| HF `grdesignbuild/volleyball-person-ball-detector` | D-FINE-S (10 M params) person+ball, tiled inference, includes 121 end-line frames. | Apache-2.0 | Card: ball AP@[.5:.95] 0.547 on pseudo-labels; on 9 end-line contact frames, 4/4 real balls with 0 light hits at thr 0.5. | Permissive single-frame fallback + person detector |
| openvolley/ovml / ovmlpy | YOLOv4 "mvb" volleyball net (R / Python). | MIT | Experimental. | No |
| masouduut94/volleyball_analytics | YOLO-based game-state + action + ball detection, FastAPI. | **GPL-2.0** | 135★. | Ideas only (copyleft) |

### Public ball data

| Dataset | Content | License | Notes |
|---|---|---|---|
| **Volleylitics ball detection v8** (HF `GYdevy/volleyball-big-ds`, Roboflow export) | 41,210 train / 249 valid / 182 test images, 1 class, YOLO format, **indoor end-line camera**, 960×960 stretched, ×2 augmentation | MIT (Roboflow card) | The best public proxy for UF film. Frame names encode `matchN_{rally,non_rally}_frame`, so splits can be done by match. |
| Volleyball dataset (Ibrahim et al. CVPR'16) + ball annotations (Perez et al., GIRN) | 55 broadcast YouTube matches, 4,830 annotated clips × 41 frames; ball centres added | Code BSD-2; data = research use, cite | Broadcast side-elevated view, **not** end line. Google-Drive quotas. Used by WASB. |
| Beach Volley Vision `data/annotations/*_ball.csv` | Small beach GT | MIT | Beach only |

## 2. Player detection and tracking

| Project | License | Version (PyPI 2026-10-15) | Notes |
|---|---|---|---|
| ultralytics (YOLO11/YOLOv8 detect/pose/track) | **AGPL-3.0** | 8.4.177 | Fastest path. AGPL: fine for internal lab research. Distributing it or offering it as a network service obliges you to release source. **Ask the lab before any external product.** |
| roboflow/rf-detr | Apache-2.0 | rfdetr 1.11.2 | Permissive replacement for YOLO detection |
| Peterande/D-FINE, lyuwenyu/RT-DETR | Apache-2.0 | — | Permissive DETRs |
| ByteTrack (ifzhang) | MIT | — | Motion-only association. Also built into ultralytics and supervision. |
| OC-SORT (noahcao/OC_SORT) | MIT | — | Better for non-linear motion |
| BoT-SORT (NirAharon) | MIT | — | ReID + camera-motion compensation |
| roboflow/trackers | Apache-2.0 | — | Clean re-implementations |
| mikel-brostrom/boxmot | **AGPL-3.0** | 25.0.0 | Many trackers + ReID; AGPL |
| supervision | MIT | 0.30.9 | ByteTrack, annotators, zones |
| SportsMOT (MCG-NJU, arXiv 2304.05170) | No license on repo; dataset terms on request | — | 240 clips incl. volleyball (Olympics/NCAA, 720p, 25 fps). Broadcast view. MixSort tracker. Good for tracker eval, not the end-line view. |

**Bet.** Detector → feet point (bottom-centre of box) → homography → court-polygon filter → ByteTrack. Then a volleyball constraint: each team stays on its side of the net, and there are at most 6 per side.

## 3. Court calibration

| Option | License | Notes |
|---|---|---|
| OpenCV `findHomography` from clicked court corners/lines | Apache-2.0 | Fixed camera, so one calibration per session. Indoor court: 18 × 9 m, attack line 3 m from net, net 2.43 m (men) / 2.24 m (women). |
| asigatchov/Court-Keypoint-Detection | **No license** (all rights reserved by default) | YOLO11-pose, 8 keypoints, back-view indoor gyms. Do not copy code. The keypoint schema is a useful idea. |
| PnLCalib (arXiv 2404.08401) | **GPL-2.0** | Points+lines optimisation, soccer. |
| TVCalib, KaliCalib, *Universal protocol to benchmark camera calibration for sports* (2404.09807) | — | Metrics reference (reprojection error in metres, IoU of projected court). |
| roboflow/sports | MIT | Keypoint-based pitch homography examples |

## 4. Pose, jump height, serve/attack speed

| Item | License | Notes |
|---|---|---|
| rtmlib (Tau-J) | Apache-2.0, v0.0.16 | RTMPose/RTMO/DWPose via onnxruntime, no mmcv. CPU-friendly. |
| mmpose | Apache-2.0, v1.3.2 | Heavier. Last push 2025-08. |
| ViTPose | Apache-2.0 | Higher accuracy, GPU |
| YOLO-pose (ultralytics) | AGPL-3.0 | Easiest; licence caveat |
| *Gravity as a Reference…* (1909.02211) | paper | Pixel→metric from free-fall; 3.9 cm MAE on jumps |
| *Jump height with a single smartphone* (2302.10749) | paper | OpenPose CMJ, ICC 0.84–0.99 vs force plate |
| *AI-assisted jump detection… waist-worn IMU* (2505.05907) | paper | IMU ground truth. A route to **validation data** if UF S&C already uses VERT/IMUs. |

**Physics note (important for serve speed).** A ground-plane homography only maps points *on the floor*. Mapping the ball's pixel position through it gives nonsense. The usable approaches are: (a) fit a 3D parabola with known gravity, constrained by court geometry (contact point near the end line, net crossing above 2.43 m), as `calc_speed.py` does; or (b) flight time over a known distance (end line → net crossing = 9 m horizontal) as a robust lower bound. We implement (b), plus a 2D parabola-fit check.

**Jump height.** From the end line, vertical displacement is foreshortened, and depth changes the pixel scale. Flight time (take-off → landing frames from ankle keypoints) gives h = g·t²/8 with no metric scale. At 30 fps, ±1 frame on a 0.6 s flight is about ±7 % height error, so **60 fps footage roughly halves that** (see FOOTAGE_REQUEST.md).

## 5. Event / touch detection and rally segmentation

| Item | License | Notes |
|---|---|---|
| **Volleylitics rally/no-rally** (HF `GYdevy/volleyball_rally_no_rally`) | Apache-2.0 | 5,767 rally + 5,767 non-rally frames from 13 matches, **end-line indoor**, 1280×720 |
| **Volleylitics whistles** (HF `GYdevy/volleyball-whistles`) | MIT | 14+ full-match WAVs + whistle timestamps → audio rally boundaries |
| VREN (arXiv 2209.13846) | paper | NCAA D-I + pro rally notation language; tasks: set/hit type prediction |
| Volleyball dataset (Ibrahim) | research | 9 player actions (spike, set, block, dig, …), broadcast view |
| E2E-Spot (jhong93/spot) | BSD-3 | Frame-precise event spotting |
| T-DEED (arturxe2) | **GPL-3.0** | SOTA precise event spotting on FigureSkating/FineDiving |
| *Prompt, Probe, Train, or Annotate?* (arXiv 2609.28049, 2026-09) | paper | Volleyball, single fixed camera, 66 matches, 46,648 labelled contacts. Trained temporal models beat VLM prompting at contact spotting; rules recover outcomes; identity is the bottleneck. **Closest published study to this project.** |
| PathFinder / PathFinderPlus (arXiv 2309.14753) | paper | Set-tactic classification from a single camera |
| *Deep learning for sports video event detection* survey (2505.03991) | paper | TAL vs action spotting vs precise event spotting taxonomy |
| DataVolley / openvolley `.dvw` | MIT (datavolley R pkg) | **Target output schema.** Coaches already read it. |

**Note on frontier VLMs.** 2609.28049 shows prompting VLMs segments matches well. But sending UF film to a third-party API is **forbidden** by the project rules. Only a locally hosted open-weights model on HiPerGator would be allowed, and we did not evaluate one.

## 6. Annotation tools
CVAT (MIT) supports video and interpolated tracks, and is the best fit for ball/player tracks. Label Studio (Apache-2.0) is good for event timestamps. Both can be self-hosted on HiPerGator (Open OnDemand), so UF data stays on site.

## 7. Library versions (PyPI, 2026-10-15) and licenses
torch 2.14.1 (BSD-style, multiple) · torchvision 0.29.1 (BSD) · opencv-python 5.0.0.93 (Apache-2.0) · onnxruntime 1.31.0 (MIT; 1.23.2 installed on py3.10) · ultralytics 8.4.177 (AGPL-3.0) · supervision 0.30.9 (MIT) · boxmot 25.0.0 (AGPL-3.0) · rfdetr 1.11.2 (Apache-2.0) · mmpose 1.3.2 (Apache-2.0) · rtmlib 0.0.16 (Apache-2.0) · trackeval 1.3.0 (MIT) · scikit-learn 1.9.1 (BSD-3) · numpy 2.5.4 (BSD-3) · pandas 3.0.6 (BSD-3) · filterpy 1.4.5 (MIT) · lap 0.5.13 (BSD-2) · pyyaml 6.0.3 (MIT) · yt-dlp 2026.8.19. The versions actually installed in the test env are in `docs/pip_versions.txt`.

## 8. License policy adopted in this repo
- Core pipeline code depends only on permissive packages (MIT/BSD/Apache).
- AGPL `ultralytics` is an **optional** backend (`detect.backend: ultralytics`), isolated in one module, so the lab can drop it.
- No GPL/AGPL code is copied into the repo. MIT VballNet ONNX weights are downloaded at runtime, not vendored.
