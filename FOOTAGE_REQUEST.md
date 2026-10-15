# Footage request for UF coaches / video staff

Everything below stays on HiPerGator (`/blue/<group>/...`). Do not upload it anywhere else (no Google Drive, no cloud labeling tools, no LLM APIs).

## What to ask for

| Item | Ask | Minimum | Why |
|---|---|---|---|
| Camera position | Fixed, on a tripod, **centered behind one end line**, 3-6 m back, lens 4-8 m high (balcony or lift) | Behind end line, does not move during the match | Every stage assumes a fixed camera: one court calibration per match. Height reduces player occlusion of the far court and of the ball over the net. |
| Framing | All four court corners, both net posts and the antenna tops in frame, plus about 2 m above the net | All 4 corners + net | Calibration needs at least 4 known court points. Net/antenna tops give a height reference for jump and ball height. |
| Zoom / lens | No zoom or pan during the match. Avoid fisheye/"superwide" modes; a normal wide lens is fine | Constant zoom | Each zoom change invalidates calibration. The superwide public test clip had 0.33 m homography fit RMSE and 1.25 m leave-one-out error with hand-read points (results/calibrate_public_gtu.json). |
| Frame rate | **60 fps** | 30 fps | Serve speed: at 30 fps a 80 km/h serve moves ~0.75 m between frames and the ball is motion-blurred; contact timing for touches and jump take-off/landing is ~2x sharper at 60 fps. |
| Resolution | **1920x1080** (4K welcome if storage allows) | 1280x720 | In the public end-line dataset (960x960 images) the held-out ball box is 13.5 px at p10 and 19.2 px median (results/ball_size_stats.txt); D-FINE recall on those held-out frames is only 0.50-0.62 (results/eval_ball_frames_dfine.json). 720p shrinks the ball by a third and makes this worse. |
| Shutter | Fast shutter (1/500 s or faster) if the camera allows it | Auto | Motion blur turns the ball into a streak; detectors are trained on mostly sharp balls. |
| Audio | Keep the audio track | - | Referee whistles mark rally start/end; a whistle detector is a cheap, robust rally-boundary signal (public whistle dataset exists). |
| Format | Original camera files (MP4/MOV, H.264/H.265), no re-encode, no overlays/scoreboards burned in | Any MP4 | Re-encoding blurs small balls; overlays create false detections. |
| Matches | **10 full matches** to start (mix of home/away opponents, at least 2 different gyms if away film exists) | 4 matches | Hold out whole matches for evaluation (train on 7, test on 3). Fewer than 4 matches cannot give an honest held-out number. |
| Same camera spot | Mark the tripod spot on the floor and reuse it | - | Calibration can then be reused across matches, and models do not have to generalize across viewpoints. |
| Metadata per match | Date, opponent, gym, men/women (net height), roster with jersey numbers, and the existing stat sheet / DataVolley / Hudl export if one exists | Date + opponent + roster | Existing stat files are free labels: rally counts, serve order, touch types per rally can validate segmentation and event labels without hand-labeling. |

## What not to ask for
- Broadcast or panning/zooming camera film: the pipeline cannot use it without per-frame calibration.
- Clips cut to highlights only: rally segmentation needs dead time between rallies.

## One-off calibration capture (5 minutes, once per gym)
- With the camera in its match position and the court empty, record 10 seconds of video. This gives a clean frame for clicking court points (`scripts/click_court.py`).
- If possible, measure camera height and distance to the end line with a tape measure; write them in the match metadata.
- If possible, film one person standing still at the 4 court corners and at the net center, and one jump-reach at the net with a known reach (e.g., a jump-reach test). This gives ground truth for court-position and jump-height error.

## Serve-speed / jump ground truth (optional, high value)
- If the team has a radar gun (Pocket Radar etc.), record serve speeds for ~30 serves during a practice filmed from the same camera spot.
- If a Vert / jump-tracking wearable or force plate is used in practice, record a practice session with it. These are the only ways to measure our jump/serve error rather than guess it.
