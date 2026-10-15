# Labeling guide (Leo / Alex)

All UF labeling happens on HiPerGator with a self-hosted tool (CVAT or Label Studio running in an Open OnDemand / SLURM session). Never use a cloud labeling service for UF film.

## 1. Build the queue for a match
```
python -m vbcv all -c configs/<match>.yaml
python scripts/make_labeling_queue.py -c configs/<match>.yaml --export-images
```
Outputs in `runs/<match>/labeling_queue/`:
- `clips.csv`: `video,task,priority,frame,why,start_frame,end_frame`. Sorted by priority. Each row is a short window (+-2 s) around a predicted rally boundary, touch or jump.
- `ball_frames.csv` + `images/`: single frames where the ball model disagreed with the tracker (`rejected_by_tracker`), found nothing (`missed`), was unsure (`low_conf`), or a few confident ones (`kept`) as a sanity check.

Label in this order. Stop a task once the target count is reached and move on.

| Order | Task | Target per match | Time estimate |
|---|---|---|---|
| 1 | Rally boundaries | every rally (~150-200/match) | ~1 h/match by scrubbing at 2x |
| 2 | Court points | 1 per camera setup | 5 min |
| 3 | Touches | 30 rallies fully labeled | ~1.5 h |
| 4 | Ball points | 300 frames | ~1 h |
| 5 | Jumps | 50 jumps | ~30 min |
| 6 | Player boxes + jersey IDs | 100 frames | ~2 h |

Inferred estimates, not measured; adjust after the first match.

## 2. Definitions

### Rally boundaries (CSV: `video,rally_id,start_frame,end_frame`)
- `start_frame`: frame where the server's hand contacts the ball on the serve (not the toss).
- `end_frame`: frame where the ball touches the floor, goes out of play (touches antenna, ceiling, wall, out-of-bounds floor) or the referee whistles for a fault, whichever is first.
- Re-serves after a let or a whistle before the serve: not a rally; skip.

### Court points (`configs/courts/<setup>.yaml`)
```
python scripts/click_court.py <video> configs/courts/<setup>.yaml --frame <idx> --holdout far_attack_left far_attack_right
```
Click the **outer edge of the line where two lines meet** (lines are part of the court). Use a frame with an empty court. Click every point the window prompts for; press `s` to skip a hidden point, `q` to finish. Points named in `--holdout` are saved under `holdout` and not used for the fit, so calibration error is measured on points the fit never saw. Run `python -c "from vbcv.court import court_points; print(list(court_points()))"` for valid point names.

### Touches (CSV: `video,rally_id,frame,label,team,jersey,quality`)
- `frame`: the first frame the ball is visibly in contact with the player (or the frame of closest approach if it is not visible).
- `label`: one of `serve, pass, set, attack, block, dig, freeball, other`.
  - `serve`: the serve contact.
  - `pass`: first contact of the receiving team after a serve (serve receive).
  - `dig`: first contact of a team after an attack (not a serve).
  - `set`: any contact intended to put the ball up for an attacker (usually 2nd).
  - `attack`: any contact that sends the ball over the net that is not a block (spike, tip, roll shot, overpass on 2).
  - `block`: contact by a player at the net reaching above the net to stop an opponent attack. Block touch is not counted as a team contact.
  - `freeball`: easy non-attack ball sent over (3rd contact underhand/overhead pass).
- `team`: `near` or `far` (relative to camera).
- `jersey`: number if readable, else blank.
- `quality`: optional 0-3 (0 = error, 3 = perfect), only if coaches' stat sheet is unavailable.
Label every touch in a rally; a rally is only useful if fully labeled.

### Ball points (CVAT points / CSV `image,x,y,visible`)
- Click the ball center. `visible=0` if out of frame or fully hidden; `visible=2` if partly hidden or motion-blurred (click the center of the streak).
- Label only the ball in play; ignore warm-up balls, balls in a cart, etc.

### Jumps (CSV: `video,frame_takeoff,frame_landing,track_id,jersey,type`)
- `frame_takeoff`: last frame both feet touch the floor. `frame_landing`: first frame a foot touches the floor.
- `type`: `attack, block, jump_serve, jump_set, other`.
- Flight time t gives height h = g t^2 / 8 (assumes same takeoff/landing posture).

### Player boxes
- Box from top of head to bottom of the lowest foot, including arms when extended above head. Give every on-court player a box; give a box to bench/referee only if they stand inside the court margin, with label `non_player`.
- Use the same `track_id` for the same player within a clip; add `jersey` once readable.

## 3. Quality checks
- Double-label 10% of rallies (both Leo and Alex) and compute agreement: rally boundary within 5 frames, touch label match. Target 90%+ before scaling up.
- Keep held-out matches separate: decide the 3 test matches **before** labeling and never tune thresholds on them.

## 4. Where labels go
`/blue/<group>/<user>/vbcv_labels/<match>/{rallies,touches,ball,jumps}.csv` on HiPerGator. Never commit labels or frames of UF athletes to git.
