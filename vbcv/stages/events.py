import numpy as np
import pandas as pd

from vbcv.court import load_homography
from vbcv.kinematics import camera_from_h, fit_ballistic

from ..checkpoint import stage_dir
from .ingest import load_meta

G = 9.81


def segment_rallies(ball, meta, c):
    fps = meta["fps"]
    n0, n1 = meta["start_frame"], meta["end_frame"]
    vis = np.zeros(n1 - n0)
    if len(ball):
        b = ball.sort_values("frame")
        idx = b.frame.to_numpy(int) - n0
        ok = (idx >= 0) & (idx < len(vis))
        min_speed = c.get("min_speed_px", 0)
        if min_speed and len(b) > 1:
            f, bx, by = b.frame.to_numpy(float), b.x.to_numpy(float), b.y.to_numpy(float)
            segs = b.seg.to_numpy() if "seg" in b else np.zeros(len(b), int)
            speed = np.zeros(len(b))
            for s in np.unique(segs):
                m = np.flatnonzero(segs == s)
                if len(m) > 1:
                    speed[m] = np.hypot(np.gradient(bx[m], f[m]), np.gradient(by[m], f[m]))
            ok &= speed >= min_speed
        vis[idx[ok]] = 1
    w = max(1, int(round(c["window_s"] * fps)))
    ratio = np.convolve(vis, np.ones(w) / w, mode="same")
    active = ratio >= c["min_ball_ratio"]
    spans = []
    i = 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            spans.append([i, j - 1])
            i = j
        else:
            i += 1
    merged = []
    for s in spans:
        if merged and (s[0] - merged[-1][1]) / fps <= c["merge_gap_s"]:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    pad = int(round(c["pad_s"] * fps))
    rows = []
    for a, b in merged:
        if (b - a) / fps < c["min_rally_s"]:
            continue
        a, b = max(0, a - pad) + n0, min(len(active) - 1, b + pad) + n0
        rows.append((len(rows), a, b, a / fps, b / fps, float(vis[a - n0 : b - n0 + 1].mean())))
    return pd.DataFrame(rows, columns=["rally_id", "start_frame", "end_frame", "start_s", "end_s", "ball_ratio"])


def smooth(v, k=5):
    if len(v) < k:
        return v
    return np.convolve(np.pad(v, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def find_contacts(ball, c):
    rows = []
    for seg, g in ball.groupby("seg"):
        g = g.sort_values("frame")
        if len(g) < 7:
            continue
        x, y, f = smooth(g.x.to_numpy()), smooth(g.y.to_numpy()), g.frame.to_numpy()
        vx, vy = np.gradient(x), np.gradient(y)
        sp = np.hypot(vx, vy)
        cand = []
        for i in range(3, len(g) - 3):
            a = np.array([x[i] - x[i - 3], y[i] - y[i - 3]])
            b = np.array([x[i + 3] - x[i], y[i + 3] - y[i]])
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na / 3 < c["min_speed_px"] and nb / 3 < c["min_speed_px"]:
                continue
            ang = np.degrees(np.arccos(np.clip(a @ b / max(na * nb, 1e-6), -1, 1)))
            ratio = max(na, nb) / max(min(na, nb), 1e-6)
            score = ang + 20 * np.log(ratio)
            up_to_down = vy[i - 2] < 0 and vy[i + 2] > 0
            if (ang >= c["min_angle_deg"] or ratio > 2.5) and not (up_to_down and ang < 90):
                cand.append((score, i))
        taken = []
        for _score, i in sorted(cand, reverse=True):
            if all(abs(i - j) >= c["min_sep_frames"] for j in taken):
                taken.append(i)
        for i in sorted(taken):
            rows.append((int(f[i]), int(seg), float(x[i]), float(y[i]), float(sp[min(i + 2, len(sp) - 1)])))
    return pd.DataFrame(rows, columns=["frame", "seg", "x", "y", "speed_px"])


def attach_players(contacts, players, max_frames=3):
    if not len(contacts) or not len(players):
        contacts = contacts.copy()
        for k in ["track_id", "side", "court_x", "court_y", "player_dist"]:
            contacts[k] = np.nan
        return contacts
    pl = players[players.on_court]
    out = []
    for r in contacts.itertuples():
        g = pl[(pl.frame - r.frame).abs() <= max_frames]
        if not len(g):
            out.append((np.nan, None, np.nan, np.nan, np.nan))
            continue
        cx = np.clip(r.x, g.x1, g.x2)
        cy = np.clip(r.y, g.y1, g.y2)
        h = (g.y2 - g.y1).clip(lower=1)
        d = np.hypot(r.x - cx, r.y - cy) / h
        k = int(np.argmin(d.to_numpy()))
        b = g.iloc[k]
        out.append((int(b.track_id), b.side, b.court_x, b.court_y, float(d.iloc[k])))
    contacts = contacts.copy()
    contacts[["track_id", "side", "court_x", "court_y", "player_dist"]] = pd.DataFrame(out, index=contacts.index)
    return contacts


def label_touches(contacts, rallies, max_dist, net_y, block_window):
    contacts = contacts.copy()
    contacts["rally_id"] = -1
    contacts["label"] = "unknown"
    contacts["touch_n"] = 0
    for r in rallies.itertuples():
        m = (contacts.frame >= r.start_frame) & (contacts.frame <= r.end_frame)
        contacts.loc[m, "rally_id"] = r.rally_id
    contacts.loc[contacts.player_dist > max_dist, "side"] = None
    for _rid, g in contacts[contacts.rally_id >= 0].groupby("rally_id"):
        side, n, prev = None, 0, None
        for k, (i, r) in enumerate(g.iterrows()):
            s = r.side if isinstance(r.side, str) else None
            if k == 0:
                lab, n, side = "serve", 0, s
            elif prev is not None and prev[1] == "attack" and s and s != side and abs(r.court_y - net_y) < 2.0 and r.frame - prev[0] <= block_window:
                lab, n, side = "block", 0, s
            else:
                if s is None or s != side:
                    n, side = 1, s
                else:
                    n += 1
                lab = {1: "pass", 2: "set", 3: "attack"}.get(n, "extra")
            contacts.loc[i, ["label", "touch_n"]] = [lab, n]
            prev = (r.frame, lab)
    return contacts


def serve_speeds(contacts, fps, ball=None, P=None):
    rows = []
    for rid, g in contacts[contacts.rally_id >= 0].groupby("rally_id"):
        g = g.sort_values("frame")
        if len(g) < 2 or g.iloc[0].label != "serve":
            continue
        s, r = g.iloc[0], g.iloc[1]
        dt = (r.frame - s.frame) / fps
        d = float(np.hypot(r.court_x - s.court_x, r.court_y - s.court_y)) if dt > 0 and not (np.isnan(s.court_x) or np.isnan(r.court_x)) else np.nan
        v3d, rms, status = np.nan, np.nan, "no_camera"
        if P is not None and ball is not None:
            seg = ball[(ball.frame >= s.frame) & (ball.frame <= r.frame)]
            status = "short_segment"
            if len(seg) >= 6:
                fit = fit_ballistic(P, seg.frame.to_numpy() / fps, seg[["x", "y"]].to_numpy())
                v3d, rms, status = fit["avg_kmh"], fit["rms_px"], "ok"
        rows.append((rid, int(s.frame), int(r.frame), dt, d, d / dt * 3.6 if dt > 0 else np.nan, v3d, rms, status))
    return pd.DataFrame(
        rows,
        columns=[
            "rally_id",
            "serve_frame",
            "receive_frame",
            "flight_s",
            "ground_dist_m",
            "avg_ground_speed_kmh",
            "ballistic_avg_kmh",
            "ballistic_rms_px",
            "fit_status",
        ],
    )


def find_jumps(players, fps, c):
    rows = []
    for tid, g in players[players.on_court].groupby("track_id"):
        g = g.sort_values("frame")
        if len(g) < fps:
            continue
        f = g.frame.to_numpy()
        foot = g.y2.to_numpy()
        h = (g.y2 - g.y1).to_numpy()
        base = pd.Series(foot).rolling(int(fps * 2), center=True, min_periods=1).quantile(0.8).to_numpy()
        rise = (base - foot) / np.maximum(h, 1)
        air = rise > c["min_rise_frac"]
        i = 0
        while i < len(air):
            if air[i]:
                j = i
                while j < len(air) and air[j]:
                    j += 1
                a0, b0 = i, j
                edge = c.get("edge_rise_frac", 0.0)
                while a0 > 0 and rise[a0 - 1] > edge:
                    a0 -= 1
                while b0 < len(air) and rise[b0] > edge:
                    b0 += 1
                t = (f[b0] - f[a0 - 1]) / fps - 1 / fps if 0 < a0 and b0 < len(f) else (f[b0 - 1] - f[a0] + 1) / fps
                gap_ok = np.all(np.diff(f[i:j]) <= 2) if j - i > 1 else False
                if c["min_air_s"] <= t <= c["max_air_s"] and gap_ok:
                    pk = i + int(np.argmax(rise[i:j]))
                    rows.append(
                        (
                            int(tid),
                            int(f[a0]),
                            int(f[b0 - 1]),
                            int(f[pk]),
                            t,
                            G * t * t / 8 * 100,
                            float(rise[pk]),
                            g.side.iloc[pk],
                            float(g.court_x.iloc[pk]),
                            float(g.court_y.iloc[pk]),
                        )
                    )
                i = j
            else:
                i += 1
    return pd.DataFrame(
        rows, columns=["track_id", "takeoff_frame", "landing_frame", "peak_frame", "air_s", "height_cm_flight", "peak_rise_bbox", "side", "court_x", "court_y"]
    )


def run(cfg):
    meta = load_meta(cfg)
    tdir = stage_dir(cfg, "track")
    out = stage_dir(cfg, "events")
    ec = cfg["events"]
    ball = pd.read_csv(tdir / "ball.csv")
    players = pd.read_csv(tdir / "players.csv")
    rallies = segment_rallies(ball, meta, ec["rally"])
    rallies.to_csv(out / "rallies.csv", index=False, float_format="%.3f")
    contacts = attach_players(find_contacts(ball, ec["contact"]), players)
    contacts = label_touches(contacts, rallies, ec["contact"]["max_player_dist_m"], cfg["court"]["length_m"] / 2, int(meta["fps"] * 0.6))
    contacts.to_csv(out / "contacts.csv", index=False, float_format="%.3f")
    hpath = stage_dir(cfg, "calibrate") / "homography.json"
    P = None
    if hpath.exists():
        H, hmeta, _ = load_homography(hpath)
        cam = camera_from_h(H, hmeta.get("image_size") or [meta["width"], meta["height"]])
        P = cam["P"] if cam else None
    serves = serve_speeds(contacts, meta["fps"], ball, P)
    serves.to_csv(out / "serves.csv", index=False, float_format="%.3f")
    jumps = find_jumps(players, meta["fps"], ec["jump"])
    jumps.to_csv(out / "jumps.csv", index=False, float_format="%.3f")
    fit_status = {k: int(v) for k, v in serves.fit_status.value_counts().items()} if len(serves) else {}
    return {"rallies": len(rallies), "contacts": len(contacts), "serves": len(serves), "jumps": len(jumps), "camera_ok": P is not None, "serve_fit": fit_status}
