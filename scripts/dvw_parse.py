import argparse
import csv
import collections
import statistics
from pathlib import Path

SKILLS = {"S": "serve", "R": "receive", "E": "set", "A": "attack", "B": "block", "D": "dig", "F": "freeball"}
TEAMS = {"*": "home", "a": "visitor"}


def scout_lines(path):
    inside = False
    for raw in open(path, encoding="latin-1").read().splitlines():
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            inside = s == "[3SCOUT]"
            continue
        if inside and s:
            yield s


def parse(path):
    touches, points = [], []
    rally = 0
    for n, line in enumerate(scout_lines(path)):
        f = line.split(";")
        code = f[0]
        if len(code) < 2 or code[0] not in TEAMS:
            continue

        def field(i):
            return f[i] if i < len(f) else ""

        try:
            vt = int(field(12))
        except ValueError:
            vt = None
        base = {"line": n, "team": TEAMS[code[0]], "set": field(8), "video_file": field(11), "video_time": vt}
        if code[1] == "p":
            score = code[2:].split(":")
            if len(score) == 2:
                points.append({**base, "rally": rally, "home_score": score[0], "visitor_score": score[1]})
            continue
        if len(code) >= 4 and code[3] in SKILLS and code[1:3].isdigit():
            if code[3] == "S":
                rally += 1
            touches.append(
                {
                    **base,
                    "rally": rally,
                    "player": int(code[1:3]),
                    "skill": SKILLS[code[3]],
                    "skill_type": code[4:5],
                    "evaluation": code[5:6],
                    "start_coord": field(4),
                    "mid_coord": field(5),
                    "end_coord": field(6),
                }
            )
    return touches, points


def rally_table(touches, points):
    by_rally = collections.defaultdict(list)
    for t in touches:
        by_rally[t["rally"]].append(t)
    end = {}
    for p in points:
        end.setdefault(p["rally"], p)
    rows = []
    for r, ts in sorted(by_rally.items()):
        first = ts[0]
        last = end.get(r)
        times = [t["video_time"] for t in ts if t["video_time"] is not None]
        rows.append(
            {
                "rally": r,
                "set": first["set"],
                "serving_team": first["team"] if first["skill"] == "serve" else "",
                "start_time": first["video_time"],
                "end_time": last["video_time"] if last and last["video_time"] is not None else (max(times) if times else None),
                "touches": len(ts),
                "attacks": sum(t["skill"] == "attack" for t in ts),
                "blocks": sum(t["skill"] == "block" for t in ts),
                "point_to": last["team"] if last else "",
            }
        )
    return rows


def write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dvw")
    p.add_argument("out_dir")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    touches, points = parse(a.dvw)
    rallies = rally_table(touches, points)
    write(out / "touches.csv", touches)
    write(out / "points.csv", points)
    write(out / "rallies.csv", rallies)
    print(f"{len(touches)} touches, {len(rallies)} rallies, {len(points)} point lines")
    print("skills:", dict(collections.Counter(t["skill"] for t in touches)))
    print("touches per rally: median", statistics.median(r["touches"] for r in rallies), "max", max(r["touches"] for r in rallies))
    d = [r["end_time"] - r["start_time"] for r in rallies if r["end_time"] is not None and r["start_time"] is not None and r["end_time"] >= r["start_time"]]
    if d:
        print(f"rally length (video seconds): median {statistics.median(d):.0f}, mean {statistics.mean(d):.1f}, max {max(d)}")
    times = [t["video_time"] for t in touches if t["video_time"] is not None]
    print("video time range (s):", min(times), "to", max(times), "; touches without a video time:", len(touches) - len(times))
    print("rallies per set:", dict(sorted(collections.Counter(r["set"] for r in rallies).items())))
    print("wrote", out)


if __name__ == "__main__":
    main()
