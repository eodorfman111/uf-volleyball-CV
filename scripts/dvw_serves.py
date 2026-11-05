import argparse
import csv


def serve_frames(rallies_csv, offset, fps, lo_s, hi_s):
    out = []
    for r in csv.DictReader(open(rallies_csv)):
        if not r["start_time"]:
            continue
        t = float(r["start_time"]) - offset
        if lo_s <= t < hi_s:
            out.append(int(round(t * fps)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dvw_dir")
    p.add_argument("--offset", type=float, required=True)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--start-s", type=float, required=True)
    p.add_argument("--end-s", type=float, required=True)
    a = p.parse_args()
    print(",".join(str(f) for f in serve_frames(f"{a.dvw_dir}/rallies.csv", a.offset, a.fps, a.start_s, a.end_s)))


if __name__ == "__main__":
    main()
