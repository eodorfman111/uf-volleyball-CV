import argparse

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("inputs", nargs="+")
p.add_argument("--exclude", nargs="*", default=[], metavar="START:END", help="drop labels with START <= frame < END")
a = p.parse_args()
df = pd.concat([pd.read_csv(f) for f in a.inputs]).drop_duplicates("frame", keep="last").sort_values("frame")
for e in a.exclude:
    lo, hi = map(int, e.split(":"))
    before = len(df)
    df = df[(df.frame < lo) | (df.frame >= hi)]
    print(f"dropped {before - len(df)} labels in {lo}:{hi}")
df.to_csv(a.out, index=False)
print(f"{len(df)} frames ({int(df.visible.sum())} with ball) -> {a.out}")
