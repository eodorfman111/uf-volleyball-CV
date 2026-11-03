import sys

import pandas as pd

out, *ins = sys.argv[1:]
df = pd.concat([pd.read_csv(f) for f in ins]).drop_duplicates("frame", keep="last").sort_values("frame")
df.to_csv(out, index=False)
print(f"{len(df)} frames ({int(df.visible.sum())} with ball) -> {out}")
