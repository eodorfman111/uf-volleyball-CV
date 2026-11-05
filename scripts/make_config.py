import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vbcv.config import set_dotted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.src)) or {}
    for item in a.set:
        k, v = item.split("=", 1)
        set_dotted(cfg, k, v)
    with open(a.dst, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {a.dst}")


if __name__ == "__main__":
    main()
