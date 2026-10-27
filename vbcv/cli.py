import argparse
import importlib
import sys
import time
import traceback

from .checkpoint import STAGES, is_done, log_failure, mark_done
from .config import load_config, run_dir


def run_stage(cfg, stage, force=False):
    if not force and is_done(cfg, stage):
        print(f"[{stage}] up to date, skipping ({run_dir(cfg) / stage})")
        return True
    mod = importlib.import_module(f"vbcv.stages.{stage}")
    started = time.time()
    print(f"[{stage}] running")
    for attempt in (1, 2):
        try:
            extra = mod.run(cfg)
            info = mark_done(cfg, stage, started, extra)
            print(f"[{stage}] done in {info['seconds']}s")
            return True
        except Exception as e:
            n = log_failure(cfg, stage, e)
            traceback.print_exc()
            print(f"[{stage}] attempt {attempt} failed ({n} total failures logged)")
    return False


def main(argv=None):
    p = argparse.ArgumentParser(prog="vbcv")
    p.add_argument("stage", nargs="+", choices=STAGES + ["all"])
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--continue-on-fail", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.set)
    todo = STAGES if "all" in a.stage else sorted(set(a.stage), key=STAGES.index)
    ok_all = True
    for s in todo:
        ok = run_stage(cfg, s, force=a.force)
        ok_all &= ok
        if not ok and not a.continue_on_fail:
            return 1
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
