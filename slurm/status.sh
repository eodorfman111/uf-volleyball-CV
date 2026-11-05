#!/usr/bin/env bash
cd "$(dirname "$0")/.."
echo "== queue =="
squeue -u "$USER" -o "%.10i %.9P %.10j %.2t %.9M %.30R"
echo
echo "== latest logs (last 3 lines each) =="
for f in $(ls -t logs/*.out 2>/dev/null | head -${1:-4}); do
  echo "-- $f"
  tail -n 3 "$f" | cut -c1-200
done
echo
echo "== runs =="
ls -t runs 2>/dev/null | head -8
echo
echo "== scout/label files =="
ls /blue/uf-sal/leodorfman/labels /blue/uf-sal/leodorfman/labels_test /blue/uf-sal/leodorfman/dvw 2>/dev/null | head -20
