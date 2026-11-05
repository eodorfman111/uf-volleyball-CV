#!/usr/bin/env bash
cd /blue/uf-sal/leodorfman/uf-volleyball-CV
module load conda
conda activate /blue/uf-sal/leodorfman/conda/envs/vbcv
PLAYS=$(python scripts/dvw_serves.py /blue/uf-sal/leodorfman/dvw/ufuk --offset 526 --start-s 7500 --end-s 8100)
echo "tuning half: $(echo "$PLAYS" | tr ',' '\n' | wc -l) scout serves"
python scripts/sweep_rally.py -c configs/ufuk_sec.yaml --plays "$PLAYS" --span 225000 243000 --grid 'events.rally.min_speed_px=0|10|20|30|40' --grid 'events.rally.min_rally_s=2|3|4|5' --grid 'events.rally.merge_gap_s=2|3|5|8' --grid 'events.rally.min_ball_ratio=0.3|0.4|0.5|0.6' | sed 's/events.rally.//g' | head -12
