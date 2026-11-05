#!/usr/bin/env bash
cd /blue/uf-sal/leodorfman/uf-volleyball-CV
bash slurm/status.sh 3
ls -la runs/ufuk_full/detect 2>/dev/null | tail -3
