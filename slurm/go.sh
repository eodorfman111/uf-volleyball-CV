#!/usr/bin/env bash
cd /blue/uf-sal/leodorfman/uf-volleyball-CV
git pull -q
mkdir -p logs
bash slurm/next.sh 2>&1 | cut -c1-110 | tee logs/go.out
