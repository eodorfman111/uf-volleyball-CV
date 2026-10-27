# uf-volleyball-cv (`vbcv`)

Computer-vision pipeline for indoor volleyball filmed by a **fixed camera behind the end line**.
Stages: `ingest -> calibrate -> detect -> track -> events -> stats`, config-driven, resumable, one command per stage.

Start here: [HANDOFF.md](HANDOFF.md). Results: [EVAL.md](EVAL.md). Literature/model survey: [SURVEY.md](SURVEY.md).
Humans: [LABELING_QUEUE.md](LABELING_QUEUE.md), [docs/LABELING_GUIDE.md](docs/LABELING_GUIDE.md), [FOOTAGE_REQUEST.md](FOOTAGE_REQUEST.md).

**UF athlete/team footage must stay on HiPerGator.** Never copy it to laptops, cloud tools, or third-party APIs, and never commit frames or labels.

## Quickstart (public clip, CPU)
```
python -m venv ~/venv && source ~/venv/bin/activate
pip install -r requirements.txt && pip install -e .
bash scripts/fetch_models.sh
git clone https://github.com/asigatchov/fast-volleyball-tracking-inference ~/ref/fvti
python -m vbcv all -c configs/public_gtu.yaml
python scripts/render_overlay.py -c configs/public_gtu.yaml --start 0 --end 400
```
Outputs land in `runs/public_gtu/<stage>/`. Each stage writes `_done.json`; rerunning skips finished stages unless the stage's config changed or `--force` is passed. Failures are appended to `_failures.jsonl` and each stage is retried once.

## HiPerGator
```
bash slurm/setup_env.sh                       # once: conda env + models
export SBATCH_ACCOUNT=<group> SBATCH_PARTITION=<partition>   # confirm both with Research Computing
bash slurm/submit_match.sh configs/<match>.yaml               # CPU ingest+calibrate -> GPU detect -> CPU track+events+stats
STAGES="track events stats" CONFIG=configs/<match>.yaml sbatch slurm/stage.sbatch   # rerun CPU stages only
DATA=/blue/<group>/<user>/ball_ds sbatch slurm/train_ball.sbatch
```

## Tests and lint
```
python -m pytest -q tests
ruff check vbcv scripts tests && ruff format --check vbcv scripts tests
```
