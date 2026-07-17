# Run from the Euler LOGIN node, not a compute node - file uploads go through
# storage.googleapis.com directly, which only the login node can reach.
# Usage: .venv/bin/python scripts/upload_to_wandb.py

from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parents[1]
api = wandb.Api()
runs_by_name = {r.name: r for r in api.runs("volpfn/volpfn")}

for run_dir in (ROOT / "checkpoints").iterdir():
    run = runs_by_name.get(run_dir.name)
    if run is None:
        continue
    for fname in ["train.log", "eval.txt", "final.pt"]:
        path = run_dir / fname
        if path.exists():
            run.upload_file(str(path), root=str(run_dir))
    print(f"{run_dir.name}: uploaded")
