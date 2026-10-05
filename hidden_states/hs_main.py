"""Run selected PLM models through extraction, decoding, and alignment in order."""

import json
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    with (root / "models/models_lookup.json").open(encoding="utf-8") as file:
        registry = json.load(file)
    model_ids = sys.argv[1:] or ["plm_1b", "plm_3b", "plm_8b"]
    if any(model_id not in registry or registry[model_id]["family"] != "plm" for model_id in model_ids):
        raise ValueError("Select registered PLM models.")
    for model_id in model_ids:
        stages = [
            ("extraction", "hidden_states", ["-c", f"from hs_plm import extract_hidden_states; extract_hidden_states({model_id!r}, instructions=True)"]),
            ("decoding", "decoding", ["lr_main.py", model_id]),
            ("alignment", "alignment", ["alignment_main.py", model_id]),
        ]
        for stage, directory, args in stages:
            print(f"START {model_id}: {stage}", flush=True)
            subprocess.run([sys.executable, "-u", "-B", *args], cwd=root / directory, check=True)
        print(f"COMPLETE {model_id}: both comparison plots saved", flush=True)
