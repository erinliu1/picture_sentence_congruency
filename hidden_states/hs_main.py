"""Run each LLaVA model through extraction, decoding, and alignment in order."""

import json
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    with (root / "models/models_lookup.json").open(encoding="utf-8") as file:
        registry = json.load(file)
    model_ids = [model_id for model_id, info in registry.items() if info["family"] == "llava"]
    for model_id in model_ids:
        stages = [
            ("extraction", "hidden_states", ["-c", f"from hs_llava import extract_hidden_states; extract_hidden_states({model_id!r}, instructions=True)"]),
            ("decoding", "decoding", ["lr_main.py", model_id]),
            ("alignment", "alignment", ["alignment_main.py", model_id]),
        ]
        for stage, directory, args in stages:
            print(f"START {model_id}: {stage}", flush=True)
            subprocess.run([sys.executable, "-u", "-B", *args], cwd=root / directory, check=True)
        print(f"COMPLETE {model_id}: both comparison plots saved", flush=True)
