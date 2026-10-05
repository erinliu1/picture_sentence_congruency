from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    for model_id in ("gemma3_12b", "gemma3_27b"):
        # Isolate model loads so CUDA state and model memory are released between models.
        subprocess.run(
            [sys.executable, "-B", "-c",
             f"from hs_gemma3 import extract_hidden_states; "
             f"extract_hidden_states({model_id!r}, instructions=True)"],
            cwd=Path(__file__).resolve().parent,
            check=True,
        )
