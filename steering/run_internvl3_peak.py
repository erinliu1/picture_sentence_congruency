"""Run InternVL3-1B steering at its peak rating-readout decoding layer."""

from pathlib import Path

import pandas as pd

from save_steering_probes import save_steering_probes
import save_projections
from save_steered_logits import process_all
from save_steering_sensitivity import get_slope_bootstrap_distribution


def run():
    model_id = "internvl3_1b_instruct"
    root = Path(__file__).resolve().parents[1]
    accuracy = pd.read_csv(root / "models" / model_id / "decoding_instruction/layerwise_accuracy.csv")
    peak = accuracy.sort_values(["all_accuracy", "layer_index"], ascending=[False, True]).iloc[0]
    layer = int(peak.layer_index)
    print(f"Peak rating readout: layer {layer}, accuracy {peak.all_accuracy:.6f}", flush=True)
    save_steering_probes(model_id, layers=[layer])
    save_projections.MODEL_ID = model_id
    save_projections.save_all_projections()
    save_projections.check_raw_hs_perturbation_all([-2, -1, 0, 1, 2])
    process_all(layers=[layer], k_list=[-2, -1, 0, 1, 2], model_id=model_id)
    get_slope_bootstrap_distribution(root / "steering/results" / model_id / "steering_summary.csv")


if __name__ == "__main__":
    run()
