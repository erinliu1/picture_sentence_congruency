import sys

from alignment_rep_bev import get_alignment
from make_final_layer_pairs import make_final_layer_pairs
from visualize_instruction_comparison import plot_instruction_comparison


def run_alignment(model_id):
    get_alignment(model_id, instructions=True)
    make_final_layer_pairs(model_id, instructions=True)
    plot_instruction_comparison(model_id)


if __name__ == "__main__":
    model_ids = sys.argv[1:] or ("gemma4_e2b", "gemma4_e4b", "gemma4_12b", "gemma4_26b_a4b", "gemma4_31b")
    for model_id in model_ids:
        run_alignment(model_id)
