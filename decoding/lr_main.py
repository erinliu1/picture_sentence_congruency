import sys

from lr_train import lr_train
from lr_bootstrap import lr_bootstrap
from visualize_instruction_comparison import plot_instruction_comparison


def run_decoding(model_id):
    lr_train(model_id, instructions=True)
    lr_bootstrap(model_id, instructions=True)
    plot_instruction_comparison(model_id)


if __name__ == "__main__":
    model_ids = sys.argv[1:] or ("gemma4_e2b", "gemma4_e4b", "gemma4_12b", "gemma4_26b_a4b", "gemma4_31b")
    for model_id in model_ids:
        run_decoding(model_id)
