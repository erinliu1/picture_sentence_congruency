from lr_train import lr_train
from lr_bootstrap import lr_bootstrap
from visualize_instruction_comparison import plot_instruction_comparison


if __name__ == "__main__":
    for model_id in ("gemma3_12b", "gemma3_27b"):
        lr_train(model_id, instructions=True)
        lr_bootstrap(model_id, instructions=True)
        plot_instruction_comparison(model_id)
