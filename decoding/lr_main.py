from lr_train import lr_train
from lr_bootstrap import lr_bootstrap
from lr_permutation import run_all_permutations
from lr_significance import lr_significance
from lr_visualize import lr_visualize, lr_visualize_all_models

# model_id = "qwen3_vl_30b_a3b_instruct"

# for model_id in ["qwen3_vl_32b_instruct", "qwen3_vl_30b_a3b_instruct"]:
#     lr_train(model_id)
#     lr_bootstrap(model_id)
#     run_all_permutations(model_id, n_permutations=500)
#     lr_significance(model_id)
#     lr_visualize(model_id)

# for model_id in ["qwen3_vl_4b_instruct", "qwen3_vl_32b_instruct", "qwen3_vl_30b_a3b_instruct"]:
#     lr_train(model_id)
#     lr_bootstrap(model_id)
#     run_all_permutations(model_id, n_permutations=1000)
#     lr_significance(model_id)
#     lr_visualize(model_id)

lr_visualize_all_models("qwen")