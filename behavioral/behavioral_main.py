from behavioral_qwen import extract_ratings
from behavioral_accuracy import save_all_accuracy
from behavioral_correlation import save_all_correlation
from behavioral_health import save_all_health

dense_models = [f"qwen3_vl_{size}b_instruct" for size in [2, 4, 8, 32]]
moe_models = ["qwen3_vl_30b_a3b_instruct"]

# run what model you need here
# extract_ratings("qwen3_vl_235b_a22b_instruct")

save_all_accuracy()
save_all_correlation()
save_all_health()