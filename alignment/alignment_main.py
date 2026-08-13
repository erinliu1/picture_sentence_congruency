from alignment_rep_bev import get_alignment
from alignment_visualize import plot_alignment

model_id = "qwen3_vl_30b_a3b_instruct"

get_alignment(model_id)
plot_alignment(model_id)