from pathlib import Path
from malt.search.value_iteration import ValueIterationConfig, value_iteration_over_jsonl

input_path = Path("data/gsm8k_trajectories.jsonl")
value_iteration_over_jsonl(input_path, cfg=ValueIterationConfig(task="gsm8k"))