from pathlib import Path
from malt.search import TreeSearchConfig, run_tree_search_for_gsm8k_split

cfg = TreeSearchConfig(branching_factor=3, output_path=Path("data/gsm8k_trajectories.jsonl"))
run_tree_search_for_gsm8k_split("train", cfg)