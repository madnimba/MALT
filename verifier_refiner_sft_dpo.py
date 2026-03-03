from pathlib import Path
import gc
import torch

from malt.training.sft_trainer import SftTrainingConfig, train_verifier_sft, train_refiner_sft
from malt.training.dpo_trainer import DpoTrainingConfig, train_verifier_dpo, train_refiner_dpo

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

valued_path = Path("data/gsm8k_trajectories.valued.jsonl")

v_sft_cfg = SftTrainingConfig(output_dir=Path("checkpoints/verifier_sft"), num_train_epochs=1)
r_sft_cfg = SftTrainingConfig(output_dir=Path("checkpoints/refiner_sft"), num_train_epochs=1)

train_verifier_sft(valued_path, v_sft_cfg)
cleanup()

train_refiner_sft(valued_path, r_sft_cfg)
cleanup()

v_dpo_cfg = DpoTrainingConfig(output_dir=Path("checkpoints/verifier_dpo"), bf16=True, fp16=False)
r_dpo_cfg = DpoTrainingConfig(output_dir=Path("checkpoints/refiner_dpo"), bf16=True, fp16=False)

train_verifier_dpo(valued_path, v_dpo_cfg)
cleanup()

train_refiner_dpo(valued_path, r_dpo_cfg)
cleanup()