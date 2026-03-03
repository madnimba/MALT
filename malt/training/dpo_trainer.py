from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from torch.utils.data import Dataset
from transformers import TrainingArguments, PreTrainedTokenizerBase
from trl import DPOTrainer

from malt.data.preference_builders import (
    VerifierDpoSample,
    RefinerDpoSample,
    build_verifier_dpo_samples,
    build_refiner_dpo_samples,
)
from malt.models import (
    MaltModelConfig,
    load_malt_llama_with_adapters,
    set_active_role_adapter,
    ROLE_VERIFIER,
    ROLE_REFINER,
)
from malt.models.prompts import (
    build_verifier_prompt,
    build_refiner_prompt,
)
from malt.search.value_iteration import (
    ValueIterationConfig,
    apply_value_iteration_to_trajectories,
)
from malt.utils.io import read_jsonl


@dataclass
class DpoTrainingConfig:
    """
    Configuration for DPO training of Verifier / Refiner adapters.
    """

    output_dir: Path
    beta: float = 0.2

    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    max_seq_length: int = 1024
    max_train_samples: int | None = None

    logging_steps: int = 50
    save_steps: int = 500
    save_total_limit: int = 2

    bf16: bool = False
    fp16: bool = True


class DpoTextPairDataset(Dataset):
    """
    Dataset for DPO training.

    Each item is a dict with:
        {
          "prompt": str,
          "chosen": str,
          "rejected": str,
        }

    DPOTrainer (from `trl`) handles tokenization internally using these
    fields, so we do not tokenize here.
    """

    def __init__(self, samples: Sequence[Tuple[str, str, str]]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        prompt, chosen, rejected = self.samples[idx]
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }


def _build_verifier_dpo_text_triples_from_trajectories(
    valued_trajectories_path: Path,
    max_train_samples: int | None,
) -> List[Tuple[str, str, str]]:
    """
    Build (prompt, chosen, rejected) triples for Verifier DPO.
    """
    trajs = read_jsonl(valued_trajectories_path)
    valued = apply_value_iteration_to_trajectories(
        trajectories=trajs,
        cfg=ValueIterationConfig(task="gsm8k"),
    )
    samples: List[VerifierDpoSample] = build_verifier_dpo_samples(valued)

    if max_train_samples is not None:
        samples = samples[: max_train_samples]

    triples: List[Tuple[str, str, str]] = []
    for s in samples:
        prompt = build_verifier_prompt(s.question, s.generator_output)
        triples.append((prompt, s.chosen, s.rejected))
    return triples


def _build_refiner_dpo_text_triples_from_trajectories(
    valued_trajectories_path: Path,
    max_train_samples: int | None,
) -> List[Tuple[str, str, str]]:
    """
    Build (prompt, chosen, rejected) triples for Refiner DPO.
    """
    trajs = read_jsonl(valued_trajectories_path)
    valued = apply_value_iteration_to_trajectories(
        trajectories=trajs,
        cfg=ValueIterationConfig(task="gsm8k"),
    )
    samples: List[RefinerDpoSample] = build_refiner_dpo_samples(valued)

    if max_train_samples is not None:
        samples = samples[: max_train_samples]

    triples: List[Tuple[str, str, str]] = []
    for s in samples:
        prompt = build_refiner_prompt(
            s.question,
            s.generator_output,
            s.verifier_output,
        )
        triples.append((prompt, s.chosen, s.rejected))
    return triples


def _build_dpo_trainer(
    model,
    ref_model,
    train_dataset: Dataset,
    cfg: DpoTrainingConfig,
):
    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        report_to=[],
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=cfg.beta,
        train_dataset=train_dataset,
    )
    return dpo_trainer


def train_verifier_dpo(
    valued_trajectories_path: Path,
    cfg: DpoTrainingConfig,
    model_cfg: MaltModelConfig | None = None,
):
    """
    Train the Verifier adapter with DPO on valued trajectories.

    Assumes the Verifier adapter has already been SFT-trained; both `model`
    and `ref_model` are initialized identically from that checkpoint and
    the DPOTrainer updates only `model`.
    """
    model_cfg = model_cfg or MaltModelConfig()
    model, tokenizer = load_malt_llama_with_adapters(model_cfg)
    set_active_role_adapter(model, ROLE_VERIFIER)

    # Reference model is a frozen copy of the starting SFT weights.
    ref_model, _ = load_malt_llama_with_adapters(model_cfg)
    set_active_role_adapter(ref_model, ROLE_VERIFIER)

    triples = _build_verifier_dpo_text_triples_from_trajectories(
        valued_trajectories_path=valued_trajectories_path,
        max_train_samples=cfg.max_train_samples,
    )
    dataset = DpoTextPairDataset(triples)

    dpo_trainer = _build_dpo_trainer(
        model=model,
        ref_model=ref_model,
        train_dataset=dataset,
        cfg=cfg,
    )
    dpo_trainer.train()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.output_dir))


def train_refiner_dpo(
    valued_trajectories_path: Path,
    cfg: DpoTrainingConfig,
    model_cfg: MaltModelConfig | None = None,
):
    """
    Train the Refiner adapter with DPO on valued trajectories.
    """
    model_cfg = model_cfg or MaltModelConfig()
    model, tokenizer = load_malt_llama_with_adapters(model_cfg)
    set_active_role_adapter(model, ROLE_REFINER)

    ref_model, _ = load_malt_llama_with_adapters(model_cfg)
    set_active_role_adapter(ref_model, ROLE_REFINER)

    triples = _build_refiner_dpo_text_triples_from_trajectories(
        valued_trajectories_path=valued_trajectories_path,
        max_train_samples=cfg.max_train_samples,
    )
    dataset = DpoTextPairDataset(triples)

    dpo_trainer = _build_dpo_trainer(
        model=model,
        ref_model=ref_model,
        train_dataset=dataset,
        cfg=cfg,
    )
    dpo_trainer.train()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.output_dir))

