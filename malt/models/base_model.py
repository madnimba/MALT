from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

try:
    # BitsAndBytesConfig is only available when bitsandbytes is installed.
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None  # type: ignore

from peft import LoraConfig, get_peft_model, PeftModel


ROLE_GENERATOR: Literal["generator"] = "generator"
ROLE_VERIFIER: Literal["verifier"] = "verifier"
ROLE_REFINER: Literal["refiner"] = "refiner"


@dataclass
class MaltModelConfig:
    """
    Configuration for loading the base Llama model and attaching LoRA adapters.

    The defaults are chosen to be compatible with a single 24GB GPU using
    4-bit quantization (QLoRA-style).
    """

    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    load_in_4bit: bool = True
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.bfloat16

    # LoRA configuration
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def _build_quantization_config(cfg: MaltModelConfig):
    """
    Build a BitsAndBytes quantization config if requested and available.
    """
    if not cfg.load_in_4bit:
        return None
    if BitsAndBytesConfig is None:
        raise ImportError(
            "BitsAndBytesConfig not available. Make sure `bitsandbytes` is "
            "installed if you want 4-bit loading, or set load_in_4bit=False "
            "in MaltModelConfig."
        )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=cfg.torch_dtype,
    )


def load_malt_llama_with_adapters(
    cfg: MaltModelConfig | None = None,
) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Load a 4-bit-quantized Llama-3.1-8B-Instruct base model and attach
    three LoRA adapters for the Generator, Verifier, and Refiner roles.

    Returns a PEFT-wrapped model and its tokenizer.

    The returned model contains three named adapters:
        - "generator"
        - "verifier"
        - "refiner"
    Use `set_active_role_adapter` to switch between them at inference / training time.
    """
    cfg = cfg or MaltModelConfig()

    quant_config = _build_quantization_config(cfg)

    # Load base model (optionally in 4-bit)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map=cfg.device_map,
        quantization_config=quant_config,
        torch_dtype=cfg.torch_dtype if quant_config is None else None,
    )
    print("loaded llama")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        # Use eos_token as pad_token to enable left padding for generation.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Create a shared LoRA configuration
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
    )
    print("lora coooonfig crrrrrrrrrreated")

    # Wrap the model in PEFT and register three role-specific adapters.
    peft_model = get_peft_model(model, lora_config, adapter_name=ROLE_GENERATOR)
    # Add additional adapters for other roles with the same LoRA hyperparameters.
    peft_model.add_adapter(ROLE_VERIFIER, lora_config)
    peft_model.add_adapter(ROLE_REFINER, lora_config)

    # Default to generator role.
    peft_model.set_adapter(ROLE_GENERATOR)

    print("added roleeee adapters")

    return peft_model, tokenizer


def set_active_role_adapter(model: PeftModel, role: str) -> None:
    """
    Switch the active LoRA adapter on a shared base model.

    Args:
        model: A PeftModel with adapters named "generator", "verifier", "refiner".
        role: One of ROLE_GENERATOR, ROLE_VERIFIER, ROLE_REFINER.
    """
    if role not in (ROLE_GENERATOR, ROLE_VERIFIER, ROLE_REFINER):
        raise ValueError(f"Unknown role adapter: {role!r}")
    model.set_adapter(role)


def load_malt_llama_with_trained_adapters(
    cfg: MaltModelConfig | None = None,
    generator_checkpoint: str | Path | None = None,
    verifier_checkpoint: str | Path | None = None,
    refiner_checkpoint: str | Path | None = None,
) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Load the base Llama model with role adapters, then optionally load
    trained adapters for each role from PEFT checkpoints on disk.

    Each *_checkpoint argument should point to a directory created by
    `model.save_pretrained(...)` in the SFT/DPO training code. When
    provided, the corresponding adapter weights are loaded into the
    shared PEFT model under the canonical adapter names:
      - "generator"
      - "verifier"
      - "refiner"
    """
    model, tokenizer = load_malt_llama_with_adapters(cfg)

    # Normalize to strings for peft.load_adapter
    if verifier_checkpoint is not None:
        ckpt = Path(verifier_checkpoint)
        model.load_adapter(str((ckpt / ROLE_VERIFIER).resolve()), adapter_name=ROLE_VERIFIER)
    if generator_checkpoint is not None:
        ckpt = Path(generator_checkpoint)
        model.load_adapter(str((ckpt / ROLE_GENERATOR).resolve()), adapter_name=ROLE_GENERATOR)

    if refiner_checkpoint is not None:
        ckpt = Path(refiner_checkpoint)
        model.load_adapter(str((ckpt / ROLE_REFINER).resolve()), adapter_name=ROLE_REFINER)

    # Default active adapter: generator
    model.set_adapter(ROLE_GENERATOR)

    return model, tokenizer

