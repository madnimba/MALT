from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import logging
import os
import signal
import subprocess
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None  # type: ignore

from peft import LoraConfig, get_peft_model, PeftModel

log = logging.getLogger(__name__)

ROLE_GENERATOR: Literal["generator"] = "generator"
ROLE_VERIFIER: Literal["verifier"] = "verifier"
ROLE_REFINER: Literal["refiner"] = "refiner"

# Llama-3.1-8B in 4-bit + three LoRA adapters needs roughly this much VRAM.
_MIN_FREE_VRAM_GIB = 14.0


@dataclass
class MaltModelConfig:
    """
    Configuration for loading the base Llama model and attaching LoRA adapters.

    The defaults are chosen to be compatible with a single 24GB GPU using
    4-bit quantization (QLoRA-style).

    Notes on torch_compile:
        torch.compile(mode="reduce-overhead") internally spawns worker
        subprocesses for Triton kernel compilation. These processes inherit the
        parent's CUDA context and each hold a slice of VRAM for the duration of
        the run. On a shared or single-GPU machine this is a frequent source of
        the "stale GPU processes" OOM described in the logs. Set
        use_torch_compile=False if you are running on a machine where VRAM is
        tight or other jobs share the GPU.
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


# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------

def _gib(bytes_: int) -> float:
    return bytes_ / (1024 ** 3)


def _free_vram_gib(device: int = 0) -> float:
    """Return free VRAM on the given CUDA device in GiB, or 0.0 if unavailable."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        free, _ = torch.cuda.mem_get_info(device)
        return _gib(free)
    except Exception:
        return 0.0


def _get_other_gpu_pids(device: int = 0) -> list[int]:
    """
    Use nvidia-smi to find PIDs using the GPU other than the current process.
    Returns an empty list if nvidia-smi is unavailable or parsing fails.
    """
    own_pid = os.getpid()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        pids = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if line.isdigit():
                pid = int(line)
                if pid != own_pid:
                    pids.append(pid)
        return pids
    except Exception:
        return []


def _kill_pids(pids: list[int]) -> None:
    """Send SIGKILL to each pid, ignoring errors for already-dead processes."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            log.info("Killed stale GPU process %d", pid)
        except ProcessLookupError:
            pass  # already gone
        except PermissionError:
            log.warning("No permission to kill PID %d — skipping", pid)


def release_gpu_memory() -> None:
    """Free PyTorch's cached (but unallocated) VRAM before loading."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        log.info("GPU cache cleared. Free VRAM: %.1f GiB", _free_vram_gib())


def check_gpu_memory(
    min_free_gib: float = _MIN_FREE_VRAM_GIB,
    device: int = 0,
    auto_kill: bool = True,
    poll_interval_seconds: float = 2.0,
    poll_max_wait_seconds: float = 30.0,
) -> None:
    """
    Ensure there is enough free VRAM to load the model.

    If free VRAM is below *min_free_gib* and *auto_kill* is True:
      1. Sends SIGKILL to every other compute process on the GPU.
      2. Polls nvidia-smi every *poll_interval_seconds* until either:
           - The killed PIDs have fully disappeared from nvidia-smi AND
             free VRAM has reached *min_free_gib*, or
           - *poll_max_wait_seconds* has elapsed (raises RuntimeError).

    The polling loop is necessary because nvidia-smi continues reporting
    a killed process and its VRAM usage for several seconds after SIGKILL
    while the kernel finalises the CUDA context teardown. A single fixed
    sleep is not reliable — the reclaim time varies with how much CUDA state
    the process had allocated.

    If *auto_kill* is False, raises RuntimeError immediately with the
    kill command in the message.
    """
    if not torch.cuda.is_available():
        log.warning("CUDA not available — skipping VRAM check.")
        return

    free_gib = _free_vram_gib(device)
    total_gib = _gib(torch.cuda.get_device_properties(device).total_memory)
    log.info(
        "GPU %d: %.1f GiB free / %.1f GiB total (need %.1f GiB)",
        device, free_gib, total_gib, min_free_gib,
    )

    if free_gib >= min_free_gib:
        return

    other_pids = _get_other_gpu_pids(device)

    if not other_pids:
        raise RuntimeError(
            f"Insufficient VRAM: {free_gib:.1f} GiB free, {min_free_gib:.1f} GiB needed. "
            "No other compute processes detected — "
            "try setting PYTORCH_ALLOC_CONF=expandable_segments:True."
        )

    if not auto_kill:
        pid_str = " ".join(str(p) for p in other_pids)
        raise RuntimeError(
            f"Insufficient VRAM: {free_gib:.1f} GiB free, {min_free_gib:.1f} GiB needed. "
            f"Stale GPU processes: {other_pids}. "
            f"Kill them with: kill -9 {pid_str}"
        )

    # auto_kill path: kill then poll until nvidia-smi confirms they are gone
    # AND free VRAM is sufficient.
    log.warning(
        "Only %.1f GiB free (need %.1f GiB). Auto-killing stale GPU processes: %s",
        free_gib, min_free_gib, other_pids,
    )
    _kill_pids(other_pids)
    killed_set = set(other_pids)

    deadline = time.monotonic() + poll_max_wait_seconds
    attempt = 0
    while time.monotonic() < deadline:
        time.sleep(poll_interval_seconds)
        attempt += 1

        # Clear PyTorch's own allocator cache so mem_get_info reflects reality.
        release_gpu_memory()

        still_listed = set(_get_other_gpu_pids(device))
        free_gib_now = _free_vram_gib(device)

        # Processes that were killed and are still showing up in nvidia-smi.
        still_dying = killed_set & still_listed
        # Genuinely new processes (different PIDs, not from our kill list).
        new_pids = still_listed - killed_set

        log.info(
            "Poll %d: %.1f GiB free | still dying: %s | new PIDs: %s",
            attempt, free_gib_now,
            sorted(still_dying) or "none",
            sorted(new_pids) or "none",
        )

        if new_pids:
            # A truly new process appeared — warn but keep waiting; it may
            # just be a transient spawn that will also exit shortly.
            log.warning("New GPU processes detected during wait: %s", sorted(new_pids))

        if not still_dying and free_gib_now >= min_free_gib:
            log.info(
                "All killed processes gone. %.1f GiB free — proceeding.",
                free_gib_now,
            )
            return

    # Timed out.
    final_pids = _get_other_gpu_pids(device)
    final_free = _free_vram_gib(device)
    raise RuntimeError(
        f"Timed out after {poll_max_wait_seconds:.0f}s waiting for VRAM to free up. "
        f"Still only {final_free:.1f} GiB free (need {min_free_gib:.1f} GiB). "
        f"Remaining GPU processes: {final_pids}. "
        "If these are system processes you cannot kill, consider reducing "
        "min_free_gib in check_gpu_memory() or freeing VRAM another way."
    )


# ---------------------------------------------------------------------------
# Quantization config
# ---------------------------------------------------------------------------

def _build_quantization_config(cfg: MaltModelConfig):
    if not cfg.load_in_4bit:
        return None
    if BitsAndBytesConfig is None:
        raise ImportError(
            "BitsAndBytesConfig not available. Install `bitsandbytes` or set "
            "load_in_4bit=False in MaltModelConfig."
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=cfg.torch_dtype,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_malt_llama_with_adapters(
    cfg: MaltModelConfig | None = None,
) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Load a 4-bit-quantized Llama-3.1-8B-Instruct base model and attach
    three LoRA adapters for the Generator, Verifier, and Refiner roles.

    Automatically clears the GPU cache and kills any stale processes before
    loading. Returns a PEFT-wrapped model and its tokenizer with three named
    adapters: "generator", "verifier", "refiner".
    """
    cfg = cfg or MaltModelConfig()

    release_gpu_memory()
    check_gpu_memory()

    quant_config = _build_quantization_config(cfg)

    # core_model_loading.py uses "from concurrent.futures import ThreadPoolExecutor"
    # so patching concurrent.futures.ThreadPoolExecutor is not enough — the name
    # in that module's namespace is already bound. We must patch the attribute
    # directly on the module object, and also on concurrent.futures for safety.
    import concurrent.futures as _cf
    import importlib as _il

    # Patch both namespaces.
    _real_tpe = _cf.ThreadPoolExecutor
    class _SingleThreadedTPE(_real_tpe):
        def __init__(self, *a, **kw):
            kw["max_workers"] = 1
            super().__init__(*a, **kw)

    _cf.ThreadPoolExecutor = _SingleThreadedTPE

    # Also patch the transformers.core_model_loading module namespace directly,
    # since it does "from concurrent.futures import ThreadPoolExecutor" at import
    # time, binding the name locally and bypassing the concurrent.futures patch.
    try:
        import transformers.core_model_loading as _cml
        _real_cml_tpe = getattr(_cml, "ThreadPoolExecutor", None)
        if _real_cml_tpe is not None:
            _cml.ThreadPoolExecutor = _SingleThreadedTPE
    except ImportError:
        _real_cml_tpe = None
        _cml = None

    # Kill any new processes that appeared since the check (e.g. from an
    # external job scheduler) and wait for them to fully release VRAM before
    # handing control to from_pretrained.
    _pre_load_pids = set(_get_other_gpu_pids())
    if _pre_load_pids:
        log.warning(
            "New GPU processes appeared just before loading: %s — killing.",
            sorted(_pre_load_pids),
        )
        _kill_pids(list(_pre_load_pids))
        time.sleep(5.0)
        release_gpu_memory()
        log.info("Pre-load cleanup done. Free VRAM: %.1f GiB", _free_vram_gib())

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            device_map=cfg.device_map,
            quantization_config=quant_config,
            torch_dtype=cfg.torch_dtype if quant_config is None else None,
        )
    finally:
        # Restore both namespaces unconditionally.
        _cf.ThreadPoolExecutor = _real_tpe
        if _cml is not None and _real_cml_tpe is not None:
            _cml.ThreadPoolExecutor = _real_cml_tpe
    log.info("Base model loaded")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
    )

    peft_model = get_peft_model(model, lora_config, adapter_name=ROLE_GENERATOR)
    peft_model.add_adapter(ROLE_VERIFIER, lora_config)
    peft_model.add_adapter(ROLE_REFINER, lora_config)
    peft_model.set_adapter(ROLE_GENERATOR)

    log.info("LoRA adapters attached (generator / verifier / refiner)")

    return peft_model, tokenizer


def set_active_role_adapter(model: PeftModel, role: str) -> None:
    """Switch the active LoRA adapter on a shared base model."""
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
    trained adapter weights from PEFT checkpoints on disk.
    """
    model, tokenizer = load_malt_llama_with_adapters(cfg)

    if verifier_checkpoint is not None:
        ckpt = Path(verifier_checkpoint)
        model.load_adapter(str((ckpt / ROLE_VERIFIER).resolve()), adapter_name=ROLE_VERIFIER)
    if generator_checkpoint is not None:
        ckpt = Path(generator_checkpoint)
        model.load_adapter(str((ckpt / ROLE_GENERATOR).resolve()), adapter_name=ROLE_GENERATOR)
    if refiner_checkpoint is not None:
        ckpt = Path(refiner_checkpoint)
        model.load_adapter(str((ckpt / ROLE_REFINER).resolve()), adapter_name=ROLE_REFINER)

    model.set_adapter(ROLE_GENERATOR)

    return model, tokenizer