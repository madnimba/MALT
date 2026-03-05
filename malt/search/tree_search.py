from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Literal, Sequence

import torch
from peft import PeftModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from malt.data import Gsm8kExample
from malt.models import (
    MaltModelConfig,
    load_malt_llama_with_adapters,
    set_active_role_adapter,
    ROLE_GENERATOR,
    ROLE_VERIFIER,
    ROLE_REFINER,
)
from malt.models.prompts import (
    build_generator_prompt,
    build_verifier_prompt,
    build_refiner_prompt,
)


@dataclass
class TreeSearchConfig:
    """
    Configuration for G→V→R tree search.
    """

    branching_factor: int = 3

    max_new_tokens_generator: int = 256
    max_new_tokens_verifier: int = 256
    max_new_tokens_refiner: int = 256

    temperature_generator: float = 0.7
    temperature_verifier: float = 0.5
    temperature_refiner: float = 0.5

    top_p: float = 0.95
    top_k: int = 50

    # Output path for JSONL trajectories (each line is a single question's tree).
    output_path: Path = Path("data/malt_gsm8k_trajectories.jsonl")


@dataclass
class RefinerNode:
    text: str


@dataclass
class VerifierNode:
    text: str
    refiners: List[RefinerNode] = field(default_factory=list)


@dataclass
class GeneratorNode:
    text: str
    verifiers: List[VerifierNode] = field(default_factory=list)


def _sample_texts(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[str]:
    """
    Sample one completion per prompt using the current active adapter.

    For simplicity and to avoid unexpected VRAM spikes on a 24GB GPU, this
    helper currently generates one prompt at a time. It can be batched and
    optimized later if needed.
    """
    model.eval()
    outputs: List[str] = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Decode only the newly generated tokens.
            gen_text = tokenizer.decode(
                generated_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            outputs.append(gen_text.strip())

    return outputs


def run_tree_search_for_questions(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: Sequence[Gsm8kExample],
    cfg: TreeSearchConfig,
) -> Iterable[dict]:
    """
    Run G→V→R tree search for a list of questions.

    Yields one Python dict per question containing the full trajectory tree:

        {
          "id": ...,
          "question": ...,
          "answer_gt": ...,
          "generator_nodes": [
            {
              "text": ...,
              "verifier_nodes": [
                {
                  "text": ...,
                  "refiner_nodes": [{"text": ...}, ...]
                },
                ...
              ],
            },
            ...
          ],
        }
    """
    n = cfg.branching_factor

    for ex in questions:
        # 1) Generator stage
        print(f"Processing question id {ex.id}/{len(questions)}")

        set_active_role_adapter(model, ROLE_GENERATOR)
        gen_prompts = [build_generator_prompt(ex.question) for _ in range(n)]
        gen_texts = _sample_texts(
            model=model,
            tokenizer=tokenizer,
            prompts=gen_prompts,
            max_new_tokens=cfg.max_new_tokens_generator,
            temperature=cfg.temperature_generator,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
        )

        generator_nodes: List[GeneratorNode] = []

        # For each generator output, sample n verifier outputs.
        for g_text in gen_texts:
            set_active_role_adapter(model, ROLE_VERIFIER)
            v_prompts = [
                build_verifier_prompt(ex.question, g_text) for _ in range(n)
            ]
            v_texts = _sample_texts(
                model=model,
                tokenizer=tokenizer,
                prompts=v_prompts,
                max_new_tokens=cfg.max_new_tokens_verifier,
                temperature=cfg.temperature_verifier,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
            )

            verifier_nodes: List[VerifierNode] = []

            # For each verifier output, sample n refiner outputs.
            for v_text in v_texts:
                set_active_role_adapter(model, ROLE_REFINER)
                r_prompts = [
                    build_refiner_prompt(ex.question, g_text, v_text)
                    for _ in range(n)
                ]
                r_texts = _sample_texts(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=r_prompts,
                    max_new_tokens=cfg.max_new_tokens_refiner,
                    temperature=cfg.temperature_refiner,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                )

                refiner_nodes = [RefinerNode(text=t) for t in r_texts]
                verifier_nodes.append(VerifierNode(text=v_text, refiners=refiner_nodes))

            generator_nodes.append(GeneratorNode(text=g_text, verifiers=verifier_nodes))

        trajectory = {
            "id": ex.id,
            "question": ex.question,
            "answer_gt": ex.answer_target,
            "generator_nodes": [
                {
                    "text": g.text,
                    "verifier_nodes": [
                        {
                            "text": v.text,
                            "refiner_nodes": [
                                {"text": r.text}
                                for r in v.refiners
                            ],
                        }
                        for v in g.verifiers
                    ],
                }
                for g in generator_nodes
            ],
        }

        yield trajectory


def _get_last_processed_id(path: Path) -> int | None:
    """
    If a trajectory file already exists, read the last line and return
    the question id that was last processed.
    """
    if not path.exists():
        return None

    last_line = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line

    if last_line is None:
        return None

    try:
        obj = json.loads(last_line)
        return obj.get("id")
    except Exception:
        return None
    

def run_tree_search_for_gsm8k_split(
    split: Literal["train", "test", "validation"],
    cfg: TreeSearchConfig,
    model_cfg: MaltModelConfig | None = None,
) -> Path:
    """
    Convenience function: load a GSM8K split and run tree search over it,
    writing one JSON line per question to `cfg.output_path`.
    """
    from malt.data import load_gsm8k_split  # imported lazily to avoid cycles

    # Ensure output directory exists.
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    print("output path exists/created")

    last_id = _get_last_processed_id(cfg.output_path)

    examples = load_gsm8k_split(split)
    print("loaded gsm8k split")

    if last_id is not None:
        print(f"Resuming after id {last_id}")
        examples = [ex for ex in examples if str(ex.id) > str(last_id)]

    model, tokenizer = load_malt_llama_with_adapters(model_cfg)
    print("loaded llama with adapters")

    mode = "a" if cfg.output_path.exists() else "w"

    with cfg.output_path.open(mode, encoding="utf-8") as f:
        for traj in run_tree_search_for_questions(
            model=model,
            tokenizer=tokenizer,
            questions=examples,
            cfg=cfg,
        ):
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    return cfg.output_path

