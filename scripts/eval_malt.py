import argparse
from pathlib import Path

from malt.data import load_gsm8k_split
from malt.inference.pipeline import (
    InferenceConfig,
    run_single_agent_generator_gsm8k,
    run_multi_agent_malt_gsm8k,
)
from malt.inference.qwen_baseline import (
    QwenBaselineConfig,
    load_qwen_model_and_tokenizer,
    run_qwen_zero_shot_gsm8k,
)
from malt.models import (
    MaltModelConfig,
    load_malt_llama_with_adapters,
    load_malt_llama_with_trained_adapters,
    set_active_role_adapter,
    ROLE_GENERATOR,
    ROLE_VERIFIER,
    ROLE_REFINER,
)
from malt.utils.eval import evaluate_gsm8k_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MALT and Qwen baselines on GSM8K.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "validation"])
    parser.add_argument("--num-samples", type=int, default=3, help="Number of trajectories per question for MV@k.")
    parser.add_argument("--qwen2-model", type=str, default="Qwen/Qwen2.5-1.5B", help="Qwen 2.5 base model name.")
    parser.add_argument("--qwen3-model", type=str, default="Qwen/Qwen-3-1.8B", help="Qwen 3 base model name.")
    parser.add_argument(
        "--gen-checkpoint",
        type=str,
        default=None,
        help="Optional path to Generator adapter checkpoint (PEFT `save_pretrained` directory).",
    )
    parser.add_argument(
        "--ver-checkpoint",
        type=str,
        default=None,
        help="Optional path to Verifier adapter checkpoint (PEFT `save_pretrained` directory).",
    )
    parser.add_argument(
        "--ref-checkpoint",
        type=str,
        default=None,
        help="Optional path to Refiner adapter checkpoint (PEFT `save_pretrained` directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load data
    examples = load_gsm8k_split(args.split)
    gt_answers = [ex.answer_target for ex in examples]

    # Inference configuration
    inf_cfg = InferenceConfig(num_samples=args.num_samples)

    # Qwen 2.5 baseline
    qwen2_cfg = QwenBaselineConfig(model_name=args.qwen2_model, num_samples=1)
    qwen2_model, qwen2_tok = load_qwen_model_and_tokenizer(qwen2_cfg)
    qwen2_preds = run_qwen_zero_shot_gsm8k(qwen2_model, qwen2_tok, examples, qwen2_cfg)
    qwen2_stats = evaluate_gsm8k_predictions(qwen2_preds, gt_answers)

    # Qwen 3 baseline
    qwen3_cfg = QwenBaselineConfig(model_name=args.qwen3_model, num_samples=1)
    qwen3_model, qwen3_tok = load_qwen_model_and_tokenizer(qwen3_cfg)
    qwen3_preds = run_qwen_zero_shot_gsm8k(qwen3_model, qwen3_tok, examples, qwen3_cfg)
    qwen3_stats = evaluate_gsm8k_predictions(qwen3_preds, gt_answers)

    # Llama model + adapters: either base-only, or with trained adapters loaded.
    llama_cfg = MaltModelConfig()
    if args.gen_checkpoint or args.ver_checkpoint or args.ref_checkpoint:
        llama_model, llama_tok = load_malt_llama_with_trained_adapters(
            cfg=llama_cfg,
            generator_checkpoint=args.gen_checkpoint,
            verifier_checkpoint=args.ver_checkpoint,
            refiner_checkpoint=args.ref_checkpoint,
        )
    else:
        llama_model, llama_tok = load_malt_llama_with_adapters(llama_cfg)

    # Single-agent Llama (generator only) uses the current generator adapter
    # (either base or fine-tuned, depending on the checkpoints provided).
    set_active_role_adapter(llama_model, ROLE_GENERATOR)
    single_preds = run_single_agent_generator_gsm8k(llama_model, llama_tok, examples, inf_cfg)
    single_stats = evaluate_gsm8k_predictions(single_preds, gt_answers)

    # Multi-agent MALT baseline using the same underlying adapters.
    # In a full setup, you would load separately fine-tuned checkpoints
    # for G, V, and R; here we reuse the same model for simplicity.
    set_active_role_adapter(llama_model, ROLE_GENERATOR)
    gen_model = llama_model
    set_active_role_adapter(llama_model, ROLE_VERIFIER)
    ver_model = llama_model
    set_active_role_adapter(llama_model, ROLE_REFINER)
    ref_model = llama_model

    malt_preds = run_multi_agent_malt_gsm8k(gen_model, ver_model, ref_model, llama_tok, examples, inf_cfg)
    malt_stats = evaluate_gsm8k_predictions(malt_preds, gt_answers)

    print("GSM8K Evaluation Results")
    print("------------------------")
    print(f"Qwen 2.5 base ({args.qwen2_model}):  {qwen2_stats.correct}/{qwen2_stats.total}  acc={qwen2_stats.accuracy:.4f}")
    print(f"Qwen 3 base ({args.qwen3_model}):    {qwen3_stats.correct}/{qwen3_stats.total}  acc={qwen3_stats.accuracy:.4f}")
    print(f"Llama single-agent (G only):         {single_stats.correct}/{single_stats.total}  acc={single_stats.accuracy:.4f}")
    print(f"MALT multi-agent (G→V→R):            {malt_stats.correct}/{malt_stats.total}  acc={malt_stats.accuracy:.4f}")


if __name__ == "__main__":
    main()

