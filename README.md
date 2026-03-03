## MALT: Multi-Agent LLM Training (LoRA, 24GB GPU)

This repository implements a MALT-style multi-agent reasoning pipeline with three
LoRA-adapted copies of a base Llama-3.1-8B-Instruct model (Generator, Verifier,
Refiner), plus Qwen 2.5 / Qwen 3 base-model baselines.

The code targets a **single 24GB CUDA GPU** by combining:

- 4-bit quantization (QLoRA-style) for the 8B base model.
- LoRA adapters for role-specific fine-tuning.
- Small effective batch sizes with gradient accumulation.

The training and evaluation pipeline follows the **MALT paper**:

- Generate–Verify–Refine three-agent reasoning.
- Tree-based search over reasoning trajectories with branching factor `n`.
- Value-iteration-based credit assignment from refinement leaves back to generator / verifier nodes.
- SFT + DPO post-training (Verifier, Refiner) and SFT-only for the Generator.

Once fully implemented, you will be able to:

- Train three role-specific LoRA adapters (`G`, `V`, `R`) on GSM8K and MATH.
- Run multi-agent inference with majority voting.
- Compare against:
  - A single-agent Llama-3.1-8B baseline.
  - Qwen 2.5 base and Qwen 3 base zero-shot baselines (no fine-tuning, no agents).

---

### 1. Environment setup

1. Create and activate a Python environment (Python 3.10+ recommended):

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows PowerShell
# or
source .venv/bin/activate  # Linux/macOS
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

You need:

- A **CUDA-enabled GPU** with **≈24GB VRAM**.
- A working CUDA toolkit and drivers compatible with your `torch` / `bitsandbytes` build.

> **Note (bitsandbytes on Windows)**  
> Native `bitsandbytes` wheels for Windows are still evolving. If installation fails,
> check the official documentation or community builds for your CUDA + Windows version.
> Alternatively, consider running the code inside WSL2 or a Linux environment, where
> `bitsandbytes` has more mature support.

---

### 2. Project layout (high level)

The codebase will be organized as:

- `config/`
  - `malt_gsm8k.yaml` – hyperparameters and paths for GSM8K experiments.
  - `malt_math.yaml` – hyperparameters and paths for MATH experiments.
- `malt/`
  - `models/` – base model loader + LoRA adapters, and role prompts.
  - `data/` – dataset loaders for GSM8K and MATH, plus evaluation helpers.
  - `search/` – tree search and value-iteration credit assignment.
  - `training/` – SFT and DPO training loops.
  - `inference/` – multi-agent pipeline and Qwen baselines.
  - `utils/` – shared utilities for I/O and evaluation.
- `scripts/`
  - Entry points to:
    - Generate reasoning trees and labeled datasets.
    - Train role adapters.
    - Run evaluation and produce comparison tables.

The concrete modules and scripts will be implemented in subsequent steps of the plan.

---

### 3. GSM8K workflow (end-to-end)

The typical end-to-end pipeline for GSM8K is:

1. **Generate trajectories with tree search**

   ```bash
   python -m malt.search.tree_search  # or write a small script that calls
                                      # run_tree_search_for_gsm8k_split(...)
   ```

   Programmatically:

   ```python
   from pathlib import Path
   from malt.search import TreeSearchConfig, run_tree_search_for_gsm8k_split

   cfg = TreeSearchConfig(branching_factor=3, output_path=Path("data/gsm8k_trajectories.jsonl"))
   run_tree_search_for_gsm8k_split("train", cfg)
   ```

2. **Apply value iteration and build labeled data**

   ```python
   from pathlib import Path
   from malt.search.value_iteration import ValueIterationConfig, value_iteration_over_jsonl

   input_path = Path("data/gsm8k_trajectories.jsonl")
   value_iteration_over_jsonl(input_path, cfg=ValueIterationConfig(task="gsm8k"))
   # Produces data/gsm8k_trajectories.valued.jsonl
   ```

3. **Train LoRA adapters**

   - **Generator SFT**:

     ```python
     from pathlib import Path
     from malt.training.sft_trainer import SftTrainingConfig, train_generator_sft

     sft_cfg = SftTrainingConfig(output_dir=Path("checkpoints/generator_sft"), num_train_epochs=1)
     train_generator_sft(Path("data/gsm8k_trajectories.valued.jsonl"), sft_cfg)
     ```

   - **Verifier + Refiner SFT + DPO** (high level):

     ```python
     from pathlib import Path
     from malt.training.sft_trainer import SftTrainingConfig, train_verifier_sft, train_refiner_sft
     from malt.training.dpo_trainer import DpoTrainingConfig, train_verifier_dpo, train_refiner_dpo

     valued_path = Path("data/gsm8k_trajectories.valued.jsonl")

     v_sft_cfg = SftTrainingConfig(output_dir=Path("checkpoints/verifier_sft"), num_train_epochs=1)
     r_sft_cfg = SftTrainingConfig(output_dir=Path("checkpoints/refiner_sft"), num_train_epochs=1)

     train_verifier_sft(valued_path, v_sft_cfg)
     train_refiner_sft(valued_path, r_sft_cfg)

     v_dpo_cfg = DpoTrainingConfig(output_dir=Path("checkpoints/verifier_dpo"))
     r_dpo_cfg = DpoTrainingConfig(output_dir=Path("checkpoints/refiner_dpo"))

     train_verifier_dpo(valued_path, v_dpo_cfg)
     train_refiner_dpo(valued_path, r_dpo_cfg)
     ```

4. **Evaluate on GSM8K**

   A convenience script is provided:

   ```bash
   python scripts/eval_malt.py --split test --num-samples 3
   ```

   This reports:

   - Qwen 2.5 base zero-shot baseline.
   - Qwen 3 base zero-shot baseline.
   - Single-agent Llama baseline (Generator only).
   - Multi-agent MALT-style baseline (G→V→R).

### 4. MATH workflow (high level)

Support for the MATH dataset mirrors the GSM8K pipeline:

- `malt.data.math` provides:
  - `load_math_split(...)` to load train/test splits.
  - `extract_math_answer` / `normalize_math_answer` / `math_exact_match` for evaluation.
- `malt.search.value_iteration` understands `task="math"` for value propagation.
- The same training and inference utilities can be reused with MATH-specific
  data and answer normalization.

You can adapt the GSM8K examples above by:

- Swapping `load_gsm8k_split` for `load_math_split`.
- Using `ValueIterationConfig(task="math")`.
- Evaluating with `evaluate_math_predictions` from `malt.utils.eval`.

