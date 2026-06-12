# Token Efficiency in Math Reasoning

Measuring the minimum "thinking" token budget at which DeepSeek-R1-Distill-Qwen-7B
reaches 95% of its uncapped accuracy, across three math benchmarks of increasing
difficulty: GSM8K, MATH-500, and AIME (1983-2024).

## Setup

- **Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (fp16)
- **Hardware:** RunPod, RTX A5000 (24GB)
- **Samples:** 200 per benchmark, fixed seed (42)
- **Budgets tested:** 256, 512, 1024, 2048, 4096, uncapped (max 10,000 tokens)
- **Budget forcing:** a custom logits processor (`BudgetForcingProcessor`) forces
  the `</think>` token once the thinking-token count reaches the budget, rather
  than hard-capping `max_new_tokens` (which would cut the model off mid-thought).

## Files

- `eval_token_efficiency.py` — main evaluation script. Runs all benchmarks ×
  budgets, checkpoints every 10 samples to `results/`, computes B* (minimum
  budget reaching 95% of uncapped accuracy) with bootstrap 95% CIs.
- `make_figures.py` — generates all figures from `results/` into `figures/`.
- `results/` — per-benchmark, per-budget JSON checkpoints + `final_summary.json`.
- `figures/` — generated plots (see below).

## Usage

```bash
pip install transformers accelerate datasets torch matplotlib scipy --break-system-packages -q

# Quick validation (20 samples)
python eval_token_efficiency.py --validate

# Full run (200 samples, all benchmarks, resumes from checkpoints)
python eval_token_efficiency.py --benchmark all

# Generate figures
python make_figures.py
```

## Results Summary

| Benchmark | B* | 95% CI | Uncapped Accuracy |
|---|---|---|---|
| GSM8K | 256 | [256, 512] | 0.740 |
| MATH-500 | 256 | [256, 4096] | 0.625 |
| AIME | None (no convergence) | — | 0.595* |

\* AIME uncapped accuracy is a **lower bound** — 41.5% of samples (87/200) were
truncated at the 10,000-token generation limit without emitting `</think>`.

## Key Findings

**1. Plateau effect (GSM8K, MATH-500).** Accuracy is statistically flat across
all tested budgets, including uncapped. Both benchmarks reach 95% of their
uncapped accuracy already at budget=256 — additional thinking budget does not
meaningfully improve accuracy.

**2. AIME does not converge to a single B\*.** Accuracy at budgets 256–4096
hovers between 0.46–0.52, but jumps to 0.595 uncapped. No finite tested budget
reaches 95% of the uncapped accuracy.

**3. AIME splits into a bimodal convergence distribution.**
- **113/200 samples (56.5%) converge naturally**, emitting `</think>` at an
  average of ~4,100 tokens, with **96.5% accuracy** [Wilson 95% CI: 91.3–98.6%].
- **87/200 samples (43.5%) never emit `</think>`** even at the 10,000-token
  cap, with **11.5% accuracy** [Wilson 95% CI: 6.4–19.9%].

**4. Convergence predicts accuracy beyond difficulty alone.** Non-convergence
correlates with AIME problem number (point-biserial r=0.431, p<0.0001,
r²=0.186), but the converged/non-converged accuracy gap holds — and widens —
within every difficulty tercile, including the hardest (100% vs 0%, n=18 vs
n=41). Problem number is itself an imperfect difficulty proxy (assigned by
competition organizers based on aggregate human solve rates, not per-model
difficulty), so convergence carries information beyond what problem number
alone provides.

## Figures

- **Figure 1** — Accuracy vs. thinking token budget, all three benchmarks
  (with bootstrap 95% CI bands). Shows the plateau for GSM8K/MATH-500 and
  AIME's distinct uncapped jump.
- **Figure 2** — AIME: per-sample scatter of total tokens generated vs.
  problem number, colored by correctness and shaped by convergence status.
- **Figure 3** — AIME: histogram of total generation length, showing the
  bimodal converged (~2k–9k tokens) vs. non-converged (pinned at 10k) split.
- **Figure 4** — AIME: accuracy by convergence status, broken down by
  difficulty tercile (Easy 1-5 / Medium 6-10 / Hard 11-15). The centerpiece
  figure — shows convergence predicts accuracy even within the Hard tercile.
- **Figure 5** — AIME: overall accuracy by convergence status (96.5% vs
  11.5%), the headline summary bar.

## Limitations

- AIME's "uncapped" run used a practical 10,000-token cap, not a true
  unlimited budget. The 41.5% non-convergence rate may partly reflect this
  cap rather than genuine infinite-loop behavior — we did not capture raw
  reasoning traces to distinguish "looping" from "slow but coherent progress"
  for non-converged samples.
- All results are for a single model (DeepSeek-R1-Distill-Qwen-7B). Whether
  the plateau effect and convergence bimodality generalize to other model
  sizes or families is untested.
- Sample size is 200 per benchmark; AIME sub-group breakdowns (e.g., per
  problem-number, per-tercile-by-convergence) have small n in some cells
  (as low as n=4).
