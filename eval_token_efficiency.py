"""
Token Efficiency in Math Reasoning
Measures minimum thinking token budget (B*) at which DeepSeek-R1-Distill-7B
achieves 95% of uncapped accuracy, across GSM8K / MATH-500 / AIME.
"""

import re
import json
import random
import time
import argparse
import numpy as np
from pathlib import Path
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME        = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SEED              = 42
N_SAMPLES         = 200
BUDGETS           = [256, 512, 1024, 2048, 4096, None]   # None = uncapped
MAX_NEW_TOKENS    = 6000   # budgeted runs
MAX_NEW_UNCAPPED  = 10000  # uncapped runs
BOOTSTRAP_N       = 1000
RESULTS_DIR       = Path("results")
THINK_START_ID    = 151648
THINK_END_ID      = 151649

random.seed(SEED)
np.random.seed(SEED)


# ── Logits processor ──────────────────────────────────────────────────────────
class BudgetForcingProcessor(LogitsProcessor):
    def __init__(self, budget: int):
        self.budget = budget
        self.thinking = True   # FIX: prompt already ends with <think>, so we start in thinking mode
        self.think_token_count = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        last_token = input_ids[0, -1].item()

        if last_token == THINK_START_ID:
            # Model re-entered thinking (shouldn't happen normally, but handle it)
            self.thinking = True
            self.think_token_count = 0
        elif last_token == THINK_END_ID:
            self.thinking = False
        elif self.thinking:
            self.think_token_count += 1
            if self.think_token_count >= self.budget:
                forced = torch.full_like(scores, float("-inf"))
                forced[:, THINK_END_ID] = 0.0
                return forced

        return scores


# ── Dataset loaders ───────────────────────────────────────────────────────────
def load_gsm8k(n: int) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED).select(range(n))
    return [{"question": r["question"], "answer": r["answer"].split("####")[-1].strip()} for r in ds]

def load_math500(n: int) -> list[dict]:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ds = ds.shuffle(seed=SEED).select(range(n))
    return [{"question": r["problem"], "answer": r["answer"]} for r in ds]

def load_aime(n: int) -> list[dict]:
    ds = load_dataset("gneubig/aime-1983-2024", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))
    return [{"question": r["Question"], "answer": str(r["Answer"])} for r in ds]


# ── Answer extraction ─────────────────────────────────────────────────────────
def extract_answer(response: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    if boxed:
        return boxed[-1].strip()
    ans = re.findall(r"[Tt]he answer is\s*\$?([0-9\-\/\.\,]+)", response)
    if ans:
        return ans[-1].strip()
    post_think = response.split("</think>")[-1] if "</think>" in response else response
    numbers = re.findall(r"\$?([0-9]+(?:\.[0-9]+)?)", post_think)
    if numbers:
        return numbers[-1].strip()
    return None

def normalize_answer(ans: str | None) -> str | None:
    if ans is None:
        return None
    ans = ans.replace(",", "").replace("$", "").strip()
    try:
        return str(float(ans))
    except:
        return ans.lower().strip()


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, tokenizer, prompt: str, budget: int | None) -> dict:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    if budget is not None:
        processor = BudgetForcingProcessor(budget=budget)
        logits_processors = [processor]
        max_new_tokens = MAX_NEW_TOKENS
    else:
        logits_processors = []
        max_new_tokens = MAX_NEW_UNCAPPED

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            logits_processor=logits_processors,
        )

    generated = output[0][input_ids.shape[1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=False)

    think_end_positions = (generated == THINK_END_ID).nonzero()
    think_end_pos = think_end_positions[0].item() if len(think_end_positions) > 0 else None
    hit_budget = (
        think_end_pos is not None and
        budget is not None and
        think_end_pos >= budget - 1
    )

    return {
        "response": decoded,
        "total_tokens": len(generated),
        "think_end_pos": think_end_pos,
        "hit_budget": hit_budget,
        "hit_max_tokens": len(generated) >= max_new_tokens,
    }


# ── B* computation ────────────────────────────────────────────────────────────
def compute_b_star(accuracies: dict, threshold: float = 0.95) -> int | None:
    uncapped_acc = accuracies.get(None)
    if uncapped_acc is None or uncapped_acc == 0:
        return None
    target = threshold * uncapped_acc
    finite_budgets = sorted([b for b in accuracies if b is not None])
    for b in finite_budgets:
        if accuracies[b] >= target:
            remaining = [accuracies[bb] >= target for bb in finite_budgets if bb >= b]
            if all(remaining):
                return b
    return None

def bootstrap_b_star(all_per_sample: list[dict], budget_levels: list, n_resamples: int = 1000) -> dict:
    n = len(all_per_sample)
    b_star_samples = []

    for _ in range(n_resamples):
        indices = np.random.randint(0, n, size=n)
        resampled = [all_per_sample[i] for i in indices]
        accs = {}
        for b in budget_levels:
            key = "None" if b is None else str(b)
            vals = [r[key] for r in resampled if key in r]
            accs[b] = np.mean(vals) if vals else 0.0
        b_star_samples.append(compute_b_star(accs))

    valid = [b for b in b_star_samples if b is not None]
    if not valid:
        return {"b_star": None, "ci_low": None, "ci_high": None}

    # Point estimate on full data
    full_accs = {}
    for b in budget_levels:
        key = "None" if b is None else str(b)
        vals = [r[key] for r in all_per_sample if key in r]
        full_accs[b] = np.mean(vals) if vals else 0.0

    return {
        "b_star": compute_b_star(full_accs),
        "ci_low": int(np.percentile(valid, 2.5)),
        "ci_high": int(np.percentile(valid, 97.5)),
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def checkpoint_path(benchmark: str, budget: int | None) -> Path:
    budget_str = "uncapped" if budget is None else str(budget)
    return RESULTS_DIR / f"{benchmark}_{budget_str}.json"

def load_checkpoint(benchmark: str, budget: int | None) -> list[dict] | None:
    p = checkpoint_path(benchmark, budget)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

def save_checkpoint(benchmark: str, budget: int | None, data: list[dict]):
    with open(checkpoint_path(benchmark, budget), "w") as f:
        json.dump(data, f, indent=2)


# ── Eval loop ─────────────────────────────────────────────────────────────────
def eval_benchmark(model, tokenizer, name: str, samples: list[dict], budget: int | None) -> dict:
    budget_str = "uncapped" if budget is None else str(budget)
    print(f"\n── {name} | budget={budget_str} ──")

    existing = load_checkpoint(name, budget)
    if existing is not None and len(existing) == len(samples):
        print(f"  Loaded from checkpoint ({len(existing)} samples)")
        return summarize(existing, budget)

    results = existing or []
    start_idx = len(results)
    if start_idx > 0:
        print(f"  Resuming from sample {start_idx}")

    for i, sample in enumerate(samples[start_idx:], start=start_idx):
        t0 = time.time()
        result = run_inference(model, tokenizer, sample["question"], budget)
        extracted = extract_answer(result["response"])
        correct = normalize_answer(extracted) == normalize_answer(sample["answer"])

        budget_key = "None" if budget is None else str(budget)
        results.append({
            "idx": i,
            budget_key: correct,
            "extracted": extracted,
            "expected": sample["answer"],
            "total_tokens": result["total_tokens"],
            "think_end_pos": result["think_end_pos"],
            "hit_budget": result["hit_budget"],
            "hit_max_tokens": result["hit_max_tokens"],
            "extraction_success": extracted is not None,
            "elapsed": round(time.time() - t0, 1),
        })

        if (i + 1) % 10 == 0:
            save_checkpoint(name, budget, results)
            acc = np.mean([r[budget_key] for r in results])
            ext = np.mean([r["extraction_success"] for r in results])
            print(f"  [{i+1}/{len(samples)}] acc={acc:.3f} extraction={ext:.3f}")

    save_checkpoint(name, budget, results)
    return summarize(results, budget)

def summarize(results: list[dict], budget: int | None) -> dict:
    budget_key = "None" if budget is None else str(budget)
    correct = [r[budget_key] for r in results if budget_key in r]
    extracted = [r["extraction_success"] for r in results]
    acc_conservative = np.mean(correct) if correct else 0.0
    valid_correct = [r[budget_key] for r in results if budget_key in r and r["extraction_success"]]
    acc_valid = np.mean(valid_correct) if valid_correct else 0.0
    return {
        "budget": budget,
        "n": len(results),
        "accuracy_conservative": round(float(acc_conservative), 4),
        "accuracy_valid": round(float(acc_valid), 4),
        "extraction_rate": round(float(np.mean(extracted)), 4),
        "avg_tokens": round(float(np.mean([r["total_tokens"] for r in results])), 1),
        "pct_hit_budget": round(float(np.mean([r["hit_budget"] for r in results])), 4),
        "pct_hit_max_tokens": round(float(np.mean([r["hit_max_tokens"] for r in results])), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["gsm8k", "math500", "aime", "all"], default="all")
    parser.add_argument("--validate", action="store_true", help="Run 20-sample validation only")
    args = parser.parse_args()

    n = 20 if args.validate else N_SAMPLES
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    print(f"Model loaded | VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    # Load datasets
    benchmarks = {}
    if args.benchmark in ("gsm8k", "all"):
        print("Loading GSM8K...")
        benchmarks["gsm8k"] = load_gsm8k(n)
    if args.benchmark in ("math500", "all"):
        print("Loading MATH-500...")
        benchmarks["math500"] = load_math500(n)
    if args.benchmark in ("aime", "all"):
        print("Loading AIME...")
        benchmarks["aime"] = load_aime(n)

    # Run evals
    all_summary = {}
    for bench_name, samples in benchmarks.items():
        bench_results = []
        all_per_sample = []

        for budget in BUDGETS:
            summary = eval_benchmark(model, tokenizer, bench_name, samples, budget)
            bench_results.append(summary)
            budget_str = "uncapped" if budget is None else str(budget)
            print(f"  {bench_name} | budget={budget_str} | "
                  f"acc_conservative={summary['accuracy_conservative']:.3f} | "
                  f"acc_valid={summary['accuracy_valid']:.3f} | "
                  f"extraction={summary['extraction_rate']:.3f} | "
                  f"avg_tokens={summary['avg_tokens']:.0f}")

            if summary["extraction_rate"] < 0.95:
                print(f"  !! EXTRACTION GATE FAILED for {bench_name} budget={budget_str} "
                      f"({summary['extraction_rate']:.3f} < 0.95)")

        # Load all per-sample data for bootstrap
        for budget in BUDGETS:
            ckpt = load_checkpoint(bench_name, budget)
            if ckpt:
                budget_key = "None" if budget is None else str(budget)
                for r in ckpt:
                    idx = r["idx"]
                    while len(all_per_sample) <= idx:
                        all_per_sample.append({})
                    all_per_sample[idx][budget_key] = r[budget_key]

        bootstrap = bootstrap_b_star(all_per_sample, BUDGETS, BOOTSTRAP_N)

        all_summary[bench_name] = {
            "budget_results": bench_results,
            "b_star": bootstrap["b_star"],
            "bootstrap": bootstrap,
        }

        print(f"\n  {bench_name.upper()} B* = {bootstrap['b_star']} "
              f"[CI: {bootstrap['ci_low']} – {bootstrap['ci_high']}]")

    out_path = RESULTS_DIR / ("validation_summary.json" if args.validate else "final_summary.json")
    with open(out_path, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nSaved summary → {out_path}")


if __name__ == "__main__":
    main()
