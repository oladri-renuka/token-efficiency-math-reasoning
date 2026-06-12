"""
Figure generation for Token Efficiency in Math Reasoning paper.
Run this on the RunPod (or anywhere with the results/ directory and matplotlib).

Usage:
    pip install matplotlib scipy --break-system-packages -q
    python3 make_figures.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from scipy.stats import pointbiserialr

RESULTS_DIR = Path("results")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

BUDGETS = [256, 512, 1024, 2048, 4096]
BUDGET_LABELS = ['256', '512', '1024', '2048', '4096', 'Uncapped']


# ── Helpers ────────────────────────────────────────────────────────────────
def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (0, 0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, center - spread), min(1, center + spread)

def bootstrap_acc_ci(correct_flags, n_resamples=1000, seed=42):
    """Bootstrap CI on accuracy."""
    rng = np.random.default_rng(seed)
    correct_flags = np.array(correct_flags)
    n = len(correct_flags)
    boot_accs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_accs.append(correct_flags[idx].mean())
    return np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5)

def load_benchmark_results(name):
    """Load all budget checkpoints for a benchmark, return dict {budget: list of records}."""
    out = {}
    for b in BUDGETS:
        f = RESULTS_DIR / f"{name}_{b}.json"
        out[b] = json.load(open(f))
    f = RESULTS_DIR / f"{name}_uncapped.json"
    out[None] = json.load(open(f))
    return out

def accuracy_with_ci(records, budget_key):
    correct = [r[budget_key] for r in records]
    acc = np.mean(correct)
    lo, hi = bootstrap_acc_ci(correct)
    return acc, lo, hi


# ── Figure 1 — Combined accuracy vs budget, all 3 benchmarks ────────────────
def make_figure1():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    colors = {"gsm8k": "#4C72B0", "math500": "#DD8452", "aime": "#C44E52"}
    labels = {"gsm8k": "GSM8K", "math500": "MATH-500", "aime": "AIME 1983-2024"}

    x = np.arange(len(BUDGET_LABELS))

    for name in ["gsm8k", "math500", "aime"]:
        data = load_benchmark_results(name)
        accs, los, his = [], [], []
        for b in BUDGETS:
            key = str(b)
            acc, lo, hi = accuracy_with_ci(data[b], key)
            accs.append(acc); los.append(lo); his.append(hi)
        # uncapped
        acc, lo, hi = accuracy_with_ci(data[None], "None")
        accs.append(acc); los.append(lo); his.append(hi)

        ax.plot(x, accs, marker='o', label=labels[name], color=colors[name], linewidth=2, markersize=6)
        ax.fill_between(x, los, his, color=colors[name], alpha=0.15)

        # mark AIME uncapped as lower bound -- footnote, not arrow
        if name == "aime":
            ax.text(0.01, 0.04,
                    "* AIME 'Uncapped' accuracy is a lower bound: 41.5% of samples\n"
                    "  were truncated at the 10,000-token generation limit.",
                    transform=ax.transAxes, fontsize=8, color="#C44E52",
                    va='bottom', ha='left', style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(BUDGET_LABELS, rotation=30)
    ax.set_xlabel("Thinking token budget")
    ax.set_ylabel("Accuracy (conservative, n=200, shaded = bootstrap 95% CI)")
    ax.set_title("Accuracy vs. Thinking Token Budget\nDeepSeek-R1-Distill-Qwen-7B", fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_ylim(0.3, 1.0)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_accuracy_vs_budget.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig1_accuracy_vs_budget.png")


# ── Figure 2 — AIME scatter: problem number vs thinking tokens, colored by correctness ──
def make_figure2():
    data = json.load(open(RESULTS_DIR / "aime_uncapped.json"))

    # Need problem numbers - load dataset
    from datasets import load_dataset
    import random
    SEED = 42
    random.seed(SEED)
    ds = load_dataset("gneubig/aime-1983-2024", split="train")
    ds = ds.shuffle(seed=SEED).select(range(200))
    problem_nums = [ds[i]["Problem Number"] for i in range(200)]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for r in data:
        idx = r["idx"]
        pn = problem_nums[idx]
        tokens = r["total_tokens"]
        correct = r["None"]
        converged = not r["hit_max_tokens"]

        if converged:
            marker = 'o'
            color = '#55A868' if correct else '#C44E52'
        else:
            marker = '^'
            color = '#55A868' if correct else '#C44E52'

        # jitter x slightly for visibility
        jitter = (np.random.default_rng(idx).random() - 0.5) * 0.3
        ax.scatter(pn + jitter, tokens, marker=marker, color=color, alpha=0.7, s=45,
                   edgecolors='black', linewidths=0.4)

    # Legend (manual)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#55A868', markeredgecolor='black', markersize=9, label='Converged, Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#C44E52', markeredgecolor='black', markersize=9, label='Converged, Incorrect'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#55A868', markeredgecolor='black', markersize=9, label='Non-converged (hit 10k cap), Correct'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#C44E52', markeredgecolor='black', markersize=9, label='Non-converged (hit 10k cap), Incorrect'),
    ]
    ax.legend(handles=legend_elements, loc='center right', fontsize=8)

    ax.axhline(10000, linestyle='--', color='gray', linewidth=1, alpha=0.6)
    ax.set_xlabel("AIME Problem Number (official competition order, 1-15)")
    ax.set_ylabel("Total tokens generated (uncapped, max 10,000)")
    ax.set_title("AIME: Convergence and Correctness vs. Problem Number\n(DeepSeek-R1-Distill-Qwen-7B, uncapped run, n=200)", fontweight='bold')
    ax.set_xticks(range(1, 16))
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_aime_convergence_scatter.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig2_aime_convergence_scatter.png")


# ── Figure 3 — Distribution of token counts, converged vs non-converged ────
def make_figure3():
    data = json.load(open(RESULTS_DIR / "aime_uncapped.json"))

    converged_tokens = [r["total_tokens"] for r in data if not r["hit_max_tokens"]]
    nonconverged_tokens = [r["total_tokens"] for r in data if r["hit_max_tokens"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bins = np.linspace(0, 10500, 40)
    ax.hist(converged_tokens, bins=bins, alpha=0.7, label=f"Converged (n={len(converged_tokens)})", color="#55A868")
    ax.hist(nonconverged_tokens, bins=bins, alpha=0.7, label=f"Non-converged, hit 10k cap (n={len(nonconverged_tokens)})", color="#C44E52")

    ax.set_xlabel("Total tokens generated")
    ax.set_ylabel("Count")
    ax.set_title("AIME: Bimodal Distribution of Generation Length\n(uncapped run, n=200)", fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_aime_token_distribution.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig3_aime_token_distribution.png")


# ── Figure 4 — Accuracy by convergence x difficulty tercile ─────────────────
def make_figure4():
    data = json.load(open(RESULTS_DIR / "aime_uncapped.json"))

    from datasets import load_dataset
    import random
    SEED = 42
    random.seed(SEED)
    ds = load_dataset("gneubig/aime-1983-2024", split="train")
    ds = ds.shuffle(seed=SEED).select(range(200))
    problem_nums = [ds[i]["Problem Number"] for i in range(200)]

    def tercile(pn):
        if pn <= 5:
            return "Easy (1-5)"
        elif pn <= 10:
            return "Medium (6-10)"
        else:
            return "Hard (11-15)"

    groups = {}  # (tercile, converged) -> list of correct flags
    for r in data:
        idx = r["idx"]
        t = tercile(problem_nums[idx])
        converged = not r["hit_max_tokens"]
        key = (t, converged)
        groups.setdefault(key, []).append(r["None"])

    tercile_order = ["Easy (1-5)", "Medium (6-10)", "Hard (11-15)"]
    x = np.arange(len(tercile_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for offset, converged, label, color in [(-width/2, True, "Converged", "#55A868"),
                                              (width/2, False, "Non-converged", "#C44E52")]:
        accs, errs_lo, errs_hi, ns = [], [], [], []
        for t in tercile_order:
            flags = groups.get((t, converged), [])
            n = len(flags)
            ns.append(n)
            if n == 0:
                accs.append(0); errs_lo.append(0); errs_hi.append(0)
                continue
            acc = np.mean(flags)
            lo, hi = wilson_ci(acc, n)
            accs.append(acc)
            errs_lo.append(acc - lo)
            errs_hi.append(hi - acc)

        bars = ax.bar(x + offset, accs, width, label=label, color=color,
                       yerr=[errs_lo, errs_hi], capsize=4, zorder=3)

        for bar, n, acc, hi_err in zip(bars, ns, accs, errs_hi):
            top = max(acc + hi_err, 0.03)
            ax.text(bar.get_x() + bar.get_width()/2, top + 0.03,
                    f'n={n}\nacc={acc:.0%}', ha='center', fontsize=8, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(tercile_order)
    ax.set_ylabel("Accuracy (Wilson 95% CI)")
    ax.set_ylim(0, 1.25)
    ax.set_title("AIME: Accuracy by Convergence Status, within Difficulty Tercile\n(uncapped run, n=200)", fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_convergence_by_tercile.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig4_convergence_by_tercile.png")

    # Print the numbers so we can decide whether this figure helps or hurts the argument
    print("\n--- Tercile breakdown (for review) ---")
    for t in tercile_order:
        for converged, label in [(True, "Converged"), (False, "Non-converged")]:
            flags = groups.get((t, converged), [])
            if flags:
                print(f"{t} | {label}: n={len(flags)}, acc={np.mean(flags):.3f}")
            else:
                print(f"{t} | {label}: n=0")


# ── Figure 5 — Summary bar: overall converged vs non-converged accuracy ─────
def make_figure5():
    data = json.load(open(RESULTS_DIR / "aime_uncapped.json"))

    converged = [r["None"] for r in data if not r["hit_max_tokens"]]
    nonconverged = [r["None"] for r in data if r["hit_max_tokens"]]

    accs = [np.mean(converged), np.mean(nonconverged)]
    cis = [wilson_ci(accs[0], len(converged)), wilson_ci(accs[1], len(nonconverged))]
    errs = [[accs[i]-cis[i][0] for i in range(2)], [cis[i][1]-accs[i] for i in range(2)]]

    labels = [f'Converged\n(n={len(converged)})', f'Non-converged\n(n={len(nonconverged)})']

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(labels, accs, color=['#55A868', '#C44E52'], width=0.5, yerr=errs, capsize=6)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.04, f'{acc:.1%}', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel("Accuracy (uncapped run)")
    ax.set_ylim(0, 1.05)
    ax.set_title("AIME: Accuracy by Reasoning Convergence\n(uncapped, max 10,000 tokens)", fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_aime_bimodal_overall.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig5_aime_bimodal_overall.png")
    print(f"Converged acc CI: {cis[0]}")
    print(f"Non-converged acc CI: {cis[1]}")


# ── Correlation stat (for caption / text, not a figure) ─────────────────────
def compute_correlation():
    data = json.load(open(RESULTS_DIR / "aime_uncapped.json"))

    from datasets import load_dataset
    import random
    SEED = 42
    random.seed(SEED)
    ds = load_dataset("gneubig/aime-1983-2024", split="train")
    ds = ds.shuffle(seed=SEED).select(range(200))
    problem_nums = [ds[i]["Problem Number"] for i in range(200)]

    hit_cap = {r["idx"]: int(r["hit_max_tokens"]) for r in data}
    pn = np.array([problem_nums[i] for i in range(200)])
    flags = np.array([hit_cap[i] for i in range(200)])

    r, p = pointbiserialr(flags, pn)
    print(f"\nPoint-biserial correlation (non-convergence vs problem number): r={r:.3f}, p={p:.6f}")
    print(f"r^2 = {r**2:.3f} (variance explained)")


# ── Summary table ─────────────────────────────────────────────────────────
def print_summary_table():
    summary = json.load(open(RESULTS_DIR / "final_summary.json"))
    print("\n--- Summary Table ---")
    print(f"{'Benchmark':<12} {'B*':<10} {'95% CI':<12} {'Uncapped Acc':<14} Notes")
    for name, label in [("gsm8k", "GSM8K"), ("math500", "MATH-500"), ("aime", "AIME")]:
        b_star = summary[name]["b_star"]
        ci = summary[name]["bootstrap"]
        uncapped_acc = [r["accuracy_conservative"] for r in summary[name]["budget_results"] if r["budget"] is None][0]
        ci_str = f"[{ci['ci_low']}, {ci['ci_high']}]" if ci['ci_low'] is not None else "N/A"
        b_star_str = str(b_star) if b_star is not None else "None (no convergence)"
        print(f"{label:<12} {b_star_str:<10} {ci_str:<12} {uncapped_acc:<14.3f}")


if __name__ == "__main__":
    print("Generating figures...\n")
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    make_figure5()
    compute_correlation()
    print_summary_table()
    print("\nDone. Figures saved to figures/")
