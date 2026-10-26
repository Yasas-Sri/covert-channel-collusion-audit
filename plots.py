"""Report figures and the two statistics the findings section cites.

    .venv/bin/python plots.py            # writes results/fig_capacity_harm.png, fig_auditor_auroc.png
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

import auditor

auditor.HARM_CI = 0.70  # pre-registered from the block-rule simulation
INK, INK2, BLUE, SURFACE = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 300})

eps = auditor.load("results")
known = {k: v for k, v in eps.items() if v[0]["arm"] == "known"}
qs = sorted({m["q"] for m, _ in known.values()})
ci = {q: [auditor.final_ci(rows) for m, rows in known.values() if m["q"] == q] for q in qs}

# Paired dose-response test: one OLS slope of final CI on q per seed (seeds share channel draws across q).
by_seed = {}
for m, rows in known.values():
    by_seed.setdefault(m["seed"], {})[m["q"]] = auditor.final_ci(rows)
slopes = [np.polyfit(qs, [d[q] for q in qs], 1)[0] for d in by_seed.values()]
print(f"per-seed slope of final CI on q: mean {np.mean(slopes):+.3f}, {sum(s > 0 for s in slopes)}/{len(slopes)} "
      f"positive, Wilcoxon p = {wilcoxon(slopes).pvalue:.4f}")

# Paired AUROC difference A_trace - A_market on the same episodes and bootstrap draws.
(Xm, y, meta), (Xt, _, _) = auditor.dataset(eps, "market"), auditor.dataset(eps, "trace")
sm, st = auditor.oof_scores(Xm, y, meta), auditor.oof_scores(Xt, y, meta)
rng, pos, neg = np.random.default_rng(0), np.flatnonzero(y == 1), np.flatnonzero(y == 0)
diff = []
for _ in range(auditor.N_BOOT):
    i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
    diff.append(roc_auc_score(y[i], st[i]) - roc_auc_score(y[i], sm[i]))
lo, hi = np.percentile(diff, [2.5, 97.5])
print(f"AUROC(A_trace) - AUROC(A_market) = {roc_auc_score(y, st) - roc_auc_score(y, sm):+.3f}, "
      f"95% CI [{lo:+.3f}, {hi:+.3f}]")

# Figure 1: capacity vs harm, episodes as dots, mean +- SE.
fig, ax = plt.subplots(figsize=(4.5, 3.2))
jit = np.random.default_rng(1)
for q in qs:
    ax.scatter(q + jit.uniform(-0.03, 0.03, len(ci[q])), ci[q], s=14, color=BLUE, alpha=0.3, lw=0)
means = [np.mean(ci[q]) for q in qs]
se = [np.std(ci[q], ddof=1) / np.sqrt(len(ci[q])) for q in qs]
ax.errorbar(qs, means, yerr=se, color=BLUE, lw=2, marker="o", ms=6, capsize=4, label="mean ± SE")
ax.axhline(auditor.HARM_CI, color=INK2, ls="--", lw=1, label=f"harm threshold (CI = {auditor.HARM_CI:.2f})")
ax.legend(fontsize=7, loc="lower right", frameon=False)
ax.set(xlabel="channel capacity q (share of active rounds)", ylabel="final-window collusion index",
       xticks=qs, ylim=(0.3, 1.0))
ax.set_title(f"Capacity vs harm, known arm (n = {len(ci[qs[0]])} seeds per q)", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig("results/fig_capacity_harm.png")

# Figure 2: auditor AUROC with bootstrap 95% CIs against the chance band.
rows = list(csv.DictReader(open("results/auditor_summary.csv")))
auc = np.array([float(r["auroc"]) for r in rows])
err = np.array([[auc[i] - float(r["ci_lo"]), float(r["ci_hi"]) - auc[i]] for i, r in enumerate(rows)]).T
fig, ax = plt.subplots(figsize=(3.6, 3.2))
ax.axhspan(0.45, 0.55, color=INK2, alpha=0.12, lw=0, label="equivalence band [0.45, 0.55]")
ax.axhline(0.5, color=INK2, ls="--", lw=1, label="chance (0.5)")
ax.bar([r["auditor"] for r in rows], auc, width=0.5, color=BLUE, yerr=err, capsize=5,
       error_kw={"ecolor": INK, "lw": 1.2})
for i, a in enumerate(auc):
    ax.text(i, 0.04, f"{a:.2f}", ha="center", color="white", fontsize=9)
ax.set(ylabel="AUROC (leave-one-seed-out)", ylim=(0, 1))
ax.set_title("Auditor detection of harmful episodes", fontsize=10, loc="left")
ax.legend(fontsize=7, loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig("results/fig_auditor_auroc.png")
print("-> results/fig_capacity_harm.png, results/fig_auditor_auroc.png")
