from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from .utils import read_json


def plots(out, predictions, metrics, origin):
    out = Path(out)
    warning = "SYNTHETIC DEMO — NOT PAPER EVIDENCE" if origin == "synthetic_demo" else "Held-out test windows (see protocol)"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    y = predictions.label.to_numpy()
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, predictions.score)
        p, r, _ = precision_recall_curve(y, predictions.score)
        axes[0].plot(fpr, tpr); axes[0].plot([0, 1], [0, 1], "--", color="grey")
        axes[1].plot(r, p)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    fig.suptitle(warning, fontsize=10); fig.tight_layout()
    fig.savefig(out/"roc_pr.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for value, text in [(0, "Benign"), (1, "Attack-labelled")]:
        s = predictions.loc[predictions.label==value, "score"]
        if len(s):
            ax.hist(s, bins=30, alpha=.5, density=True, label=text)
    ax.set(xlabel="Fused empirical-rank score (not probability)", ylabel="Density", title=warning)
    ax.legend(); fig.tight_layout(); fig.savefig(out/"score_distribution.png", dpi=160); plt.close(fig)
    for policy in ("strict", "alert"):
        row = next(r for r in metrics if r["policy"] == policy and r["subset"] == "movement+timing+context")
        values = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(values, cmap="Blues")
        for (i, j), n in np.ndenumerate(values):
            ax.text(j, i, str(n), ha="center", va="center", color="black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"], yticklabels=["Benign", "Attack"],
               xlabel="Decision", ylabel="Ground truth", title=f"{policy} policy\n{warning}")
        fig.tight_layout(); fig.savefig(out/f"confusion_{policy}.png", dpi=160); plt.close(fig)
    if (out/"learning_curve.csv").exists():
        curve = pd.read_csv(out/"learning_curve.csv")
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in ("m", "t", "c"):
            ax.plot(curve.step, curve["selection_"+name], label=name)
        ax.set(xlabel="Epoch (centralized) / round (federated)", ylabel="Benign selection MSE", title=warning)
        ax.legend(); fig.tight_layout(); fig.savefig(out/"learning_curve.png", dpi=160); plt.close(fig)


def summarize(root):
    root = Path(root)
    rows = []
    for p in sorted(root.glob("*/metrics.csv")):
        rows.append(pd.read_csv(p))
    if not rows:
        raise ValueError("No completed run metrics found")
    all_metrics = pd.concat(rows, ignore_index=True)
    all_metrics.to_csv(root/"all_metrics.csv", index=False)
    cols = ["precision", "recall", "f1", "fpr", "roc_auc", "average_precision", "pr_auc_trapezoid", "uncertainty_rate"]
    summary = all_metrics.groupby(["method", "subset", "policy"])[cols].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(root/"summary.csv", index=False)
    family_files = list(root.glob("*/per_scenario_attack.csv"))
    if family_files:
        family = pd.concat([pd.read_csv(p) for p in family_files], ignore_index=True)
        family.to_csv(root/"all_per_scenario_attack.csv", index=False)
        family_summary = family.groupby(["method", "scenario", "attack", "policy"])[cols].agg(["mean", "std", "count"])
        family_summary.columns = ["_".join(c) for c in family_summary.columns]
        family_summary.reset_index().to_csv(root/"per_scenario_attack_summary.csv", index=False)
    display = summary.reset_index()
    selected = display[(display.policy=="strict") & (display.subset.isin(["movement+timing+context", "all_features"]))]
    tex = [r"\begin{tabular}{lrrrr}", r"Method & F1 & FPR & ROC-AUC & AP \\ \hline"]
    for _, row in selected.iterrows():
        values = ["--" if pd.isna(row[c+"_mean"]) else f"{row[c+'_mean']:.4f}" for c in ("f1", "fpr", "roc_auc", "average_precision")]
        tex.append(row["method"].replace("_", r"\_")+" & "+" & ".join(values)+r" \\")
    tex += [r"\end{tabular}", "% Point estimates here; see summary.csv for seed std/count. Correlated windows are not independent trials."]
    (root/"results_table.tex").write_text("\n".join(tex), encoding="utf-8")
    origins = {read_json(p)["origin"] for p in root.glob("*/run.json")}
    warning = "SYNTHETIC DEMO ONLY: do not present these numbers as VeReMi research results." if "synthetic_demo" in origins else "Dataset experiment: review audits, configurations and limits before publication."
    text = ["# Experiment results", "", warning, "", "## Main strict-policy metrics", "", selected[["method", "f1_mean", "fpr_mean", "roc_auc_mean", "average_precision_mean"]].to_string(index=False), "",
        "## How to interpret", "",
        "Strict: only Suspicious is positive; Uncertain is negative. Alert: Uncertain and Suspicious are positive. Neither policy silently removes uncertain cases.",
        "Results are on the current message label at each complete history window. Warm-up messages are excluded; inspect coverage.json. Scores are ranks, not calibrated attack probabilities.",
        "AP and trapezoidal PR-AUC are separate measures. Undefined metrics are blank/null, never fabricated as zero.",
        "Pool metrics are descriptive: attack archives can share benign traces. Use per_scenario_attack.csv for attack-family claims and do not treat all windows as independent observations.",
        "Seed standard deviations reflect training randomness only, not independent traffic simulations. Missing std for one seed is intentional.",
        "## Included outputs", "",
        "Each neural run has models, calibration, predictions, seven stream-subset ablations, per-attack tables, confusion matrices, ROC/PR curves, learning curves, episode delays, payload communication cost and measured CPU inference latency.",
        "The implementation is a single-process federated simulator. No secure aggregation, DP, distributed network, production OBU integration, or automatic safety action is implemented."]
    (root/"REPORT.md").write_text("\n\n".join(text), encoding="utf-8")
    return summary
