#!/usr/bin/env python
"""Plot ground truth vs predicted emission wavelengths."""

import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    with open("artifacts/cross/test_predictions_cross_ensemble.json") as f:
        data = json.load(f)

    gt = np.array([p["ground_truth_emission_nm"] for p in data["predictions"]])
    pred = np.array([p["predicted_emission_nm"] for p in data["predictions"]])
    metrics = data["metrics"]

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(gt, pred, alpha=0.6, s=30, edgecolors="k", linewidths=0.3, c="#4C72B0")

    # Perfect prediction line
    lo, hi = min(gt.min(), pred.min()) - 10, max(gt.max(), pred.max()) + 10
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect prediction")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Ground Truth Emission (nm)", fontsize=12)
    ax.set_ylabel("Predicted Emission (nm)", fontsize=12)
    ax.set_title("Cross-Embedding Ensemble: GT vs Predicted", fontsize=13)

    text = f"MAE = {metrics['MAE']:.1f} nm\nRMSE = {metrics['RMSE']:.1f} nm\nR² = {metrics['R2']:.3f}\nn = {data['n_test_samples']}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))

    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("gt_vs_predicted.png", dpi=150)
    print("Saved gt_vs_predicted.png")
    plt.show()

if __name__ == "__main__":
    main()
