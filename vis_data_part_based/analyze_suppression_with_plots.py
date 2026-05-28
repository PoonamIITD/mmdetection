"""
analyze_suppression_with_plots.py
==================================
Extends the suppression counting script with layer-wise
confidence divergence plots.

Plots produced
--------------
1. mean_trajectory.png      – mean ± std band for occluded vs occluder
                              across all 6 decoder layers (suppressed pairs only)
2. trajectory_grid.png      – individual trajectory grid (first N matched cases)
3. delta_per_layer.png      – mean Δconf (layer l → l+1) for both groups
4. suppression_heatmap.png  – per-pair confidence heatmap sorted by occluded delta
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── config ────────────────────────────────────────────────
JSON_PATH        = "occlusion_results_0.6-0.8_0.75-1.0_scores.json"
OUT_DIR          = Path("plots")
GRID_MAX_CASES   = 24          # how many individual traces to show in grid
EPS              = 1e-6
LAYER_LABELS     = [f"L{i}" for i in range(6)]

OCC_COLOR        = "#E63946"   # red   – occluded
OCCR_COLOR       = "#2A9D8F"   # teal  – occluder
BAND_ALPHA       = 0.18

# ─────────────────────────────────────────────────────────
# CONDITIONS  (unchanged from original)
# ─────────────────────────────────────────────────────────

def occluded_not_maintained(scores):
    end  = scores[-1]
    peak = max(scores)
    return (peak > end + EPS) or (end < scores[0] - EPS)

def occluder_increasing(scores):
    return scores[-1] > scores[0] + EPS

# ─────────────────────────────────────────────────────────
# LOAD & FILTER
# ─────────────────────────────────────────────────────────

with open(JSON_PATH) as f:
    data = json.load(f)

total_pairs   = len(data)
matched_cases = []

for item in data:
    occ_s  = item["occluded_scores_per_layer"]
    occr_s = item["occluder_scores_per_layer"]
    if occluded_not_maintained(occ_s) and occluder_increasing(occr_s):
        matched_cases.append({
            "image":          item["image_name"],
            "occluded_query": item["occluded_query"],
            "occluder_query": item["occluder_query"],
            "occluded_scores":  np.array(occ_s),
            "occluder_scores":  np.array(occr_s),
            "occlusion_ratio":  item.get("occlusion_ratio", 0),
            "same_class":       item.get("same_class", False),
            "occluded_delta":   item.get("occluded_delta", occ_s[-1] - occ_s[0]),
        })

count = len(matched_cases)

print("=" * 60)
print(f"Total pairs                          : {total_pairs}")
print(f"Suppressed occluded + incr. occluder : {count}")
print(f"Percentage                           : {count/total_pairs*100:.2f}%")
print("=" * 60)

OUT_DIR.mkdir(exist_ok=True)

# stack arrays  [N, 6]
occ_mat  = np.stack([m["occluded_scores"]  for m in matched_cases])   # [N,6]
occr_mat = np.stack([m["occluder_scores"]  for m in matched_cases])   # [N,6]
layers   = np.arange(6)

# ─────────────────────────────────────────────────────────
# PLOT 1 – MEAN TRAJECTORY  (main figure)
# ─────────────────────────────────────────────────────────

occ_mean  = occ_mat.mean(0);   occ_std  = occ_mat.std(0)
occr_mean = occr_mat.mean(0);  occr_std = occr_mat.std(0)

fig, ax = plt.subplots(figsize=(7, 4.5))

# std bands
ax.fill_between(layers, occ_mean  - occ_std,  occ_mean  + occ_std,
                color=OCC_COLOR,  alpha=BAND_ALPHA)
ax.fill_between(layers, occr_mean - occr_std, occr_mean + occr_std,
                color=OCCR_COLOR, alpha=BAND_ALPHA)

# mean lines
ax.plot(layers, occ_mean,  color=OCC_COLOR,  lw=2.5, marker="o",
        markersize=7, label=f"Occluded  (n={count})")
ax.plot(layers, occr_mean, color=OCCR_COLOR, lw=2.5, marker="s",
        markersize=7, label=f"Occluder  (n={count})")

# annotate start / end delta
for i, (y, c) in enumerate([(occ_mean, OCC_COLOR), (occr_mean, OCCR_COLOR)]):
    delta = y[-1] - y[0]
    sign  = "+" if delta >= 0 else ""
    ax.annotate(f"{sign}{delta:.3f}",
                xy=(5, y[-1]), xytext=(5.1, y[-1]),
                color=c, fontsize=9, va="center")

ax.set_xticks(layers)
ax.set_xticklabels(LAYER_LABELS)
ax.set_xlabel("Decoder Layer", fontsize=12)
ax.set_ylabel("Class Confidence", fontsize=12)
ax.set_title("Layer-wise Confidence Divergence\n"
             "(Suppressed pairs: occluded vs occluder)", fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "mean_trajectory.png", dpi=150)
plt.close(fig)
print(f"[SAVED] {OUT_DIR}/mean_trajectory.png")

# ─────────────────────────────────────────────────────────
# PLOT 2 – INDIVIDUAL TRAJECTORY GRID
# ─────────────────────────────────────────────────────────

n_show = min(GRID_MAX_CASES, count)
ncols  = 4
nrows  = (n_show + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(ncols * 3.5, nrows * 2.8),
                         sharey=True)
axes_flat = axes.flatten() if n_show > 1 else [axes]

for i in range(n_show):
    ax  = axes_flat[i]
    m   = matched_cases[i]
    ax.plot(layers, m["occluded_scores"],  color=OCC_COLOR,  lw=1.8,
            marker="o", ms=5, label="Occluded")
    ax.plot(layers, m["occluder_scores"],  color=OCCR_COLOR, lw=1.8,
            marker="s", ms=5, label="Occluder")
    ax.set_title(f"{Path(m['image']).stem}\n"
                 f"occ_q={m['occluded_query']} | "
                 f"Δ={m['occluded_delta']:+.2f}",
                 fontsize=7.5)
    ax.set_xticks(layers);  ax.set_xticklabels(LAYER_LABELS, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7, loc="upper right")

# hide unused subplots
for j in range(n_show, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle(f"Individual Confidence Trajectories  "
             f"(first {n_show} suppressed pairs)", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "trajectory_grid.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[SAVED] {OUT_DIR}/trajectory_grid.png")

# ─────────────────────────────────────────────────────────
# PLOT 3 – MEAN Δ CONFIDENCE PER LAYER STEP
# Δ(l) = mean(score[l+1] - score[l])   for l = 0..4
# ─────────────────────────────────────────────────────────

occ_deltas  = np.diff(occ_mat,  axis=1).mean(0)   # [5]
occr_deltas = np.diff(occr_mat, axis=1).mean(0)   # [5]
step_labels = [f"L{i}→L{i+1}" for i in range(5)]
x = np.arange(5)
w = 0.35

fig, ax = plt.subplots(figsize=(7, 4))
bars1 = ax.bar(x - w/2, occ_deltas,  width=w, color=OCC_COLOR,
               label="Occluded",  alpha=0.85)
bars2 = ax.bar(x + w/2, occr_deltas, width=w, color=OCCR_COLOR,
               label="Occluder",  alpha=0.85)

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x);  ax.set_xticklabels(step_labels, fontsize=10)
ax.set_xlabel("Layer Transition", fontsize=12)
ax.set_ylabel("Mean ΔConfidence", fontsize=12)
ax.set_title("Mean Confidence Change per Decoder Step\n"
             "(Suppressed pairs)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, axis="y", linestyle="--", alpha=0.4)

# value labels on bars
for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            h + (0.002 if h >= 0 else -0.006),
            f"{h:+.3f}", ha="center", va="bottom" if h >= 0 else "top",
            fontsize=8)

fig.tight_layout()
fig.savefig(OUT_DIR / "delta_per_layer.png", dpi=150)
plt.close(fig)
print(f"[SAVED] {OUT_DIR}/delta_per_layer.png")

# ─────────────────────────────────────────────────────────
# PLOT 4 – HEATMAP: per-pair trajectories sorted by occluded delta
# ─────────────────────────────────────────────────────────

sort_idx  = np.argsort([m["occluded_delta"] for m in matched_cases])
occ_sorted  = occ_mat[sort_idx]
occr_sorted = occr_mat[sort_idx]

fig = plt.figure(figsize=(12, max(4, count * 0.18 + 1.5)))
gs  = gridspec.GridSpec(1, 2, wspace=0.05)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

im1 = ax1.imshow(occ_sorted,  aspect="auto", cmap="RdYlGn",
                 vmin=0, vmax=1, interpolation="nearest")
im2 = ax2.imshow(occr_sorted, aspect="auto", cmap="RdYlGn",
                 vmin=0, vmax=1, interpolation="nearest")

for ax, title in [(ax1, "Occluded"), (ax2, "Occluder")]:
    ax.set_xticks(range(6));  ax.set_xticklabels(LAYER_LABELS, fontsize=9)
    ax.set_xlabel("Decoder Layer", fontsize=10)
    ax.set_title(title, fontsize=11)

ax1.set_ylabel(f"Pair index (sorted by occluded Δ, n={count})", fontsize=9)
ax2.set_yticks([])

fig.colorbar(im1, ax=ax2, fraction=0.046, pad=0.04, label="Confidence")
fig.suptitle("Per-pair Confidence Heatmap  (sorted by occluded Δconf)",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "suppression_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[SAVED] {OUT_DIR}/suppression_heatmap.png")

# ─────────────────────────────────────────────────────────
# OPTIONAL: print first 20 cases (same as original)
# ─────────────────────────────────────────────────────────

print()
for i, m in enumerate(matched_cases[:20]):
    print(f"Case {i+1}  |  {m['image']}  |  "
          f"occ_q={m['occluded_query']}  occr_q={m['occluder_query']}")
    print("  Occluded :", ["%.3f" % x for x in m["occluded_scores"]])
    print("  Occluder :", ["%.3f" % x for x in m["occluder_scores"]])