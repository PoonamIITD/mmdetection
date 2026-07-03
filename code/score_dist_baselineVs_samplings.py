import json, numpy as np

b = json.load(open("results_baseline_GDINO_final_val.json"))
u = json.load(open("results_sampling_loss_final_val.json"))

b_scores = [p["score"] for p in b if p["score"] >= 0.05]
u_scores = [p["score"] for p in u if p["score"] >= 0.05]

print(f"Baseline  mean score: {np.mean(b_scores):.4f}  median: {np.median(b_scores):.4f}")
print(f"Updated   mean score: {np.mean(u_scores):.4f}  median: {np.median(u_scores):.4f}")