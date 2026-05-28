import json

JSON_PATH = "occlusion_results_different_class.json"

# ------------------------------------------------------------
# Conditions

# for occluded max(s0,s1,...)>s_final
# for occluder s_final> s_start
# This captures:

# rise-then-fall,
# continuous fall,
# unstable confidence trajectories of occluded queries,
# while the occluder query keeps strengthening through decoder refinement.
# ------------------------------------------------------------

def occluded_not_maintained(scores, eps=1e-6):
    """
    Count as suppressed if:
    1. final score is lower than earlier peak
    OR
    2. overall decreasing
    """

    start = scores[0]
    end = scores[-1]
    peak = max(scores)

    # peak occurs before final layer
    peak_before_end = peak > end + eps

    # overall decrease
    overall_drop = end < start - eps

    return peak_before_end or overall_drop


def occluder_increasing(scores, eps=1e-6):
    """
    Occluder confidence should increase overall.
    """
    return scores[-1] > scores[0] + eps


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

with open(JSON_PATH, "r") as f:
    data = json.load(f)

total_pairs = len(data)
count = 0

matched_cases = []

for item in data:

    occ_scores = item["occluded_scores_per_layer"]
    occdr_scores = item["occluder_scores_per_layer"]

    cond1 = occluded_not_maintained(occ_scores)
    cond2 = occluder_increasing(occdr_scores)

    if cond1 and cond2:
        count += 1

        matched_cases.append({
            "image": item["image_name"],
            "occluded_query": item["occluded_query"],
            "occluder_query": item["occluder_query"],
            "occluded_scores": occ_scores,
            "occluder_scores": occdr_scores
        })

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("=" * 60)
print(f"Total pairs: {total_pairs}")
print(f"Suppressed occluded + increasing occluder: {count}")
print(f"Percentage: {(count / total_pairs) * 100:.2f}%")
print("=" * 60)

# Optional detailed printing
for i, m in enumerate(matched_cases[:20]):

    print(f"\nCase {i+1}")
    print("Image:", m["image"])
    print("Occluded query:", m["occluded_query"])
    print("Occluder query:", m["occluder_query"])

    print("Occluded :", ["%.3f" % x for x in m["occluded_scores"]])
    print("Occluder :", ["%.3f" % x for x in m["occluder_scores"]])