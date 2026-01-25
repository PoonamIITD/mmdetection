import json
import argparse

def fix_category_ids(pred_json, out_json, offset=1):
    with open(pred_json, "r") as f:
        preds = json.load(f)

    for p in preds:
        p["category_id"] = int(p["category_id"]) + offset

    with open(out_json, "w") as f:
        json.dump(preds, f, indent=2)

    print(f"✅ Fixed category_ids (+{offset})")
    print(f"✅ Saved to: {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="input preds json")
    parser.add_argument("--out", required=True, help="output fixed preds json")
    parser.add_argument("--offset", type=int, default=1)
    args = parser.parse_args()

    fix_category_ids(args.pred, args.out, args.offset)
