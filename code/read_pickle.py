import mmengine
import pprint

# Load the file
pkl_path = "results_sampling_loss.pkl"
data = mmengine.load(pkl_path)

# Inspect the very first entry
print("--- Structure of first entry in pickle ---")
pprint.pprint(data[0])

# Inspect keys specifically for pred_instances
if 'pred_instances' in data[0]:
    print("\n--- Keys in pred_instances ---")
    print(data[0]['pred_instances'].keys())