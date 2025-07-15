import json

def compute_and_append_average_recall(json_path, output_path=None):
    with open(json_path, 'r') as f:
        data = json.load(f)

    recall_sums = {"R@20": 0.0, "R@50": 0.0, "R@100": 0.0}
    class_count = 0

    for class_name, recalls in data.items():
        # Skip if this is already an aggregate key
        if class_name.lower() == "average":
            continue
        for k in ["R@20", "R@50", "R@100"]:
            recall_sums[k] += recalls.get(k, 0.0)
        class_count += 1

    average_recalls = {k: recall_sums[k] / class_count for k in recall_sums}
    data["average"] = average_recalls

    # Overwrite or write to a new file
    output_path = output_path or json_path
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Averages added to JSON under key 'average'. Saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    compute_and_append_average_recall("eval_tandem_on_real_sgg.json")  # Replace with your JSON file path