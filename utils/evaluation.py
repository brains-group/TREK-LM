import csv
import os


def save_metrics_to_csv(metrics, output_file):
    """Saves evaluation metrics to a CSV file."""
    if not metrics:
        print("No metrics to save.")
        return

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define fieldnames to ensure consistent order
    fieldnames = [
        "model",
        "data",
        "user_id",
        "num_examples",
        "Precision",
        "Recall",
        "MRR",
    ]
    hits_keys = [k for k in metrics.keys() if k.startswith("Hits@")]
    fieldnames.extend(hits_keys)

    # Add any other keys from metrics that are not already in fieldnames
    # to handle cases where more metrics are added.
    for key in metrics.keys():
        if key not in fieldnames:
            fieldnames.append(key)

    file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0

    with open(output_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    print(f"Metrics saved to {output_file}")
