import csv
import os


def save_metrics_to_csv(metrics, output_file, allStdDev=False):
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
        "Precision_SE",
        "Recall",
        "Recall_SE",
        "F1-Score",
        "F1-Score_SE",
        "MRR",
        "MRR_StdDev",
    ]
    # Add Hits@k fields, sorted numerically
    hits_keys = sorted(
        [
            k
            for k in metrics.keys()
            if k.startswith("Hits@")
            and (not k.endswith("_SE") and not k.endswith("_StdDev"))
        ],
        key=lambda x: int(x.split("@")[1]),
    )
    for k in hits_keys:
        fieldnames.append(k)
        fieldnames.append(f"{k}_SE")
    if allStdDev:
        fieldnames = [
            fn.replace("_SE", "_StdDev") if fn.endswith("_SE") else fn
            for fn in fieldnames
        ]

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
