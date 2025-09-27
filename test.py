import argparse
import json
import os
import re
from tqdm import tqdm

from utils.evaluation import save_metrics_to_csv
from utils.models import load_peft_model, get_tokenizer


def get_prompt_text(data_point):
    """Creates a unique ID for a data point based on its prompt content."""
    return data_point["prompt"][0]["content"]


def matches_user(prompt_id, user_id):
    """Checks if the data point corresponds to the specified user ID."""
    user_id_match = re.search(r"The user's entity is represented by (\d+).", prompt_id)
    return user_id_match and user_id_match.group(1) == user_id


def main():
    """Main function to run the evaluation of a model on a test dataset."""
    MAX_NEW_TOKENS = 1024

    parser = argparse.ArgumentParser(description="Model evaluation script.")
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument(
        "--data_path", type=str, default="./data/movieKnowledgeGraphTestDataset.json"
    )
    parser.add_argument("--user_id", type=str, default=None)
    args = parser.parse_args()
    print(f"Running evaluation with arguments: {args}")

    # Determine model directory for saving predictions
    model_output_dir = (
        os.path.dirname(args.lora_path) if args.lora_path else args.base_model_path
    )
    # Construct the new metrics directory path
    metrics_base_dir = "./metrics"
    relative_model_path = os.path.relpath(model_output_dir, start="./models")
    output_dir = os.path.join(metrics_base_dir, relative_model_path)

    os.makedirs(output_dir, exist_ok=True)

    predictions_file = os.path.join(output_dir, "predictions.jsonl")
    metrics_file = os.path.join(output_dir, "metrics.csv")

    # Load existing predictions if file exists
    completed_prompts = {}
    if os.path.exists(predictions_file):
        print(f"Resuming from existing predictions file: {predictions_file}")
        with open(predictions_file, "r") as f:
            for line in f:
                pred = json.loads(line)
                completed_prompts[pred["id"]] = pred["response"]

    # Load model and tokenizer
    model = load_peft_model(args.base_model_path, args.lora_path)
    tokenizer = get_tokenizer(args.base_model_path, use_fast=False, padding_side="left")

    # Load data
    with open(args.data_path, "r") as file:
        dataset = json.load(file)

    # Run evaluation loop
    with open(predictions_file, "a") as f_out:
        for data_point in tqdm(dataset, desc="Evaluating"):
            prompt_text = get_prompt_text(data_point)

            if args.user_id and not matches_user(prompt_text, args.user_id):
                continue

            if prompt_text in completed_prompts:
                continue

            text = tokenizer.apply_chat_template(
                data_point["prompt"], tokenize=False, add_generation_prompt=True
            )

            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs, max_new_tokens=MAX_NEW_TOKENS
            )
            response = tokenizer.batch_decode(
                generated_ids[:, model_inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )[0]

            # Save progress immediately
            result = {
                "id": prompt_text,
                "response": response,
                "goal": data_point["goal"],
            }
            f_out.write(json.dumps(result) + "\n")
            completed_prompts[prompt_text] = response

    # Process all results for metrics
    true_positives, false_positives, false_negatives = 0, 0, 0
    hits = [0] * 10
    mrr = 0
    num_datapoints = 0

    for data_point in dataset:
        prompt_text = get_prompt_text(data_point)
        if prompt_text not in completed_prompts:
            continue

        if args.user_id and not matches_user(prompt_text, args.user_id):
            continue

        response = completed_prompts[prompt_text]

        # Make sure the model finished its response properly
        endThinkString = "</think>"
        endThinkIndex = response.rfind(endThinkString)
        if endThinkIndex == -1:
            print("Output Did not complete thinking")
            continue
        if (
            len(tokenizer.encode(response, add_special_tokens=True))
            > MAX_NEW_TOKENS - 4
        ):
            print("Output Did not complete after thinking")
            continue

        # Extract recommendations from the response
        recommendations = re.findall(r"(?<=\n-)([^\n]+)", response)
        recommendations = [rec.strip() for rec in recommendations]

        relevance = [1 if rec in data_point["goal"] else 0 for rec in recommendations]
        true_positives += sum(relevance)
        false_positives += len(relevance) - true_positives
        false_negatives += len(data_point["goal"]) - true_positives
        rank = relevance.index(1) if 1 in relevance else -1

        if rank >= 0:
            mrr += 1 / (rank + 1)
            if len(hits) < len(recommendations):
                hits.extend([hits[-1]] * (len(recommendations) - len(hits)))
            for i in range(rank, len(hits)):
                hits[i] += 1

        num_datapoints += 1

    # Calculate and print final metrics
    metrics = {}
    if num_datapoints > 0:
        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if true_positives > 0
            else 0
        )
        mrr_score = mrr / num_datapoints

        metrics = {
            "model": args.lora_path if args.lora_path else args.base_model_path,
            "data": args.data_path,
            "user_id": args.user_id,
            "num_examples": num_datapoints,
            "Precision": precision,
            "Recall": recall,
            "MRR": mrr_score,
        }
        for i, hit_count in enumerate(hits):
            metrics[f"Hits@{i+1}"] = hit_count / num_datapoints

        print("\n" + "=" * 36)
        print(f"|    EVALUATION RESULTS ({num_datapoints} examples)    |")
        print("=" * 36)
        print(f"| {'Metric':<12} | {'Value':<18} |")
        print("|" + "-" * 14 + "|" + "-" * 20 + "|")
        print(f"| {'Precision':<12} | {metrics['Precision']:<18.4f} |")
        print(f"| {'Recall':<12} | {metrics['Recall']:<18.4f} |")
        print(f"| {'MRR':<12} | {metrics['MRR']:<18.4f} |")
        print(f"| {'Hits@1':<12} | {metrics['Hits@1']:<18.4f} |")
        print(f"| {'Hits@3':<12} | {metrics['Hits@3']:<18.4f} |")
        print(f"| {'Hits@10':<12} | {metrics['Hits@10']:<18.4f} |")
        print("=" * 36)

        save_metrics_to_csv(metrics, metrics_file)
    else:
        print("No data points were evaluated.")

    return metrics


if __name__ == "__main__":
    main()
