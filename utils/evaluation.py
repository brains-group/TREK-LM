import json
import os
import re
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score

from .data import aggregate_datasets, load_jsonl
from .models import inference, tokenizer_param

# Fixed seed for reproducibility in evaluation
torch.manual_seed(2024)

INSTRUCTIONS = {
    "pubmedqa": {"task": "mcq", "partition": "test", "instructions": "pubmedqa"},
}

pubmedqa_instruction = {
    "system": "As an expert doctor in clinical science and medical knowledge, can you tell me if the following statement is correct? Answer yes, no, or maybe.",
    "user": "The answer is:",
    "type": "task-oriented",
    "source": "",
}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def benchmark_factory(name):
    """Creates a benchmark object."""
    factories = {
        "pubmedqa": ClosedPubMedQA,
    }
    if name not in factories:
        raise ValueError(f"Benchmark {name} not found. Select one of {list(factories.keys())}")
    return factories[name](name)


class Benchmark:
    """A class to implement a benchmark for evaluation."""

    def __init__(self, name):
        self.name = name
        self.path = None
        self.splits = None
        self.hub_name = None
        self.dir_name = None
        self.train_data = None
        self.test_data = None
        self.generations = None
        self.subsets = None
        self.has_instructions = False
        self.local_path = None

    def load_from_hub(self):
        """Downloads the benchmark data from the HuggingFace hub."""
        print(f"Downloading benchmark from HuggingFace hub ({self.hub_name}).")
        try:
            cache_dir = os.path.join(ROOT_DIR, "benchmarks", "datasets")
            if self.subsets is None:
                load_dataset(self.hub_name, cache_dir=cache_dir, download_mode="force_redownload")
            else:
                for subset in self.subsets:
                    load_dataset(self.hub_name, subset, cache_dir=cache_dir, download_mode="force_redownload")
        except Exception as e:
            raise ValueError(f"Default Huggingface loader failed for benchmark {self.name}: {e}")

    def load_data(self, partition="train"):
        """Loads benchmark data from a local directory or the Hub."""
        print(f"Loading data for benchmark {self.name}.")
        if partition not in self.splits:
            raise ValueError(f"Please provide a valid partition split: {self.splits}")
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            self.load_from_hub()
        try:
            if self.subsets is None:
                if partition == "train":
                    self.train_data = load_dataset(self.path, split=partition)
                elif partition in ["test", "validation"]:
                    self.test_data = load_dataset(self.path, split=partition)
            else:
                if partition == "train":
                    self.train_data = aggregate_datasets(self.path, self.subsets, partition=partition)
                elif partition in ["test", "validation"]:
                    self.test_data = aggregate_datasets(self.path, self.subsets, partition=partition)
        except ValueError as e:
            raise ValueError(f"Couldn't load benchmark {self.name} from local path: {e}")

    def preprocessing(self, partition="train"):
        """Applies a custom pre-processing over the partition."""
        try:
            if partition == "train":
                self.train_data = self.train_data.map(self.custom_preprocessing)
            elif partition in ["test", "validation"]:
                self.test_data = self.test_data.map(self.custom_preprocessing)
        except Exception as e:
            raise ValueError(f"Error when pre-processing {self.name} {partition} data: {e}")

    def custom_preprocessing(self):
        """Wraps a benchmark-specific pre-processing function."""
        raise NotImplementedError("Implement custom_preprocessing() in a child class.")

    def add_instruction(self, instruction=None, cot_column=None, partition="train"):
        """Adds instructions to the data."""
        def _add_instruction(row):
            row["prompt"] = f"{instruction['system']}\n{row['prompt']}\n{instruction['user']}\n"
            if cot_column:
                row["gold"] = f"{row[cot_column]}.\nThe answer is: {row['gold']} ###"
            return row

        if partition in ["train", "test", "validation"]:
            data_attr = "train_data" if partition == "train" else "test_data"
            setattr(self, data_attr, getattr(self, data_attr).map(_add_instruction))
        else:
            raise ValueError(f"Please provide a valid partition split: {self.splits}")

    def add_generations(self, data):
        """Adds the generations to the respective class attribute."""
        self.generations = Dataset.from_pandas(data) if isinstance(data, pd.DataFrame) else data

    def save_generations(self, benchmark_name, run_name):
        """Saves the generations in the respective directory."""
        path = os.path.join(ROOT_DIR, "benchmarks", "generations")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        gen_path = os.path.join(path, f"{benchmark_name}-{run_name}.jsonl")
        self.generations.to_json(gen_path, orient="records")
        print(f"Stored {len(self.generations)} generations to: {gen_path}")


class ClosedPubMedQA(Benchmark):
    """PubMedQA biomedical question answering dataset."""

    def __init__(self, name="pubmedqa"):
        super().__init__(name)
        self.hub_name = "bigbio/pubmed_qa"
        self.dir_name = "bigbio___pubmed_qa"
        self.path = os.path.join(ROOT_DIR, "benchmarks", "datasets", self.dir_name)
        self.splits = ["train", "validation", "test"]
        self.subsets = ["pubmed_qa_labeled_fold0_source"]
        self.num_options = 3

    @staticmethod
    def custom_preprocessing(row):
        context = "\n".join(row["CONTEXTS"])
        row["prompt"] = f"{context}\n{row['QUESTION']}"
        row["gold"] = row["final_decision"]
        row["long_answer"] = row["LONG_ANSWER"]
        return row


def benchmark_preparation(data_obj, partition):
    """Runs the benchmark preparation pipeline."""
    data_obj.load_data(partition=partition)
    data_obj.preprocessing(partition=partition)
    instruction = pubmedqa_instruction
    print(f'Instruction used for evaluation: \n\t{instruction["system"]}\n\t{instruction["user"]}\n')
    data_obj.add_instruction(instruction=instruction, partition=partition)


def clean_answer(output):
    """Cleans the generated text output."""
    output = output.encode("ascii", "ignore").decode("ascii")
    if "yesyes" in output: return "yes"
    if "nono" in output: return "no"
    if "yesno" in output: return "yes"
    if "noyes" in output: return "no"
    return output


def eval_answer(output_full, answer):
    """Evaluates if the cleaned output matches the answer."""
    output = output_full
    default = (2, output_full, answer)  # 2 indicates an error

    if "\n##" in output:
        try:
            output = output.split("\n##")[1].split("\n")[0].strip().lower()
        except Exception:
            return default
    if "###" in answer:
        try:
            answer = answer.split("answer is:")[1].split("###")[0].strip()
        except Exception:
            return default

    output = re.sub(r"[^a-zA-Z0-9]", " ", output).strip()
    output = re.sub(" +", " ", output)
    output = clean_answer(output)

    if output in ["a", "b", "c", "d", "e", "yes", "no"]:
        return output == answer, output, answer
    return default


def accuracy_metric(data):
    """Calculates accuracy and other metrics from predictions."""
    acc, counter, error = 0, 0, 0
    preds, golds = [], []
    ignored_prompts = []
    for row in data:
        answer = row["gold"].lower()
        output = row["output"].lower()
        correct, pred, gold = eval_answer(output, answer)

        preds.append(pred)
        golds.append(gold)

        if correct == 2:
            error += 1
            ignored_prompts.append(row)
        else:
            acc += int(correct)
            counter += 1

    return {
        "accuracy": accuracy_score(golds, preds),
        "correct": acc,
        "counted": counter,
        "ignored": ignored_prompts,
        "unable_to_find_answer": error,
        "total": len(data),
    }


def display(metric_dict, run_name):
    """Displays the final accuracy score."""
    print("=" * 36)
    print(f"Report accuracy for {run_name}:")
    print(f'  Accuracy: {metric_dict["accuracy"]:.4f}')
    print("=" * 36)


def evaluate(gen_dir=None, run_name="fl"):
    """Main function to run the evaluation."""
    if gen_dir is None:
        gen_dir = os.path.join(ROOT_DIR, "benchmarks", "generations")
    path = os.path.join(gen_dir, f"pubmedqa-{run_name}.jsonl")
    run_name_from_file = os.path.basename(path).split(".")[0]
    data = load_jsonl(path)
    metrics = accuracy_metric(data)
    display(metrics, run_name_from_file)
