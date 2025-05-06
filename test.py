import datasets
import argparse
import json
import sys
import re

sys.path.append("../../")
sys.path.append("../../../")
from tqdm import tqdm
import os
import torch
import sys


from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen3-4B")
parser.add_argument("--lora_path", type=str, default=None)
args = parser.parse_args()
print(args)

# ============= Extract model name from the path. The name is used for saving results. =============
if args.lora_path:
    pre_str, checkpoint_str = os.path.split(args.lora_path)
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = checkpoint_str.split("-")[-1]
    model_name = f"{exp_name}_{checkpoint_id}"
else:
    pre_str, last_str = os.path.split(args.base_model_path)
    if last_str.startswith("full"):  # if the model is merged as full model
        _, exp_name = os.path.split(pre_str)
        checkpoint_id = last_str.split("-")[-1]
        model_name = f"{exp_name}_{checkpoint_id}"
    else:
        model_name = last_str  # mainly for base model
        exp_name = model_name


# ============= Generate responses =============
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    args.base_model_path, torch_dtype=torch.float16
).to(device)
if args.lora_path is not None:
    model = PeftModel.from_pretrained(
        model, args.lora_path, torch_dtype=torch.float16
    ).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)


def runTests(dataset):
    score = 0
    for dataPoint in tqdm(dataset):
        text = tokenizer.apply_chat_template(
            dataPoint["prompt"], tokenize=False, add_generation_prompt=True
        )

        print(f"---------------- PROMPT --------------\n{text}")

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"---------------- RESPONSE --------------\n{response}")

        completion = dataPoint["completion"][0]["content"]
        print(f"---------------- COMPLETION --------------\n{completion}")
        goal = dataPoint["goal"]
        if goal in response:
            score += 1
            print(f"{goal} found in response.")
        print(f"Score: {score}")
    return score / len(dataset)


with open("./data/KGComp/movieKnowledgeGraphTestDataset.json", "r") as file:
    print("Performing Real Data Test:")
    print(f"Real Data Score: {runTests(json.load(file))}")

with open("./data/movieKnowledgeGraphSyntheticTestDataset.json", "r") as file:
    print("Performing Synthetic Data Test:")
    print(f"Synthetic Data Score: {runTests(json.load(file))}")

with open("./data/movieKnowledgeGraphSyntheticLinkTestDataset.json", "r") as file:
    print("Performing Synthetic Link Prediction Data Test:")
    print(f"Synthetic Link Prediction Data Score: {runTests(json.load(file))}")
