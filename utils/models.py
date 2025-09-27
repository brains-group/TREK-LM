import json
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

from omegaconf import DictConfig
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DataCollatorForCompletionOnlyLM


def get_model(model_cfg: DictConfig):
    """Load model with appropiate quantization config and
    other optimizations."""

    use_cuda = torch.cuda.is_available()
    quantization_config = None
    model_name = model_cfg.name
    if use_cuda:
        if model_cfg.quantization == 4:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif model_cfg.quantization == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif model_cfg.quantization is None:
            pass
        else:
            raise ValueError(
                f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
            )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    if use_cuda:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
        )

    target_modules = model_cfg.lora.target_modules
    if target_modules:
        target_modules = list(target_modules)
    peft_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    peft_model = get_peft_model(model, peft_config)
    if not (use_cuda):
        peft_model.enable_input_require_grads()

    if model_cfg.gradient_checkpointing:
        model.config.use_cache = False

    return peft_model


def get_tokenizer(model_name: str, use_fast: bool = False, padding_side: str = "left"):
    """Gets the tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=use_fast, padding_side=padding_side
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.bos_token if padding_side == "left" else tokenizer.eos_token
        )

    return tokenizer


def set_parameters(model, parameters) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def load_peft_model(base_model_path: str, lora_path: str = None):
    """Loads a base model and optionally applies a PEFT adapter."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16
    ).to(device)

    if lora_path:
        model = PeftModel.from_pretrained(
            model, lora_path, torch_dtype=torch.float16
        ).to(device)
    return model
