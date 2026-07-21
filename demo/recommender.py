"""
Inference wrapper for the FedTREK-LM demo.

Loads the (already trained) lightweight LLM together with its federated LoRA
adapter and generates movie recommendations conditioned on a user's Personal
Knowledge Graph. Model loading and generation reuse the same utilities as the
original codebase (``utils/models.py``) so the demo behaves identically to the
evaluation pipeline described in the paper.
"""

import re

import torch

from utils.models import load_peft_model, get_tokenizer
from utils import constants as C

MAX_NEW_TOKENS = 1024


def parse_recommendations(response):
    """Extracts the recommended titles from a dash-bulleted model response.

    Mirrors the recommendation extraction used during evaluation in ``test.py``.

    Args:
        response (str): Raw decoded model output.

    Returns:
        list[str]: The recommended titles, in order.
    """
    recommendations = re.findall(r"(?<=\n-)([^\n]+)", response)
    return [rec.strip() for rec in recommendations]


def strip_thinking(response):
    """Removes a Qwen3 ``<think>...</think>`` block for cleaner display."""
    end = response.rfind("</think>")
    if end != -1:
        return response[end + len("</think>") :].strip()
    return response.strip()


class Recommender:
    """Holds a loaded model/tokenizer and turns PKG prompts into recommendations."""

    def __init__(self, base_model_path="Qwen/Qwen3-0.6B", lora_path=None):
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.model = load_peft_model(base_model_path, lora_path)
        self.model.eval()
        self.tokenizer = get_tokenizer(
            base_model_path, use_fast=False, padding_side="left"
        )

    def recommend(self, system_prompt, user_request=None, max_new_tokens=MAX_NEW_TOKENS):
        """Generates a recommendation for a PKG system prompt and user request.

        Args:
            system_prompt (str): The KG preface system message (PKG serialized as JSON-LD).
            user_request (str, optional): The user's natural-language request.
                Defaults to the training-time request string.
            max_new_tokens (int): Generation budget.

        Returns:
            dict: ``{"raw", "answer", "recommendations"}``.
        """
        if not user_request:
            user_request = C.REQUEST_STRING

        messages = [
            {C.ROLE_STRING: C.SYSTEM_STRING, C.CONTENT_STRING: system_prompt},
            {C.ROLE_STRING: C.USER_STRING, C.CONTENT_STRING: user_request},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs, max_new_tokens=max_new_tokens
            )
        response = self.tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]

        answer = strip_thinking(response)
        return {
            "raw": response,
            "answer": answer,
            "recommendations": parse_recommendations(response),
        }
