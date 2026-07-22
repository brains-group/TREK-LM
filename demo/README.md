# FedTREK-LM Interactive Demo

An interactive [Gradio](https://gradio.app) application that lets a user
**pick a Personal Knowledge Graph (PKG)** and **ask the trained FedTREK-LM model
for a movie recommendation**, seeing exactly how the recommendation is grounded
in the selected PKG.

This demo accompanies the paper *Federated Personal Knowledge Graph Completion
with Lightweight Large Language Models for Personalized Recommendations*
(European Semantic Web Conference, 2026) and reuses the model-loading, prompt
construction, and recommendation-parsing code from the main branch so its
behavior matches the reported evaluation exactly.

## What it does

1. **Choose a PKG.** Six curated taste profiles (action, science-fiction,
   animation, horror, romantic comedy, classic drama) are provided in
   `pkg_library.py`. Each is serialized to the same JSON-LD system prompt used in
   training (`utils/kg_creation.py`).
2. **(Optional) Evolve the PKG.** Add a new liked or disliked movie to simulate
   the PKG growing as the user interacts, then re-run to see how the
   recommendation changes. "Reset PKG" restores the original profile.
3. **Ask for a recommendation.** Enter a request (or use the default) and the
   federated, KTO-fine-tuned lightweight LLM reasons over the current PKG.
4. **Inspect the result.** The app shows the parsed recommendation list, the full
   model response, and the raw JSON-LD PKG that conditioned the model.

## Setup

```bash
pip install -r requirements.txt          # repository root deps
pip install -r demo/requirements.txt     # adds gradio
```

## Running

The base model is downloaded from the Hugging Face Hub; supply the trained
federated LoRA adapter via `--lora_path` (a local directory or a Hub repo id):

```bash
python demo/app.py \
    --base_model_path Qwen/Qwen3-0.6B \
    --lora_path /path/to/federated/adapter
```

Then open the printed local URL (default <http://127.0.0.1:7860>). Use `--share`
for a public link, or set `BASE_MODEL_PATH` / `LORA_PATH` environment variables
instead of flags.

> Running without `--lora_path` loads the untuned base Qwen3 model. This is only
> useful for smoke-testing the interface; the personalization behavior described
> in the paper requires the trained adapter.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Gradio UI: PKG selector, request box, recommendation display. |
| `recommender.py` | Loads the model (via `utils/models.py`) and generates/parses recommendations. |
| `pkg_library.py` | Curated PKG taste profiles and their JSON-LD serialization. |
| `requirements.txt` | Extra demo dependency (`gradio`). |
