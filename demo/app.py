"""
Interactive Gradio demo for FedTREK-LM.

The user picks one of several Personal Knowledge Graphs (PKGs), each capturing a
distinct movie taste profile, optionally *evolves* the PKG by adding new liked or
disliked movies, and asks the federated, KTO-fine-tuned lightweight LLM for a
movie recommendation. The demo shows both the PKG that conditions the model and
the recommendation it produces, and lets the user watch the recommendation change
as the PKG grows.

Usage:
    python demo/app.py --lora_path /path/to/federated/adapter \
                       --base_model_path Qwen/Qwen3-0.6B

The base model is pulled from the Hugging Face Hub; the LoRA adapter path points
to the released federated checkpoint (see demo/README.md). If no adapter is
given, the untuned base model is used, which is useful only for smoke-testing
the interface.
"""

import argparse
import json
import os
import sys

# Allow running as `python demo/app.py` from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from demo.pkg_library import (
    PKG_LIBRARY,
    build_system_prompt,
    list_profiles,
    clone_profile,
    add_preference,
)
from demo.recommender import Recommender
from utils import constants as C


def format_pkg_markdown(profile):
    """Renders a profile's liked/disliked movies as Markdown for display."""
    liked = "\n".join(f"- 👍 {m}" for m in profile.get("liked", [])) or "- _(none)_"
    disliked = (
        "\n".join(f"- 👎 {m}" for m in profile.get("disliked", [])) or "- _(none)_"
    )
    return (
        f"**{profile['name']}** &mdash; {profile['description']}\n\n"
        f"**Liked movies**\n{liked}\n\n"
        f"**Disliked movies**\n{disliked}"
    )


def pretty_jsonld(profile):
    """Returns the PKG's JSON-LD block (extracted from the system prompt) for display."""
    prompt = build_system_prompt(profile)
    start = prompt.find("{")
    end = prompt.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.dumps(json.loads(prompt[start : end + 1]), indent=2)
        except json.JSONDecodeError:
            pass
    return prompt


def build_ui(recommender):
    """Constructs the Gradio Blocks interface bound to a loaded recommender."""
    default_key = list_profiles()[0][0]
    choices = [(name, key) for key, name in list_profiles()]

    def on_select(profile_key):
        """Load (reset to) the chosen library PKG."""
        profile = clone_profile(profile_key)
        return (
            profile,
            format_pkg_markdown(profile),
            pretty_jsonld(profile),
            f"Loaded the '{profile['name']}' PKG.",
        )

    def on_add(profile, movie, relation):
        """Evolve the current PKG by adding a liked/disliked movie."""
        profile, message = add_preference(profile, movie, relation)
        return (
            profile,
            format_pkg_markdown(profile),
            pretty_jsonld(profile),
            message,
            "",  # clear the movie textbox
        )

    def on_reset(profile):
        """Restore the current PKG to its original library state."""
        key = profile.get("_key", default_key)
        profile = clone_profile(key)
        return (
            profile,
            format_pkg_markdown(profile),
            pretty_jsonld(profile),
            "Reverted the PKG to its original state.",
        )

    def on_recommend(profile, user_request):
        """Generate a recommendation for the current (possibly evolved) PKG."""
        system_prompt = build_system_prompt(profile)
        result = recommender.recommend(system_prompt, user_request)
        if result["recommendations"]:
            recs_md = "\n".join(f"- 🎬 {r}" for r in result["recommendations"])
        else:
            recs_md = "_The model did not return a parseable recommendation list._"
        return recs_md, result["answer"]

    with gr.Blocks(title="FedTREK-LM Demo") as demo:
        gr.Markdown(
            "# FedTREK-LM: Personalized Movie Recommendations over Personal "
            "Knowledge Graphs\n"
            "Pick a Personal Knowledge Graph (PKG), optionally **evolve** it by "
            "adding new preferences, then ask the federated, KTO-fine-tuned "
            "lightweight LLM for a movie recommendation. The model reasons over "
            "the structured PKG to suggest movies the user has not already seen."
        )

        # Holds the current, possibly-evolved PKG for the session.
        profile_state = gr.State(clone_profile(default_key))

        with gr.Row():
            with gr.Column(scale=1):
                selector = gr.Dropdown(
                    choices=choices,
                    value=default_key,
                    label="Choose a Personal Knowledge Graph",
                )
                pkg_view = gr.Markdown(
                    format_pkg_markdown(PKG_LIBRARY[default_key])
                )
                with gr.Accordion("View PKG as JSON-LD", open=False):
                    jsonld_view = gr.Code(
                        value=pretty_jsonld(PKG_LIBRARY[default_key]),
                        language="json",
                        label="PKG (JSON-LD)",
                    )
                with gr.Accordion("Evolve this PKG", open=True):
                    gr.Markdown(
                        "Add a new preference to simulate the PKG evolving as the "
                        "user interacts, then re-run to see how the recommendation "
                        "changes."
                    )
                    movie_box = gr.Textbox(
                        label="Movie title (e.g., Dune (2021))", lines=1
                    )
                    relation_radio = gr.Radio(
                        choices=[C.LIKED_STRING, C.DISLIKED_STRING],
                        value=C.LIKED_STRING,
                        label="Add as",
                    )
                    with gr.Row():
                        add_btn = gr.Button("Add to PKG")
                        reset_btn = gr.Button("Reset PKG")
                    evolve_status = gr.Markdown("")
            with gr.Column(scale=1):
                request_box = gr.Textbox(
                    value=C.REQUEST_STRING,
                    label="Your request to the recommender",
                    lines=2,
                )
                run_btn = gr.Button("Get Recommendation", variant="primary")
                recs_view = gr.Markdown(label="Recommendations")
                with gr.Accordion("Full model response", open=False):
                    raw_view = gr.Textbox(label="Model output", lines=6)

        selector.change(
            on_select,
            inputs=selector,
            outputs=[profile_state, pkg_view, jsonld_view, evolve_status],
        )
        add_btn.click(
            on_add,
            inputs=[profile_state, movie_box, relation_radio],
            outputs=[profile_state, pkg_view, jsonld_view, evolve_status, movie_box],
        )
        reset_btn.click(
            on_reset,
            inputs=profile_state,
            outputs=[profile_state, pkg_view, jsonld_view, evolve_status],
        )
        run_btn.click(
            on_recommend,
            inputs=[profile_state, request_box],
            outputs=[recs_view, raw_view],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="FedTREK-LM interactive demo.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=os.environ.get("BASE_MODEL_PATH", "Qwen/Qwen3-0.6B"),
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=os.environ.get("LORA_PATH"),
        help="Path to the trained federated LoRA adapter (local dir or HF repo id).",
    )
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link."
    )
    args = parser.parse_args()

    print(f"Loading model: base={args.base_model_path} lora={args.lora_path}")
    recommender = Recommender(args.base_model_path, args.lora_path)

    demo = build_ui(recommender)
    demo.launch(
        server_name=args.server_name, server_port=args.server_port, share=args.share
    )


if __name__ == "__main__":
    main()
