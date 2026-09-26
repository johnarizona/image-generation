"""
Interactive Strands agent: chat with gemini-2.5-flash-lite until the conversation
yields a solid text-to-image prompt, then call gemini-2.5-flash-image to render it.

Setup:
  python3 -m pip install 'strands-agents[gemini]' google-genai
  export GEMINI_API_KEY="..."   # or GOOGLE_API_KEY

Run:
  python3 strands_image_chat_agent.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types as genai_types
from strands import Agent, tool
from strands.handlers.callback_handler import null_callback_handler
from strands.models.gemini import GeminiModel

CHAT_MODEL_ID = "gemini-2.5-flash-lite"
# Override with env if Google renames the model id in your account.
IMAGE_MODEL_ID = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
DEFAULT_OUT_DIR = Path.cwd() / "generated_images"


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("Set GEMINI_API_KEY or GOOGLE_API_KEY.", file=sys.stderr)
        sys.exit(1)
    return key


_SHARED_CLIENT = genai.Client(api_key=_api_key())
_OUT_DIR = Path(os.environ.get("IMAGE_OUTPUT_DIR", DEFAULT_OUT_DIR)).expanduser()
_OUT_DIR.mkdir(parents=True, exist_ok=True)


def _pick_extension(mime: str) -> str:
    m = (mime or "image/png").lower()
    if "jpeg" in m or "jpg" in m:
        return ".jpg"
    if "webp" in m:
        return ".webp"
    return ".png"


def _first_image_from_response(response: genai_types.GenerateContentResponse) -> tuple[bytes, str] | None:
    cands = getattr(response, "candidates", None) or []
    if not cands:
        return None
    content = getattr(cands[0], "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return None
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue
        data = getattr(inline, "data", None)
        if data:
            mime = getattr(inline, "mime_type", None) or "image/png"
            return data, mime
    return None


def _read_multiline_prompt(edit_label: str) -> str:
    print(edit_label)
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _confirm_or_edit_image_prompt(initial: str) -> str | None:
    """Let the user approve, edit, or cancel before generation. Returns None if cancelled."""
    text = (initial or "").strip()
    if not text:
        print("Prompt is empty; nothing to generate.", file=sys.stderr)
        return None
    print()
    while True:
        print("--- Final image prompt (shown before generation) ---")
        print(text)
        print("--- End prompt ---")
        try:
            choice = (
                input("[Enter]=generate image  |  e=edit prompt  |  q=cancel: ")
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled image generation.")
            return None
        if choice in {"", "y", "yes"}:
            return text
        if choice == "q":
            print("Skipping image generation (user cancelled at review step).")
            return None
        if choice != "e":
            print("Type Enter, e, or q.")
            continue
        revised = _read_multiline_prompt(
            "Enter the full revised prompt (one line at a time; empty line finishes):"
        )
        if revised:
            text = revised
        else:
            print("No text entered; keeping the previous draft.")


@tool
def generate_and_save_image(image_prompt: str) -> str:
    """Render an image with Gemini image generation and save it to disk.

    Call this only once you have a single, detailed English prompt that fully
    describes subject, setting, style, lighting, mood, composition, and camera
    feel. The parameter must be the final prompt text only (no extra commentary).

    Before calling the image model, the user sees this prompt and may edit or
    cancel.

    Args:
        image_prompt: Complete text-to-image prompt to send to the image model.
    """
    final_prompt = _confirm_or_edit_image_prompt(image_prompt)
    if final_prompt is None:
        return "Image generation was not run: user cancelled at the preview step."

    config = genai_types.GenerateContentConfig(
        response_modalities=[genai_types.Modality.TEXT, genai_types.Modality.IMAGE],
    )
    resp = _SHARED_CLIENT.models.generate_content(
        model=IMAGE_MODEL_ID,
        contents=final_prompt,
        config=config,
    )
    blob = _first_image_from_response(resp)
    if not blob:
        return (
            "Image generation returned no image bytes. "
            "The request may have been blocked or the response had no inline image data."
        )
    data, mime = blob
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = _OUT_DIR / f"gemini_image_{stamp}{_pick_extension(mime)}"
    path.write_bytes(data)
    return f"Saved image to {path.resolve()}"


SYSTEM_PROMPT = """You are a friendly assistant helping the user create one image.

Goals:
1) Chat with the user and ask short follow-up questions until you can write a strong, specific text-to-image prompt (subject, setting, style, mood, lighting, composition, level of detail, and any important constraints).
2) When—and only when—that information is sufficient, call the tool `generate_and_save_image` exactly once with a single polished English prompt string (no preamble in the tool argument). The CLI will pause to show them that draft so they can confirm, edit line-by-line until an empty line, or cancel—do not apologize for this; briefly note they can approve or edit there.
3) After the tool runs, reply clearly and quote the full saved file path the tool returned so the user knows exactly where the file is (if generation was skipped, summarize that outcome)."""


def main() -> None:
    model = GeminiModel(
        client=_SHARED_CLIENT,
        model_id=CHAT_MODEL_ID,
        params={"temperature": 0.7, "max_output_tokens": 2048},
    )
    agent = Agent(
        model=model,
        tools=[generate_and_save_image],
        system_prompt=SYSTEM_PROMPT,
        callback_handler=null_callback_handler,
    )

    print("Image prompt agent (gemini-2.5-flash-lite → gemini-2.5-flash-image).")
    print(f"Images save under: {_OUT_DIR.resolve()}")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"quit", "exit", "q"}:
            break

        result = agent(line)
        print(f"Agent: {result}\n")


if __name__ == "__main__":
    main()
