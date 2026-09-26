"""
Interactive Strands agent: Gemini drafts a text-to-image prompt; local ComfyUI renders it.

Setup:
  python3 -m pip install 'strands-agents[gemini]' google-genai
  export GEMINI_API_KEY="..."   # or GOOGLE_API_KEY

Optional (local Comfy):
  COMFYUI_BASE_URL            default http://127.0.0.1:8188
  COMFYUI_TIMEOUT_SEC         default 900
  IMAGE_OUTPUT_DIR

Run:
  python3 image_local.py
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from strands import Agent, tool
from strands.handlers.callback_handler import null_callback_handler
#from strands.models.gemini import GeminiModel
from strands.models.ollama import OllamaModel
from strands_tools import image_reader

#CHAT_MODEL_ID = "gemini-2.5-flash-lite"
CHAT_MODEL_ID = "gemma4"
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

_COMFY_BASE = os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188").rstrip("/")

_Z_IMAGE_TURBO_API_PROMPT_TEMPLATE: dict = {
    "9": {"inputs": {"filename_prefix": "z-image", "images": ["43", 0]}, "class_type": "SaveImage", "_meta": {"title": "Save Image"}},
    "39": {"inputs": {"clip_name": r"z_turbo\qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}, "class_type": "CLIPLoader", "_meta": {"title": "Load CLIP"}},
    "40": {"inputs": {"vae_name": r"z_turbo\ae.safetensors"}, "class_type": "VAELoader", "_meta": {"title": "Load VAE"}},
    "41": {"inputs": {"width": 1024, "height": 1024, "batch_size": 1}, "class_type": "EmptySD3LatentImage", "_meta": {"title": "EmptySD3LatentImage"}},
    "42": {"inputs": {"conditioning": ["45", 0]}, "class_type": "ConditioningZeroOut", "_meta": {"title": "ConditioningZeroOut"}},
    "43": {"inputs": {"samples": ["44", 0], "vae": ["40", 0]}, "class_type": "VAEDecode", "_meta": {"title": "VAE Decode"}},
    "44": {"inputs": {"seed": 852076977738625, "steps": 9, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1, "model": ["47", 0], "positive": ["45", 0], "negative": ["42", 0], "latent_image": ["41", 0]}, "class_type": "KSampler", "_meta": {"title": "KSampler"}},
    "45": {"inputs": {"text": "A classic Nike Dunk Low shoe, made of leather, with purple and orange colors inspired by the Phoenix Suns, white laces, presented against a clean white background as if in a product catalog with soft lighting.", "clip": ["39", 0]}, "class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text Encode (Prompt)"}},
    "46": {"inputs": {"unet_name": r"z_turbo\z-image-turbo-fp8-e4m3fn.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader", "_meta": {"title": "Load Diffusion Model"}},
    "47": {"inputs": {"shift": 3, "model": ["48", 0]}, "class_type": "ModelSamplingAuraFlow", "_meta": {"title": "ModelSamplingAuraFlow"}},
    "48": {"inputs": {"sage_attention": "auto", "allow_compile": False, "model": ["46", 0]}, "class_type": "PathchSageAttentionKJ", "_meta": {"title": "Patch Sage Attention KJ"}},
}


def _comfy_json_request(url: str, body: dict | None = None) -> tuple[dict | list | None, str]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if body is not None else "GET",
    )
    to = float(os.environ.get("COMFYUI_HTTP_TIMEOUT_SEC", "60"))
    with urllib.request.urlopen(req, timeout=to) as resp:
        raw = resp.read().decode()
    if not raw:
        return None, ""
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return None, raw


def _queue_comfy_prompt(workflow: dict, prompt_id: str) -> None:
    client_id = os.environ.get("COMFYUI_CLIENT_ID") or str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id, "prompt_id": prompt_id}
    url = f"{_COMFY_BASE}/prompt"
    try:
        body, raw = _comfy_json_request(url, payload)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace") if e.fp else ""
        raise RuntimeError(f"ComfyUI /prompt HTTP {e.code}: {detail or e}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach ComfyUI at {url}: {e}. Is the server running?"
        ) from e

    if not isinstance(body, dict):
        raise RuntimeError(f"Unexpected /prompt response: {raw!r}")
    if body.get("error") is not None:
        raise RuntimeError(f"ComfyUI /prompt rejected the workflow: {body.get('error')}")
    errs = body.get("node_errors")
    if errs:
        raise RuntimeError(f"ComfyUI /prompt node_errors: {errs}")


def _comfy_history(prompt_id: str) -> dict:
    url = f"{_COMFY_BASE}/history/{urllib.parse.quote(prompt_id, safe='')}"
    body, raw = _comfy_json_request(url)
    if not isinstance(body, dict):
        raise RuntimeError(f"Unexpected /history response: {raw!r}")
    return body


def _wait_comfy_first_image(prompt_id: str) -> tuple[dict, dict]:
    timeout = float(os.environ.get("COMFYUI_TIMEOUT_SEC", "900"))
    interval = float(os.environ.get("COMFYUI_POLL_INTERVAL_SEC", "0.25"))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        h = _comfy_history(prompt_id)
        entry = h.get(prompt_id)
        if isinstance(entry, dict):
            outputs = entry.get("outputs")
            if outputs:
                for node_out in outputs.values():
                    for img in node_out.get("images") or []:
                        return entry, img
            err_st = (
                entry.get("status") or entry.get("error") or entry.get("exception")
            )
            if err_st:
                raise RuntimeError(f"ComfyUI run reported an error for {prompt_id}: {err_st}")
        time.sleep(interval)
    raise TimeoutError(
        f"Timed out after {timeout:.0f}s waiting for an image output for prompt_id={prompt_id!r}."
    )


def _comfy_view_bytes(filename: str, subfolder: str, folder_type: str) -> bytes:
    q = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder or "", "type": folder_type or "output"}
    )
    url = f"{_COMFY_BASE}/view?{q}"
    with urllib.request.urlopen(url, timeout=float(os.environ.get("COMFYUI_HTTP_TIMEOUT_SEC", "60"))) as resp:
        return resp.read()


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


def _workflow_with_local_prompt(positive_prompt: str) -> dict:
    """Clone inlined z-image turbo API workflow and set CLIP encode text on node 45; randomize KSampler seed."""
    w = copy.deepcopy(_Z_IMAGE_TURBO_API_PROMPT_TEMPLATE)
    w["45"]["inputs"]["text"] = positive_prompt
    w["44"]["inputs"]["seed"] = random.randint(0, 2**48 - 1)
    return w


@tool
def generate_and_save_image_local(image_prompt: str) -> str:
    """Run the local z-image turbo ComfyUI workflow API and save the output image locally.

    Sets node 45 ``CLIPTextEncode.inputs.text`` to the finalized positive prompt (same pattern as basic_api_example.py).

    Args:
        image_prompt: Complete English prompt for node 45 only—no preamble.
    """
    final_prompt = _confirm_or_edit_image_prompt(image_prompt)
    if final_prompt is None:
        return "Image generation was not run: user cancelled at the preview step."

    try:
        workflow = _workflow_with_local_prompt(final_prompt)
    except (KeyError, TypeError) as e:
        return f"Failed to prepare local ComfyUI workflow (node 45 text): {e}"

    prompt_id = str(uuid.uuid4())
    try:
        _queue_comfy_prompt(workflow, prompt_id)
        _, meta = _wait_comfy_first_image(prompt_id)
        data = _comfy_view_bytes(
            meta["filename"],
            str(meta.get("subfolder") or ""),
            str(meta.get("type") or "output"),
        )
    except (RuntimeError, TimeoutError, KeyError, urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        return f"ComfyUI generation failed: {e}"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fname = meta.get("filename") or "comfy_output.png"
    path = _OUT_DIR / f"{stamp}_{fname}"
    path.write_bytes(data)
    return f"Saved image to {path.resolve()}"


SYSTEM_PROMPT = """You are a friendly assistant helping the user create one image.

Goals:
1) Chat with the user and ask short follow-up questions until you can write a strong, specific text-to-image prompt (subject, setting, style, mood, lighting, composition, level of detail, and any important constraints).
2) When—and only when—that information is sufficient, call the tool `generate_and_save_image_local` exactly once with a single polished English prompt string (no preamble in the argument). The CLI pauses first so they can confirm or edit—briefly mention that.
3) After the tool runs, reply clearly and quote the saved file path when generation succeeded (otherwise summarize skipped/failed outcomes)."""


def main() -> None:
    #model = GeminiModel(
    #    client=_SHARED_CLIENT,
    #    model_id=CHAT_MODEL_ID,
    #    params={"temperature": 0.7, "max_output_tokens": 2048},
    #)
    model = OllamaModel(
        host="http://localhost:11434",  # Default local Ollama endpoint
        model_id=CHAT_MODEL_ID,
        temperature=0.7,
        max_tokens=2048,
    )
    agent = Agent(
        model=model,
        tools=[generate_and_save_image_local],
        system_prompt=SYSTEM_PROMPT,
        callback_handler=null_callback_handler,
    )

    print("Image prompt agent (Gemini chat → Comfy z-image-turbo).")
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
