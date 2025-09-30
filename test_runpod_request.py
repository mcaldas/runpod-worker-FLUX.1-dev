import base64
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests


DEFAULT_ENDPOINT_ID = "0gyjprqckgksqm"
RUN_URL_TEMPLATE = "https://api.runpod.ai/v2/{endpoint_id}/run"
STATUS_URL_TEMPLATE = "https://api.runpod.ai/v2/{endpoint_id}/status/{request_id}"
POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 600
OUTPUT_ROOT = Path("runpod_outputs")


PROMPT_PAYLOAD: Dict[str, Any] = {
    "input": {
        "prompt": (
            "a woman laying on the couch with her hairy pussy, 1girl, solo, breasts, looking at viewer, smile, open mouth, brown hair, navel, jewelry, medium breasts, nipples, nude, earrings, lying, teeth, pussy, spread legs, cum, on back, mole, pubic hair, uncensored, cum in pussy, female pubic hair, piercing, leg up, ring, couch, after sex, mole on breast, realistic, after vaginal, navel piercing"
        ),
        "negative_prompt": "blurry, low quality, deformed, ugly, text, watermark, signature",
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 28,
        "guidance_scale": 7.5,
        "seed": -1,
        "num_images": 1,
        "lora_urls": [
            "https://civitai.com/api/download/models/746602?type=Model&format=SafeTensor&token=c1362e7b9bf398980d9cd54ba1630852",
            "https://civitai.com/api/download/models/720252?type=Model&format=SafeTensor&token=c1362e7b9bf398980d9cd54ba1630852",
        ],
        "lora_scales": [0.8, 0.6],
        "lora_names": ["ns_master", "FaeTastic"],
    }
}


def load_api_key() -> str:
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit(
            "RUNPOD_API_KEY is not set. Export it or place it in your environment before running."
        )
    return api_key


def get_endpoint_id() -> str:
    return os.getenv("RUNPOD_ENDPOINT_ID", DEFAULT_ENDPOINT_ID)


def submit_job(endpoint_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
    run_url = RUN_URL_TEMPLATE.format(endpoint_id=endpoint_id)
    payload = json.loads(json.dumps(PROMPT_PAYLOAD))  # deep copy without mutating constant
    prompt_seed = payload["input"].get("seed", -1)
    if prompt_seed in (-1, None):
        prompt_seed = random.randint(0, 2**32 - 1)
        payload["input"]["seed"] = prompt_seed

    print(f"Using seed: {payload['input']['seed']}")

    response = requests.post(run_url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def poll_job(endpoint_id: str, request_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
    status_url = STATUS_URL_TEMPLATE.format(endpoint_id=endpoint_id, request_id=request_id)
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    while True:
        response = requests.get(status_url, headers=headers, timeout=60)
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        print(f"Status: {status}")
        if status in {"COMPLETED", "FAILED"}:
            return payload
        if time.time() > deadline:
            raise TimeoutError(
                f"Timed out waiting for request {request_id} after {POLL_TIMEOUT_SECONDS} seconds"
            )
        time.sleep(POLL_INTERVAL_SECONDS)


def _flatten_strings(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        results: List[str] = []
        for item in value.values():
            results.extend(_flatten_strings(item))
        return results
    if isinstance(value, Iterable):
        results = []
        for item in value:
            results.extend(_flatten_strings(item))
        return results
    return []


def _decode_base64_image(data: str) -> bytes | None:
    if not isinstance(data, str):
        return None
    payload = data
    if "base64," in data:
        _, payload = data.split("base64,", 1)
    try:
        return base64.b64decode(payload, validate=True)
    except Exception:
        return None


def save_base64_images(result: Dict[str, Any], request_id: str) -> None:
    output_payload = result.get("output")
    if output_payload is None:
        print("No output payload available to save.")
        return

    candidate_strings = _flatten_strings(output_payload)
    decoded_images: List[bytes] = []
    for candidate in candidate_strings:
        decoded = _decode_base64_image(candidate)
        if decoded:
            decoded_images.append(decoded)

    if not decoded_images:
        print("No base64 image data found in output.")
        return

    target_dir = OUTPUT_ROOT / request_id
    target_dir.mkdir(parents=True, exist_ok=True)

    for index, image_bytes in enumerate(decoded_images):
        file_path = target_dir / f"image_{index:02d}.png"
        with open(file_path, "wb") as file_handle:
            file_handle.write(image_bytes)
        print(f"Saved image to {file_path}")


def main() -> None:
    api_key = load_api_key()
    endpoint_id = get_endpoint_id()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    print(f"Submitting job to endpoint {endpoint_id}...")
    run_response = submit_job(endpoint_id, headers)
    request_id = run_response.get("id")
    if not request_id:
        print("No request ID returned; exiting without polling.")
        return

    print(
        "Request submitted:",
        json.dumps(
            {
                "id": request_id,
                "status": run_response.get("status"),
                "worker": run_response.get("workerId"),
            }
        ),
    )

    print(f"Polling status for request {request_id}...")
    final_status = poll_job(endpoint_id, request_id, headers)
    print(
        "Job finished:",
        json.dumps(
            {
                "id": request_id,
                "status": final_status.get("status"),
                "output_keys": list(final_status.get("output", {}).keys())
                if isinstance(final_status.get("output"), dict)
                else "n/a",
            }
        ),
    )

    save_base64_images(final_status, request_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"Error: {exc}", file=sys.stderr)
        raise
