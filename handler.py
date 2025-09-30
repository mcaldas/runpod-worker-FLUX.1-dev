import base64
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import runpod
import torch
from diffusers import FluxPipeline
from peft import LoraConfig
from runpod.serverless.utils import rp_cleanup, rp_upload
from runpod.serverless.utils.rp_validator import validate

from schemas import INPUT_SCHEMA

torch.cuda.empty_cache()


class LoRAHandler:
    def __init__(self):
        self.loaded_loras = {}
        self.current_adapters = []
        self.cache_dir = Path("/tmp/lora_cache")
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, lora_url: str) -> Path:
        """Generate cache file path for a LoRA URL."""
        url_hash = hashlib.md5(lora_url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.safetensors"

    def _download_lora(self, lora_url: str, cache_path: Path) -> bool:
        """
        Download LoRA from URL to cache path.

        Args:
            lora_url: URL to download from
            cache_path: Local path to save the file

        Returns:
            True if download successful, False otherwise
        """
        try:
            print(f"[LoRAHandler] Downloading LoRA from: {lora_url}")

            # Set appropriate headers for different sites
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; RunPod-FLUX-Worker/1.0)'
            }

            # Add referer for Civitai
            if 'civitai.com' in lora_url:
                headers['Referer'] = 'https://civitai.com/'

            response = requests.get(lora_url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()

            # Check if it's actually a safetensors file
            content_type = response.headers.get('content-type', '').lower()
            if 'application/octet-stream' not in content_type and 'safetensors' not in content_type:
                print(f"[LoRAHandler] Warning: Unexpected content type: {content_type}")

            # Download with progress tracking
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"[LoRAHandler] Download progress: {progress:.1f}%")

            print(f"[LoRAHandler] Successfully downloaded LoRA to: {cache_path}")
            return True

        except Exception as e:
            print(f"[LoRAHandler] Error downloading LoRA from {lora_url}: {e}")
            if cache_path.exists():
                cache_path.unlink()  # Clean up partial download
            return False

    def _is_url_accessible(self, url: str) -> bool:
        """Check if URL is accessible without downloading."""
        try:
            response = requests.head(url, timeout=10)
            return response.status_code == 200
        except:
            return False

    def load_lora(self, lora_url: str, lora_name: str, scale: float = 1.0) -> Optional[str]:
        """
        Load a LoRA from URL or local path with caching.

        Args:
            lora_url: URL or local path to LoRA weights
            lora_name: Name identifier for the LoRA
            scale: Scale factor for the LoRA

        Returns:
            Adapter name if successful, None otherwise
        """
        try:
            print(f"[LoRAHandler] Loading LoRA: {lora_name} from {lora_url} with scale {scale}")

            # Handle local file paths
            if not lora_url.startswith(('http://', 'https://')):
                if not os.path.exists(lora_url):
                    print(f"[LoRAHandler] Local LoRA file not found: {lora_url}")
                    return None
                local_path = lora_url
            else:
                # Handle URLs with caching
                cache_path = self._get_cache_path(lora_url)

                if cache_path.exists():
                    print(f"[LoRAHandler] Using cached LoRA: {cache_path}")
                    local_path = str(cache_path)
                else:
                    print(f"[LoRAHandler] Cache miss, downloading LoRA...")
                    if not self._download_lora(lora_url, cache_path):
                        return None
                    local_path = str(cache_path)

            # Store LoRA info for later use
            adapter_name = f"{lora_name}_{len(self.loaded_loras)}"
            self.loaded_loras[adapter_name] = {
                'path': local_path,
                'scale': scale,
                'name': lora_name,
                'url': lora_url
            }

            print(f"[LoRAHandler] Successfully loaded LoRA: {adapter_name}")
            return adapter_name

        except Exception as e:
            print(f"[LoRAHandler] Error loading LoRA {lora_name}: {e}")
            return None

    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cached LoRA files."""
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            for cache_file in self.cache_dir.glob("*.safetensors"):
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > max_age_seconds:
                    cache_file.unlink()
                    print(f"[LoRAHandler] Cleaned up old cache file: {cache_file}")

        except Exception as e:
            print(f"[LoRAHandler] Error during cache cleanup: {e}")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        try:
            cache_files = list(self.cache_dir.glob("*.safetensors"))
            total_size = sum(f.stat().st_size for f in cache_files)

            return {
                'cached_files': len(cache_files),
                'total_size_mb': total_size / (1024 * 1024),
                'cache_dir': str(self.cache_dir)
            }
        except Exception as e:
            print(f"[LoRAHandler] Error getting cache stats: {e}")
            return {}

    def _get_diffusers_pipe(self, pipeline: FluxPipeline) -> Optional[FluxPipeline]:
        """Return the underlying Diffusers pipeline if available."""
        if isinstance(pipeline, FluxPipeline):
            return pipeline
        pipe = getattr(pipeline, "pipeline", None)
        if pipe is None:
            pipe = getattr(pipeline, "model", None)
        if pipe is None:
            print("[LoRAHandler] Unable to access underlying Flux pipeline instance")
        return pipe

    def apply_loras_to_pipe(self, pipe: FluxPipeline, lora_names: List[str], lora_scales: List[float]) -> bool:
        """
        Apply specified LoRAs to the pipeline.

        Args:
            pipe: PrunaModel pipeline to modify
            lora_names: List of LoRA names to apply
            lora_scales: Corresponding scale values

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"[LoRAHandler] Applying {len(lora_names)} LoRAs to pipeline")

            diffusers_pipe = self._get_diffusers_pipe(pipe)
            if diffusers_pipe is None:
                raise RuntimeError("Unable to access underlying Flux pipeline; cannot apply LoRAs.")

            # Unload existing LoRAs before applying new ones
            self.clear_adapters(pipe)

            adapter_names: List[str] = []
            adapter_scales: List[float] = []

            for name, scale in zip(lora_names, lora_scales):
                adapter_name = None
                for key, lora_info in self.loaded_loras.items():
                    if lora_info['name'] == name:
                        adapter_name = key
                        break

                if adapter_name is None:
                    raise RuntimeError(f"LoRA '{name}' not found in loaded cache; ensure load_lora() succeeded.")

                try:
                    diffusers_pipe.load_lora_weights(
                        adapter_path=self.loaded_loras[adapter_name]['path'],
                        adapter_name=adapter_name,
                        low_cpu_mem_usage=True,
                    )
                    scale_value = scale if not isinstance(scale, list) else scale[0]
                    adapter_names.append(adapter_name)
                    adapter_scales.append(scale_value)
                    print(f"[LoRAHandler] Prepared LoRA {name} with scale {scale_value}")
                except Exception as exc:
                    raise RuntimeError(f"Failed to apply LoRA '{name}': {exc}") from exc

            if adapter_names:
                diffusers_pipe.set_adapters(adapter_names, adapter_scales)

            self.current_adapters = [
                {
                    'name': name,
                    'adapter_name': adapter_name,
                    'scale': scale if not isinstance(scale, list) else scale[0]
                }
                for name, adapter_name, scale in zip(lora_names, adapter_names, adapter_scales)
            ]
            print(f"[LoRAHandler] Successfully applied {len(adapter_names)} LoRAs")
            return True

        except Exception as e:
            print(f"[LoRAHandler] Error applying LoRAs: {e}")
            return False

    def clear_adapters(self, pipe: FluxPipeline):
        """Clear any currently applied LoRA adapters."""
        try:
            diffusers_pipe = self._get_diffusers_pipe(pipe)
            if diffusers_pipe is None:
                return

            if hasattr(diffusers_pipe, "unload_lora_weights"):
                for adapter in self.current_adapters:
                    try:
                        diffusers_pipe.unload_lora_weights(adapter['adapter_name'])
                    except Exception as exc:
                        print(f"[LoRAHandler] Failed unloading adapter {adapter['adapter_name']}: {exc}")

            # Reset pipeline state if supported
            if hasattr(diffusers_pipe, "set_adapters"):
                try:
                    diffusers_pipe.set_adapters({})
                except Exception as exc:
                    print(f"[LoRAHandler] Error clearing adapter scales: {exc}")

            print("[LoRAHandler] Clearing LoRA adapters")
        except Exception as e:
            print(f"[LoRAHandler] Error clearing adapters: {e}")
        finally:
            self.current_adapters = []


class ModelHandler:
    def __init__(self):
        self.pipe: Optional[FluxPipeline] = None
        self.lora_handler = LoRAHandler()
        self.load_models()

    def load_models(self):
        model_id = os.environ.get("HF_MODEL", "black-forest-labs/FLUX.1-dev")
        local_files_only = os.environ.get("HF_LOCAL_FILES_ONLY", "0") == "1"
        use_auth_token = os.environ.get("HF_AUTH_TOKEN")

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "use_safetensors": True,
            "local_files_only": local_files_only,
        }
        if use_auth_token:
            load_kwargs["use_auth_token"] = use_auth_token

        print(f"[ModelHandler] Loading Flux pipeline: {model_id} (local_only={local_files_only})")

        self.pipe = FluxPipeline.from_pretrained(model_id, **load_kwargs)

        if torch.cuda.is_available():
            self.pipe.to("cuda")
        else:
            self.pipe.to("cpu")

        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.enable_attention_slicing()

        # Clean up old cached LoRAs on startup
        self.lora_handler.cleanup_cache()


MODELS = ModelHandler()


def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get("BUCKET_ENDPOINT_URL", False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


@torch.inference_mode()
def generate_image(job):
    """
    Generate an image from text using FLUX.1-dev Model
    """
    # -------------------------------------------------------------------------
    # 🐞 DEBUG LOGGING
    # -------------------------------------------------------------------------
    import json
    import pprint

    # Log the exact structure RunPod delivers so we can see every nesting level.
    print("[generate_image] RAW job dict:")
    try:
        print(json.dumps(job, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job, depth=4, compact=False)

    # -------------------------------------------------------------------------
    # Original (strict) behaviour – assume the expected single wrapper exists.
    # -------------------------------------------------------------------------
    job_input = job["input"]

    print("[generate_image] job['input'] payload:")
    try:
        print(json.dumps(job_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job_input, depth=4, compact=False)

    # Input validation
    try:
        validated_input = validate(job_input, INPUT_SCHEMA)
    except Exception as err:
        import traceback

        print("[generate_image] validate(...) raised an exception:", err, flush=True)
        traceback.print_exc()
        # Re-raise so RunPod registers the failure (but logs are now visible).
        raise

    print("[generate_image] validate(...) returned:")
    try:
        print(json.dumps(validated_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(validated_input, depth=4, compact=False)

    if "errors" in validated_input:
        return {"error": validated_input["errors"]}
    job_input = validated_input["validated_input"]

    if job_input["seed"] is None:
        job_input["seed"] = int.from_bytes(os.urandom(2), "big")

    # Create generator with proper device handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(job_input["seed"])

    # Extract LoRA parameters
    lora_urls = job_input.get("lora_urls", [])
    lora_scales = job_input.get("lora_scales", [])
    lora_names = job_input.get("lora_names", [])

    # Load and apply LoRAs if provided
    lora_applied = False
    if lora_urls and len(lora_urls) > 0:
        print(f"[generate_image] Loading {len(lora_urls)} LoRAs")

        # Show cache stats before loading
        cache_stats = MODELS.lora_handler.get_cache_stats()
        print(f"[generate_image] LoRA cache stats: {cache_stats}")

        # Load LoRAs
        loaded_lora_names = []
        for i, (url, name) in enumerate(zip(lora_urls, lora_names)):
            scale = lora_scales[i] if i < len(lora_scales) else 1.0
            adapter_name = MODELS.lora_handler.load_lora(url, name, scale)
            if adapter_name:
                loaded_lora_names.append(name)

        # Apply LoRAs to pipeline
        if loaded_lora_names:
            try:
                lora_applied = MODELS.lora_handler.apply_loras_to_pipe(
                    MODELS.pipe, loaded_lora_names, lora_scales[:len(loaded_lora_names)]
                )
            except RuntimeError as exc:
                error_message = f"LoRA application failed: {exc}"
                print(f"[generate_image] {error_message}")
                return {
                    "error": error_message,
                    "details": {"lora_error": str(exc)},
                    "refresh_worker": False,
                }

    try:
        # Generate image using FLUX.1-dev pipeline
        with torch.inference_mode():
            result = MODELS.pipe(
                prompt=job_input["prompt"],
                negative_prompt=job_input["negative_prompt"],
                height=job_input["height"],
                width=job_input["width"],
                num_inference_steps=job_input["num_inference_steps"],
                guidance_scale=job_input["guidance_scale"],
                num_images_per_prompt=job_input["num_images"],
                generator=generator,
            )
            output = result.images
    except RuntimeError as err:
        print(f"[ERROR] RuntimeError in generation pipeline: {err}", flush=True)
        return {
            "error": f"RuntimeError: {err}, Stack Trace: {err.__traceback__}",
            "refresh_worker": True,
        }
    except Exception as err:
        print(f"[ERROR] Unexpected error in generation pipeline: {err}", flush=True)
        return {
            "error": f"Unexpected error: {err}",
            "refresh_worker": True,
        }

    # Clear LoRAs after generation
    if lora_applied:
        MODELS.lora_handler.clear_adapters(MODELS.pipe)
        print("[generate_image] Cleared LoRA adapters after generation")

    image_urls = _save_and_upload_images(output, job["id"])

    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input["seed"],
    }

    return results


runpod.serverless.start({"handler": generate_image})
