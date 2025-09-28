[![Pruna AI Logo](https://github.com/PrunaAI/pruna/raw/main/docs/assets/images/logo.png)](https://pruna.ai)

**Simply make AI models faster, cheaper, smaller, greener!**

[![Documentation](https://img.shields.io/badge/Pruna_documentation-purple?style=for-the-badge)](https://docs.pruna.ai)

[![Website](https://img.shields.io/badge/Pruna.ai-purple?style=flat-square)](https://pruna.ai)
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FPrunaAI)](https://x.com/PrunaAI)
[![Devto](https://img.shields.io/badge/dev-to-black?style=flat-square)](https://dev.to/prunaai)
[![Reddit](https://img.shields.io/badge/Follow-r%2FPrunaAI-orange?style=social)](https://reddit.com/r/PrunaAI)
[![Discord](https://img.shields.io/badge/Discord-join_us-purple?style=flat-square)](https://discord.gg/prunaai)
[![Huggingface](https://img.shields.io/badge/Huggingface-models-yellow?style=flat-square)](https://huggingface.co/PrunaAI)
[![Replicate](https://img.shields.io/badge/replicate-black?style=flat-square)](https://replicate.com/prunaai)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat-square&logo=github&logoColor=white)](https://github.com/PrunaAI)

![Pruna AI Logo](https://github.com/PrunaAI/pruna/raw/main/docs/assets/images/triple_line.png)

![FLUX.1-dev Worker Banner](https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/flux_smashed_comparison.png)

---

We can optimize any diffusers models and optimized FLUX.1-dev using the following techniques:

![FLUX.1-dev-juiced Generated Optimisation techniques](https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/flux_combination.png)

---

Run an optimized [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) as a serverless endpoint to generate images.


> **⚠️ Important Notes:**
> - **Compilation Time**: The first request may take 2-3 minutes as the model compiles for optimal performance
> - **Warmup Time**: Subsequent requests will be faster but may still have a brief warmup period

---

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-flux1-dev)](https://www.runpod.io/console/hub/PrunaAI/runpod-worker-FLUX.1-dev)

---

## Usage

The worker accepts the following input parameters:

| Parameter                 | Type    | Default  | Required  | Description                                                                                                         |
| :------------------------ | :------ | :------- | :-------- | :------------------------------------------------------------------------------------------------------------------ |
| `prompt`                  | `str`   | `None`   | **Yes**   | The main text prompt describing the desired image.                                                                  |
| `negative_prompt`         | `str`   | `None`   | No        | Text prompt specifying concepts to exclude from the image                                                           |
| `height`                  | `int`   | `1024`   | No        | The height of the generated image in pixels                                                                         |
| `width`                   | `int`   | `1024`   | No        | The width of the generated image in pixels                                                                          |
| `seed`                    | `int`   | `None`   | No        | Random seed for reproducibility. If `None`, a random seed is generated                                              |
| `num_inference_steps`     | `int`   | `25`     | No        | Number of denoising steps for the base model                                                                        |
| `guidance_scale`          | `float` | `7.5`    | No        | Classifier-Free Guidance scale. Higher values lead to images closer to the prompt, lower values more creative       |
| `num_images`              | `int`   | `1`      | No        | Number of images to generate per prompt (Constraint: must be 1 or 2)                                                |
| `lora_urls`               | `list`  | `[]`     | No        | List of URLs or file paths to LoRA model files                                                                     |
| `lora_scales`             | `list`  | `[]`     | No        | List of scale factors for each LoRA (must match lora_urls length)                                                  |
| `lora_names`              | `list`  | `[]`     | No        | List of names for each LoRA (must match lora_urls length)                                                          |

### Example Request

```json
{
  "input": {
    "prompt": "a knitted purple prune",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
    "seed": 42,
    "num_images": 1
  }
}
```

### Example with LoRA

```json
{
  "input": {
    "prompt": "A cute golden retriever puppy playing in a sunlit meadow, realistic style",
    "negative_prompt": "blurry, low quality, deformed, ugly, text, watermark, signature",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
    "seed": 123,
    "num_images": 1,
    "lora_urls": [
      "/path/to/dog_style_lora.safetensors",
      "/path/to/realistic_style_lora.safetensors"
    ],
    "lora_scales": [0.8, 0.6],
    "lora_names": ["dog_style", "realistic_style"]
  }
}
```

### Example with LoRA from URLs

```json
{
  "input": {
    "prompt": "A cute golden retriever puppy playing in a sunlit meadow, realistic style, highly detailed",
    "negative_prompt": "blurry, low quality, deformed, ugly, text, watermark, signature",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
    "seed": 123,
    "num_images": 1,
    "lora_urls": [
      "https://civitai.com/api/download/models/12345",
      "https://huggingface.co/username/lora-repo/resolve/main/realistic_style.safetensors"
    ],
    "lora_scales": [0.8, 0.6],
    "lora_names": ["dog_style", "realistic_style"]
  }
}
```

### LoRA Usage Notes

- **LoRA Support**: The worker supports loading and applying multiple LoRA models during image generation
- **File Formats**: Currently supports SafeTensors format LoRA files
- **Scaling**: Each LoRA can have its own scale factor (0.0 to 1.0) to control influence strength
- **Multiple LoRAs**: You can apply multiple LoRAs simultaneously for combined effects
- **Performance**: LoRAs are loaded per-request and cleared afterward to maintain memory efficiency
- **Caching**: Downloaded LoRAs are cached locally for 24 hours to improve performance on repeated requests
- **URL Support**: Supports direct SafeTensors URLs from Civitai, Hugging Face, and other sources
- **Headers**: Automatically sets appropriate headers for different platforms (e.g., Referer for Civitai)

which is producing an output like this:

```json
{
  "delayTime": 11449,
  "executionTime": 6120,
  "id": "447f10b8-c745-4c3b-8fad-b1d4ebb7a65b-e1",
  "output": {
    "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAIAAADwf7zU...",
    "images": [
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAIAAADwf7zU..."
    ],
    "seed": 42
  },
  "status": "COMPLETED",
  "workerId": "462u6mrq9s28h6"
}
```

and when you convert the base64-encoded image into an actual image, it looks like this:

![FLUX.1-dev-juiced Generated Image: 'A knitted purple prune'](https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/flux_smashed_comparison.png)
