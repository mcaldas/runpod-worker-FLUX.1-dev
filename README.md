<div align="center">

<img src="https://github.com/PrunaAI/pruna/raw/main/docs/assets/images/logo.png" alt="Pruna AI Logo" width=400></img>


  <img src="https://github.com/PrunaAI/pruna/raw/main/docs/assets/images/element.png" alt="Element" width=10></img>
  **Simply make AI models faster, cheaper, smaller, greener!**
  <img src="https://github.com/PrunaAI/pruna/raw/main/docs/assets/images/element.png" alt="Element" width=10></img>

<br>

[![Documentation](https://img.shields.io/badge/Pruna_documentation-purple?style=for-the-badge)][documentation]

<br>

[![Website](https://img.shields.io/badge/Pruna.ai-purple?style=flat-square)][website]
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FPrunaAI)][x]
[![Devto](https://img.shields.io/badge/dev-to-black?style=flat-square)][devto]
[![Reddit](https://img.shields.io/badge/Follow-r%2FPrunaAI-orange?style=social)][reddit]
[![Discord](https://img.shields.io/badge/Discord-join_us-purple?style=flat-square)][discord]
[![Huggingface](https://img.shields.io/badge/Huggingface-models-yellow?style=flat-square)][huggingface]
[![Replicate](https://img.shields.io/badge/replicate-black?style=flat-square)][replicate]
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat-square&logo=github&logoColor=white)][github]

<br>

<img src="https://github.com/PrunaAI/pruna/raw/main/docs/assets/images/triple_line.png" alt="Pruna AI Logo" width=600, height=30></img>

</div>

![FLUX.1-dev Worker Banner](https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/flux_smashed_comparison.png)

We can optimize any diffusers models and optimized FLUX.1-dev using the following techniques:

<img src="https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/flux_combination.png" alt="FLUX.1-dev-juiced Generated Optimisation techniques" width="100%">

---

Run an optimized [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) as a serverless endpoint to generate images.

---

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-flux1-dev)](https://www.runpod.io/console/hub/runpod-workers/worker-flux1-dev)

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

### Example Request

```json
{
  "input": {
    "prompt": "a knitted purple prune",
    "negative_prompt": None,
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
    "seed": 42,
    "num_images": 1
  }
}
```

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

<img src="https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/flux_smashed_comparison.png" alt="FLUX.1-dev-juiced Generated Image: 'A knitted purple prune'" width="512" height="512">
