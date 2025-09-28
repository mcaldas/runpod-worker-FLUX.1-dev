#!/usr/bin/env python3
"""
Demo script showing LoRA usage examples for the FLUX.1-dev RunPod worker
"""

def show_usage_examples():
    """Display various LoRA usage examples"""

    print("🎨 FLUX.1-dev LoRA Usage Examples")
    print("=" * 50)

    # Example 1: Basic LoRA usage
    print("\n1️⃣ Basic LoRA with Local Files:")
    basic_example = {
        "input": {
            "prompt": "A cute golden retriever puppy in a meadow",
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "lora_urls": [
                "/path/to/dog_lora.safetensors",
                "/path/to/realistic_style.safetensors"
            ],
            "lora_scales": [0.8, 0.6],
            "lora_names": ["dog_style", "realistic"]
        }
    }

    print(json.dumps(basic_example, indent=2))

    # Example 2: URL-based LoRAs
    print("\n2️⃣ LoRAs from URLs (Civitai & Hugging Face):")
    url_example = {
        "input": {
            "prompt": "A majestic dragon flying over mountains, fantasy art style",
            "negative_prompt": "blurry, low quality, modern elements",
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 30,
            "guidance_scale": 8.0,
            "seed": 42,
            "lora_urls": [
                "https://civitai.com/api/download/models/12345",  # Example Civitai LoRA
                "https://huggingface.co/username/fantasy-lora/resolve/main/fantasy_style.safetensors"
            ],
            "lora_scales": [0.9, 0.7],
            "lora_names": ["dragon_style", "fantasy_art"]
        }
    }

    print(json.dumps(url_example, indent=2))

    # Example 3: Multiple style LoRAs
    print("\n3️⃣ Multiple Style LoRAs:")
    style_example = {
        "input": {
            "prompt": "A portrait of a young woman, realistic photography style",
            "height": 1024,
            "width": 768,
            "num_inference_steps": 25,
            "guidance_scale": 7.0,
            "lora_urls": [
                "https://huggingface.co/sg161222/Realistic_Vision_V2.0/resolve/main/Realistic_Vision_V2.0.safetensors",
                "https://huggingface.co/username/portrait-enhancer/resolve/main/portrait_style.safetensors"
            ],
            "lora_scales": [0.8, 0.9],
            "lora_names": ["realistic_vision", "portrait_enhancer"]
        }
    }

    print(json.dumps(style_example, indent=2))

    # Example 4: Character + Style combination
    print("\n4️⃣ Character + Style Combination:")
    character_example = {
        "input": {
            "prompt": "Emma Watson as a steampunk inventor, intricate details, golden hour lighting",
            "negative_prompt": "blurry, deformed, extra limbs",
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 28,
            "guidance_scale": 7.5,
            "lora_urls": [
                "https://civitai.com/api/download/models/67890",  # Character LoRA
                "https://huggingface.co/username/steampunk-style/resolve/main/steampunk_tech.safetensors"
            ],
            "lora_scales": [1.0, 0.7],
            "lora_names": ["emma_character", "steampunk_style"]
        }
    }

    print(json.dumps(character_example, indent=2))

    print("\n💡 Tips:")
    print("• Use lora_scales between 0.1-1.0 to control LoRA influence")
    print("• Lower scales (0.1-0.3) for subtle effects")
    print("• Higher scales (0.7-1.0) for strong style changes")
    print("• Combine multiple LoRAs for unique results")
    print("• Cached LoRAs will be reused automatically")

if __name__ == "__main__":
    import json
    show_usage_examples()
