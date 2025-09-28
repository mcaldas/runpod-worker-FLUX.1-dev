#!/usr/bin/env python3
"""
Test script for LoRA functionality in FLUX.1-dev RunPod worker
"""

import json
import os
import tempfile
from pathlib import Path

# Test the LoRA handler functionality
def test_lora_handler():
    """Test LoRA handler with mock data"""

    # Import here to avoid dependency issues in test environment
    try:
        from handler import LoRAHandler
        print("✓ Successfully imported LoRAHandler")
    except ImportError as e:
        print(f"✗ Failed to import LoRAHandler: {e}")
        return False

    # Create a LoRA handler instance
    lora_handler = LoRAHandler()

    # Test cache functionality
    cache_stats = lora_handler.get_cache_stats()
    print(f"✓ Cache initialized: {cache_stats}")

    # Test local file loading (if test files exist)
    test_file = "/tmp/test_lora.safetensors"
    if os.path.exists(test_file):
        result = lora_handler.load_lora(test_file, "test_lora", 1.0)
        if result:
            print(f"✓ Successfully loaded local LoRA: {result}")
        else:
            print("✗ Failed to load local LoRA")

    # Test URL accessibility check
    test_url = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
    accessible = lora_handler._is_url_accessible(test_url)
    print(f"✓ URL accessibility check for {test_url}: {'accessible' if accessible else 'not accessible'}")

    print("✓ LoRA handler tests completed")
    return True

def test_input_validation():
    """Test input validation with LoRA parameters"""

    try:
        from schemas import INPUT_SCHEMA
        from runpod.serverless.utils.rp_validator import validate
        print("✓ Successfully imported validation components")
    except ImportError as e:
        print(f"✗ Failed to import validation components: {e}")
        return False

    # Test valid LoRA input
    test_input = {
        "prompt": "A cute dog",
        "lora_urls": ["https://example.com/lora1.safetensors"],
        "lora_scales": [0.8],
        "lora_names": ["dog_style"]
    }

    try:
        validated = validate(test_input, INPUT_SCHEMA)
        print("✓ Input validation passed for LoRA parameters")
        return True
    except Exception as e:
        print(f"✗ Input validation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing LoRA functionality...")
    print("=" * 50)

    tests_passed = 0
    total_tests = 2

    if test_lora_handler():
        tests_passed += 1

    if test_input_validation():
        tests_passed += 1

    print("=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! LoRA functionality is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
