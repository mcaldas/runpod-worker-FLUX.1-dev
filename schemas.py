INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': False,
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'height': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'width': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 25
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 7.5
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda img_count: 3 > img_count > 0
    },
    'lora_urls': {
        'type': list,
        'required': False,
        'default': []
    },
    'lora_scales': {
        'type': list,
        'required': False,
        'default': []
    },
    'lora_names': {
        'type': list,
        'required': False,
        'default': []
    },
}
