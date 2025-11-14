# utils/load_model.py
from transformers import pipeline

_cache = {}

def load_hf_model(model_name: str):
    if model_name in _cache:
        return _cache[model_name]
    print(f"🔹 Loading HuggingFace model: {model_name} ...")
    pipe = pipeline("text-generation", model=model_name, device_map="auto", max_new_tokens=256)
    _cache[model_name] = pipe
    return pipe
