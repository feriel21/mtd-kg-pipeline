"""
rag/llm_hf.py — FIXED
═════════════════════

Changes:
  1. do_sample=False when temperature < 0.1 (greedy decoding for extraction)
  2. Input length warning for small models
  3. Repetition penalty to prevent JSON loops
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

_MODEL_CACHE = {}


def hf_chat(
    model_name: str,
    system: str,
    user: str,
    max_new_tokens: int = 350,
    temperature: float = 0.3,
    top_p: float = 0.9
) -> str:
    """
    Call local HuggingFace LLM with chat template.
    """
    global _MODEL_CACHE

    if model_name not in _MODEL_CACHE:
        print(f"🔄 Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        _MODEL_CACHE[model_name] = (tokenizer, model)
        print(f"✅ Model loaded and cached")

    tokenizer, model = _MODEL_CACHE[model_name]

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # ── Input length check ──
    input_len = inputs.input_ids.shape[1]
    if "1.5B" in model_name or "0.5B" in model_name:
        if input_len > 1500:
            print(f"⚠️  Input is {input_len} tokens — may degrade 1.5B quality")

    # ── Generation parameters ──
    # For extraction (low temperature): use greedy decoding
    # For creative tasks (high temperature): use sampling
    use_sampling = temperature >= 0.1

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,  # prevents JSON key repetition loops
    )

    if use_sampling:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False
        # greedy decoding — no temperature/top_p needed

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()