"""Phase 0: check the setup and read the first hard numbers.

Run from the repo root:

    uv run python -m tachyon.env_check
"""

import platform
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
QUESTION = "Write a SQL query that counts the rows in a table named users."
MAX_NEW_TOKENS = 40

GIB = 1024**3  # bytes in one GiB, the unit memory is counted in


def pick_device() -> torch.device:
    """Pick the chip that will do the math."""
    if torch.cuda.is_available():  # an NVIDIA GPU: the rented pod, later
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # the GPU inside an Apple Silicon Mac
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    """Pick the number type: how many bytes store each parameter."""
    if device.type == "cuda":
        return torch.bfloat16  # 2 bytes per number
    return torch.float32  # 4 bytes per number, the safe choice on a Mac


def allocated_bytes(device: torch.device) -> int | None:
    """Ask the chip how much memory torch is holding on it right now."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated()
    if device.type == "mps":
        return torch.mps.current_allocated_memory()
    return None  # plain CPU memory has no simple counter in torch


def main() -> None:
    device = pick_device()
    dtype = pick_dtype(device)

    print("== 1. Software ==")
    print(f"python        {sys.version.split()[0]}")
    print(f"platform      {platform.platform()}")
    print(f"torch         {torch.__version__}")
    print(f"transformers  {transformers.__version__}")

    print("\n== 2. Device ==")
    print(f"device        {device.type}")
    if device.type == "cuda":
        print(f"gpu           {torch.cuda.get_device_name(0)}")

    print("\n== 3. Load ==")
    print(f"model         {MODEL_ID}")
    before = allocated_bytes(device)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    model.to(device)
    model.eval()
    print(f"load time     {time.perf_counter() - start:.1f} s")

    print("\n== 4. Memory math ==")
    n_params = sum(p.numel() for p in model.parameters())
    loaded_dtype = model.dtype  # read it back from the model, do not trust the request
    predicted = n_params * loaded_dtype.itemsize
    print(f"parameters        {n_params:,}")
    print(f"number type       {loaded_dtype}")
    print(f"bytes per param   {loaded_dtype.itemsize}")
    print(f"predicted memory  {predicted / GIB:.2f} GiB  (parameters x bytes)")
    after = allocated_bytes(device)
    if before is None or after is None:
        print("measured memory   not measured on plain CPU")
    else:
        measured = after - before
        print(f"measured memory   {measured / GIB:.2f} GiB  (what the chip reports)")
        print(f"gap               {100 * (measured - predicted) / predicted:+.1f} %")

    print("\nSame parameters, fewer bytes each:")
    for label, size in [("32-bit", 4), ("16-bit", 2), ("8-bit", 1), ("4-bit", 0.5)]:
        print(f"  {label:<7} {n_params * size / GIB:5.2f} GiB")

    print("\n== 5. Smoke test ==")
    messages = [{"role": "user", "content": QUESTION}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    prompt_tokens = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    new_ids = output[0, prompt_tokens:].tolist()
    elapsed = time.perf_counter() - start

    print(f"question      {QUESTION}")
    print(f"prompt        {prompt_tokens} tokens")
    print(f"answer        {len(new_ids)} new tokens in {elapsed:.1f} s  (rough, includes warm-up)")
    print("--- answer text ---")
    print(tokenizer.decode(new_ids, skip_special_tokens=True))
    print("-------------------")
    print("\nPhase 0 check passed.")


if __name__ == "__main__":
    main()