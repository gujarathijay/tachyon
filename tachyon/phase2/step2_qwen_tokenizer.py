"""Phase 2, step 2: the real Qwen tokenizer on SQL text.

Run from the repo root:

    uv run python -m tachyon.phase2.step2_qwen_tokenizer
"""

from transformers import AutoConfig, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

QUESTION = "How many users signed up in 2024?"
SQL = "SELECT COUNT(*) FROM users WHERE signup_year = 2024;"
SCHEMA = (
    "CREATE TABLE users (\n"
    "  user_id INTEGER PRIMARY KEY,\n"
    "  name TEXT,\n"
    "  email TEXT,\n"
    "  signup_year INTEGER\n"
    ");"
)
VARIANTS = ["SELECT", " SELECT", "select", " select"]
NUMBERS = ["7", "2024", "1234567"]
PHASE0_QUESTION = "Write a SQL query that counts the rows in a table named users."


def ids_of(tokenizer, text: str) -> list[int]:
    """Text to token IDs, with nothing extra added."""
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def show_tokens(tokenizer, text: str) -> None:
    """Print every token of a text: its ID, its raw piece, and the text it stands for."""
    ids = ids_of(tokenizer, text)
    print(f"  text: {text!r}")
    print(f"  {len(text)} characters -> {len(ids)} tokens")
    print(f"  {'id':>8}  {'raw piece':<14} text it stands for")
    for token_id in ids:
        raw = tokenizer.convert_ids_to_tokens(token_id)
        print(f"  {token_id:>8}  {raw:<14} {tokenizer.decode([token_id])!r}")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("== 1. A SQL query, token by token ==")
    show_tokens(tokenizer, SQL)

    print("\n== 2. Same word, different tokens ==")
    for text in VARIANTS:
        ids = ids_of(tokenizer, text)
        print(f"  {text!r:<10} -> ids {ids}  pieces {tokenizer.convert_ids_to_tokens(ids)}")

    print("\n== 3. How numbers split ==")
    for text in NUMBERS:
        ids = ids_of(tokenizer, text)
        print(f"  {text!r:<10} -> {len(ids)} tokens  pieces {tokenizer.convert_ids_to_tokens(ids)}")

    print("\n== 4. Which part of a request costs the most tokens ==")
    print(f"  {'part':<10} {'characters':>10} {'tokens':>7} {'characters per token':>21}")
    for label, text in [("question", QUESTION), ("SQL", SQL), ("schema", SCHEMA)]:
        n_tokens = len(ids_of(tokenizer, text))
        print(f"  {label:<10} {len(text):>10} {n_tokens:>7} {len(text) / n_tokens:>21.2f}")

    print("\n== 5. What the model really sees: the chat template ==")
    messages = [{"role": "user", "content": PHASE0_QUESTION}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = ids_of(tokenizer, prompt_text)
    print("  ---- start of prompt ----")
    print(prompt_text, end="")
    print("  <- the model writes its answer from here")
    print("  ---- end of prompt ----")
    print(f"  your question alone: {len(ids_of(tokenizer, PHASE0_QUESTION))} tokens")
    print(f"  full prompt:         {len(prompt_ids)} tokens")

    print("\n== 6. Special tokens: markers, not text ==")
    for token in tokenizer.all_special_tokens:
        print(f"  {token!r:<22} id {tokenizer.convert_tokens_to_ids(token)}")
    print(f"  the model stops when it writes: {tokenizer.eos_token!r}")

    print("\n== 7. Tokenizer size against the embedding table ==")
    config = AutoConfig.from_pretrained(MODEL_ID)
    print(f"  tokens the tokenizer knows: {len(tokenizer):,}")
    print(f"  rows in the embedding table: {config.vocab_size:,}")
    print(f"  spare rows: {config.vocab_size - len(tokenizer):,}")

    print("\n== 8. Round trip: text -> ids -> text ==")
    back = tokenizer.decode(ids_of(tokenizer, SCHEMA))
    print(f"  the schema comes back exactly: {back == SCHEMA}")


if __name__ == "__main__":
    main()