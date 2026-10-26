"""Phase 2, step 3: train an embedding table on one next-token example.

This is the FROM -> users example worked by hand, written as PyTorch code.

Run from the repo root:

    uv run python -m tachyon.phase2.step3_embedding
"""

import torch
from torch import nn

VOCAB = ["FROM", "users", "orders"]  # fixed by the tokenizer; token ID = position in this list

# Embedding table: one row per token we might READ. 2 numbers per token.
EMBEDDING_START = [
    [2.0, 1.0],  # FROM
    [0.5, -1.0],  # users
    [-1.0, 0.5],  # orders
]
# Output table: one row per token that might come NEXT. 2 numbers per token.
OUTPUT_START = [
    [-1.0, 1.0],  # FROM
    [0.0, 0.5],  # users
    [1.0, 0.0],  # orders
]

INPUT_TOKEN = "FROM"  # from the training text "FROM users"
TRUE_NEXT = "users"
LEARNING_RATE = 0.1


def show(table: torch.Tensor, title: str) -> None:
    """Print a table with one row per token."""
    print(f"  {title}")
    print(f"  {'token':<8} {'1st number':>11} {'2nd number':>11}")
    for token, row in zip(VOCAB, table.tolist()):
        print(f"  {token:<8} {row[0]:>11.4f} {row[1]:>11.4f}")


def show_scores(scores: torch.Tensor, target: torch.Tensor) -> None:
    """Print each score, its miss, and the model's guess."""
    print(f"  {'candidate':<10} {'score':>8} {'wanted':>7} {'miss':>8}")
    for token, score, wanted in zip(VOCAB, scores.tolist(), target.tolist()):
        print(f"  {token:<10} {score:>8.4f} {wanted:>7.0f} {score - wanted:>8.4f}")
    guess = VOCAB[scores.argmax().item()]
    loss = ((scores - target) ** 2).sum().item()
    print(f"  guess: {guess}   loss: {loss:.4f}")


def build() -> tuple[nn.Embedding, nn.Linear]:
    """The two tables, with our starting numbers typed in."""
    embedding = nn.Embedding(len(VOCAB), 2)  # 3 rows, 2 numbers each
    output = nn.Linear(2, len(VOCAB), bias=False)  # 2 numbers in, 3 scores out; grid is 3 x 2
    with torch.no_grad():  # typing in numbers, not training
        embedding.weight.copy_(torch.tensor(EMBEDDING_START))
        output.weight.copy_(torch.tensor(OUTPUT_START))
    return embedding, output


def main() -> None:
    embedding, output = build()
    input_id = torch.tensor(VOCAB.index(INPUT_TOKEN))
    # print(input_id)
    target = torch.zeros(len(VOCAB))
    target[VOCAB.index(TRUE_NEXT)] = 1.0  # wanted scores: 0, 1, 0

    print("== 1. The two tables ==")
    show(embedding.weight.detach(), "embedding table (token we read):")
    show(output.weight.detach(), "output table (token that might come next):")

    print(f"\n== 2. Lookup: '{INPUT_TOKEN}' -> id {input_id.item()} -> its row ==")
    row = embedding(input_id)
    print(f"  row = {[round(v, 4) for v in row.tolist()]}")
    one_hot = torch.zeros(len(VOCAB))
    one_hot[input_id] = 1.0
    print(f"  same thing as one-hot {one_hot.tolist()} @ table = "
          f"{[round(v, 4) for v in (one_hot @ embedding.weight).tolist()]}")

    print("\n== 3. Forward: scores, misses, loss ==")
    scores = output(row)  # each score = row's 1st x candidate's 1st + row's 2nd x candidate's 2nd
    loss = ((scores - target) ** 2).sum()
    show_scores(scores.detach(), target)

    print("\n== 4. Backward: slopes for both tables ==")
    params = list(embedding.parameters()) + list(output.parameters())
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE)
    optimizer.zero_grad()
    loss.backward()
    show(embedding.weight.grad, "embedding table slopes (only the FROM row is used):")
    show(output.weight.grad, "output table slopes:")

    print(f"\n== 5. Update, learning rate {LEARNING_RATE} ==")
    optimizer.step()
    show(embedding.weight.detach(), "embedding table after:")
    show(output.weight.detach(), "output table after:")

    print("\n== 6. Forward again ==")
    with torch.no_grad():  # just looking
        show_scores(output(embedding(input_id)), target)

    print("\n== 7. Swap the two numbers in every row of both tables ==")
    embedding, output = build()
    with torch.no_grad():
        swapped_scores = output.weight.flip(1) @ embedding.weight.flip(1)[input_id]
        original_scores = output(embedding(input_id))
    print(f"  original scores: {[round(v, 4) for v in original_scores.tolist()]}")
    print(f"  swapped scores:  {[round(v, 4) for v in swapped_scores.tolist()]}")


if __name__ == "__main__":
    main()