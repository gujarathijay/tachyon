"""Phase 2, step 4a: softmax and cross-entropy, the loss real language models use.

Same FROM -> users example and same two tables as step 3. Only the loss changes.

Run from the repo root:

    uv run python -m tachyon.phase2.step4a_softmax_loss
"""

import math

import torch
from torch import nn

from tachyon.phase2.step3_embedding import INPUT_TOKEN, TRUE_NEXT, VOCAB, build, show

LEARNING_RATE = 1.0
TINY = 0.01


def softmax(scores: list[float]) -> list[float]:
    """Scores -> probabilities that add up to 1."""
    powers = [math.exp(s) for s in scores]
    total = sum(powers)
    return [p / total for p in powers]


def cross_entropy(scores: list[float], true_index: int) -> float:
    """Loss = minus the log of the probability given to the true next token."""
    return -math.log(softmax(scores)[true_index])


def main() -> None:
    embedding, output = build()
    input_id = torch.tensor(VOCAB.index(INPUT_TOKEN))
    true_index = VOCAB.index(TRUE_NEXT)
    wanted = [1.0 if i == true_index else 0.0 for i in range(len(VOCAB))]

    scores = output(embedding(input_id))
    s = scores.tolist()

    print("== 1. Scores (same as step 3) ==")
    for token, score in zip(VOCAB, s):
        print(f"  {token:<8} {score:>7.4f}")

    print("\n== 2. Softmax by hand: e^score, add them up, divide ==")
    powers = [math.exp(v) for v in s]
    total = sum(powers)
    probs = softmax(s)
    print(f"  {'token':<8} {'score':>7} {'e^score':>9} {'probability':>12}")
    for token, score, power, prob in zip(VOCAB, s, powers, probs):
        print(f"  {token:<8} {score:>7.4f} {power:>9.4f} {power:>6.4f}/{total:.4f} = {prob:.4f}")
    print(f"  probabilities add up to {sum(probs):.4f}")
    print(f"  torch.softmax gives    {[round(p, 4) for p in torch.softmax(scores, dim=0).tolist()]}")

    print(f"\n== 3. Cross-entropy: loss = -log(probability of '{TRUE_NEXT}') ==")
    loss_by_hand = cross_entropy(s, true_index)
    loss = nn.functional.cross_entropy(scores.unsqueeze(0), torch.tensor([true_index]))
    print(f"  by hand: -log({probs[true_index]:.4f}) = {loss_by_hand:.4f}")
    print(f"  torch:   {loss.item():.4f}")

    print("\n== 4. How the loss reacts to the probability of the true token ==")
    for p in [1.0, 0.9, 0.5, 0.1, 0.01]:
        print(f"  probability {p:<5} -> loss {-math.log(p) + 0.0:.4f}")

    print("\n== 5. Slope for each score = probability - wanted ==")
    scores.retain_grad()  # ask torch to keep the slope for the scores too
    loss.backward()
    print(f"  {'token':<8} {'prob':>7} {'wanted':>7} {'by hand':>8} {'torch':>8} {'nudge':>8}")
    for i, token in enumerate(VOCAB):
        nudged = list(s)
        nudged[i] += TINY
        nudge = (cross_entropy(nudged, true_index) - loss_by_hand) / TINY
        print(f"  {token:<8} {probs[i]:>7.4f} {wanted[i]:>7.0f} {probs[i] - wanted[i]:>8.4f} "
              f"{scores.grad[i].item():>8.4f} {nudge:>8.4f}")

    print("\n== 6. Slopes for both tables (torch) ==")
    show(embedding.weight.grad, "embedding table slopes:")
    show(output.weight.grad, "output table slopes:")

    print(f"\n== 7. Update, learning rate {LEARNING_RATE}, then look again ==")
    with torch.no_grad():  # the update itself, by hand: weight = weight - lr x slope
        for weight in (embedding.weight, output.weight):
            weight -= LEARNING_RATE * weight.grad
        new_scores = output(embedding(input_id)).tolist()
    show(embedding.weight.detach(), "embedding table after:")
    show(output.weight.detach(), "output table after:")
    new_probs = softmax(new_scores)
    print(f"  {'token':<8} {'score':>8} {'probability':>12}")
    for token, score, prob in zip(VOCAB, new_scores, new_probs):
        print(f"  {token:<8} {score:>8.4f} {prob:>12.4f}")
    print(f"  guess: {VOCAB[new_probs.index(max(new_probs))]}   "
          f"loss: {loss_by_hand:.4f} -> {cross_entropy(new_scores, true_index):.4f}")


if __name__ == "__main__":
    main()