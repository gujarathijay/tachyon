"""Phase 1, step 4: backprop, the slopes through two layers.

Run from the repo root:

    uv run python -m tachyon.phase1.step4_backprop
"""

import torch

X, Y = 2.0, 1.2  # one row of data
W1, W2 = 0.5, 0.4  # two weights, one per layer
LEARNING_RATE = 0.1


def forward(w1: float, w2: float) -> tuple[float, float, float, float]:
    """Forward pass: work out the loss and keep every in-between value."""
    h = w1 * X  # layer 1
    prediction = w2 * h  # layer 2
    miss = prediction - Y
    loss = miss**2
    return h, prediction, miss, loss


def backward(w1: float, w2: float) -> tuple[float, float, float, float, float]:
    """Backward pass by hand: walk from the loss back to each weight."""
    h, prediction, miss, loss = forward(w1, w2)
    slope_miss = 2 * miss  # because loss = miss squared
    slope_prediction = slope_miss * 1  # because miss = prediction - y
    slope_w2 = slope_prediction * h  # because prediction = w2 * h
    slope_h = slope_prediction * w2  # because prediction = w2 * h
    slope_w1 = slope_h * X  # because h = w1 * x
    return slope_miss, slope_prediction, slope_w2, slope_h, slope_w1


def nudge(w1: float, w2: float, tiny: float = 0.0001) -> tuple[float, float]:
    """The slow way, as a check: nudge each weight and rerun the forward pass."""
    base = forward(w1, w2)[3]
    slope_w1 = (forward(w1 + tiny, w2)[3] - base) / tiny
    slope_w2 = (forward(w1, w2 + tiny)[3] - base) / tiny
    return slope_w1, slope_w2


def main() -> None:
    print(f"data: x = {X}, y = {Y}    weights: w1 = {W1}, w2 = {W2}")

    print("\n== 1. Forward pass, left to right ==")
    h, prediction, miss, loss = forward(W1, W2)
    print(f"h          = w1 * x          = {h:.4f}")
    print(f"prediction = w2 * h          = {prediction:.4f}")
    print(f"miss       = prediction - y  = {miss:.4f}")
    print(f"loss       = miss squared    = {loss:.4f}")

    print("\n== 2. Backward pass by hand, right to left ==")
    slope_miss, slope_prediction, slope_w2, slope_h, slope_w1 = backward(W1, W2)
    print(f"slope for miss       = 2 * miss                  = {slope_miss:.4f}")
    print(f"slope for prediction = slope for miss * 1        = {slope_prediction:.4f}")
    print(f"slope for w2         = slope for prediction * h  = {slope_w2:.4f}")
    print(f"slope for h          = slope for prediction * w2 = {slope_h:.4f}")
    print(f"slope for w1         = slope for h * x           = {slope_w1:.4f}")

    print("\n== 3. Three ways to get the slopes ==")
    nudge_w1, nudge_w2 = nudge(W1, W2)

    w1 = torch.tensor(W1, requires_grad=True)
    w2 = torch.tensor(W2, requires_grad=True)
    torch_loss = (w2 * (w1 * X) - Y) ** 2  # torch runs the forward pass and takes notes
    torch_loss.backward()  # torch runs the backward pass

    print(f"{'':16} {'slope for w1':>13} {'slope for w2':>13}")
    print(f"{'nudge':16} {nudge_w1:>13.4f} {nudge_w2:>13.4f}")
    print(f"{'by hand':16} {slope_w1:>13.4f} {slope_w2:>13.4f}")
    print(f"{'torch':16} {w1.grad.item():>13.4f} {w2.grad.item():>13.4f}")

    print(f"\n== 4. One update, learning rate = {LEARNING_RATE} ==")
    new_w1 = W1 - LEARNING_RATE * slope_w1
    new_w2 = W2 - LEARNING_RATE * slope_w2
    new_loss = forward(new_w1, new_w2)[3]
    print(f"w1: {W1} -> {new_w1:.4f}")
    print(f"w2: {W2} -> {new_w2:.4f}")
    print(f"loss: {loss:.4f} -> {new_loss:.4f}")


if __name__ == "__main__":
    main()