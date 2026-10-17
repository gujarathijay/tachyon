"""Phase 1, step 5b: the grid example (weights a to f), built from PyTorch's ready-made layers.

Run from the repo root:

    uv run python -m tachyon.phase1.step5b_layers
"""

import torch
from torch import nn

# One example with two numbers in, one number out. The outer brackets mean
# "a list of examples", and this list holds just one.
X = torch.tensor([[1.0, 2.0]])  # x1 = 1, x2 = 2
Y = torch.tensor([[8.0]])

LAYER1_START = [[1.0, 0.5], [2.0, 1.0]]  # rows: h1, h2. columns: x1, x2. That is a, b / c, d
LAYER2_START = [[0.5, 1.0]]  # row: prediction. columns: h1, h2. That is e, f
LEARNING_RATE = 0.01
STEPS = 20


def build_model() -> nn.Sequential:
    """Two layers in a row, with our starting weights copied in."""
    model = nn.Sequential(
        nn.Linear(2, 2, bias=False),  # layer 1: 2 numbers in, 2 numbers out (h1, h2)
        nn.Linear(2, 1, bias=False),  # layer 2: 2 numbers in, 1 number out (prediction)
    )
    with torch.no_grad():  # we are setting weights by hand, not training
        model[0].weight.copy_(torch.tensor(LAYER1_START))
        model[1].weight.copy_(torch.tensor(LAYER2_START))
    return model


def show_grids(model: nn.Sequential, what: str) -> None:
    """Print each weight grid (or its slope grid) with its shape."""
    for name, weight in model.named_parameters():
        grid = weight if what == "values" else weight.grad
        rows = [[round(v, 4) for v in row] for row in grid.tolist()]
        print(f"  {name:<9} shape {str(tuple(weight.shape)):<7} {rows}")


def main() -> None:
    model = build_model()
    loss_fn = nn.MSELoss()  # average of miss squared; with one example it is just miss squared
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print("== 1. The model ==")
    print(model)
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")
    show_grids(model, "values")

    print("\n== 2. Forward ==")
    h = model[0](X)  # layer 1 alone
    prediction = model(X)  # both layers in a row
    loss = loss_fn(prediction, Y)
    print(f"h1, h2      = {h.tolist()[0]}")
    print(f"prediction  = {prediction.item():.4f}")
    print(f"loss        = {loss.item():.4f}")

    print("\n== 3. Backward: one slope grid per weight grid ==")
    optimizer.zero_grad()
    loss.backward()
    show_grids(model, "slopes")
    print("  by hand:   a, b = -3, -6   c, d = -6, -12   e, f = -12, -24")

    print(f"\n== 4. Update, learning rate = {LEARNING_RATE} ==")
    optimizer.step()
    show_grids(model, "values")
    with torch.no_grad():
        new_prediction = model(X)
        new_loss = loss_fn(new_prediction, Y)
    print(f"  new prediction = {new_prediction.item():.4f}, new loss = {new_loss.item():.4f}")

    print(f"\n== 5. Keep training from the start, {STEPS} training steps ==")
    model = build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    print(f"{'step':>4} {'prediction':>11} {'loss':>10}")
    for step in range(STEPS + 1):
        prediction = model(X)
        loss = loss_fn(prediction, Y)
        if step in (0, 1, 2, 3, 5, 10, 20):
            print(f"{step:>4} {prediction.item():>11.4f} {loss.item():>10.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()