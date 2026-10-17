"""Phase 1, step 5c: why a network needs an activation, and what an MLP is.

Run from the repo root:

    uv run python -m tachyon.phase1.step5c_activation
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to a file, never open a window
import matplotlib.pyplot as plt
import torch
from torch import nn

# A V shape: y is x with the minus sign dropped. No straight line can fit it.
XS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
X = torch.tensor(XS).reshape(-1, 1)  # 9 examples, 1 number each
Y = X.abs()

HIDDEN = 8  # numbers in the middle layer
LEARNING_RATE = 0.05
STEPS = 2000
CHART_PATH = Path("docs/img/phase1_activation.png")


def relu(z: float) -> float:
    """Keep a positive number. Turn a negative number into 0."""
    return z if z > 0 else 0.0


def build(kind: str) -> nn.Sequential:
    """Three models to compare. Same start every time, thanks to the seed."""
    torch.manual_seed(0)
    if kind == "one layer":
        return nn.Sequential(nn.Linear(1, 1))
    if kind == "two layers, no activation":
        return nn.Sequential(nn.Linear(1, HIDDEN), nn.Linear(HIDDEN, 1))
    return nn.Sequential(nn.Linear(1, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))


def train(model: nn.Sequential) -> float:
    """The same training loop as 5a and 5b. Returns the final loss."""
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    for _ in range(STEPS):
        loss = loss_fn(model(X), Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return loss_fn(model(X), Y).item()


def draw_chart(models: dict[str, nn.Sequential]) -> None:
    """One small chart per model: the data as dots, the model as a line."""
    grid = torch.linspace(-2, 2, 101).reshape(-1, 1)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor="#fcfcfb", sharey=True)
    for axis, (label, model) in zip(axes, models.items()):
        with torch.no_grad():
            line = model(grid)
        axis.plot(grid.flatten(), line.flatten(), color="#2a78d6", linewidth=2, label="model")
        axis.plot(XS, Y.flatten(), "o", color="#52514e", markersize=6, label="data")
        axis.set_title(label, color="#0b0b0b", loc="left")
        axis.set_xlabel("x", color="#52514e")
        axis.set_facecolor("#fcfcfb")
        axis.grid(axis="y", color="#e1e0d9", linewidth=1)
        axis.tick_params(colors="#898781", length=0)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color("#c3c2b7")
    axes[0].set_ylabel("y", color="#52514e")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", frameon=False, ncol=2)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(CHART_PATH, dpi=150)


def main() -> None:
    print("== 1. ReLU by hand ==")
    for z in [-2.0, -0.5, 0.0, 0.5, 2.0]:
        print(f"  relu({z:>4}) = {relu(z)}")

    print("\n== 2. Two hidden numbers and ReLU make the V exactly ==")
    print("  h1 = relu(1 * x),  h2 = relu(-1 * x),  prediction = 1 * h1 + 1 * h2")
    print(f"  {'x':>5} {'h1':>5} {'h2':>5} {'prediction':>11} {'true y':>7}")
    for x in XS:
        h1, h2 = relu(1 * x), relu(-1 * x)
        print(f"  {x:>5} {h1:>5} {h2:>5} {1 * h1 + 1 * h2:>11} {abs(x):>7}")

    print(f"\n== 3. Train three models on the V ({STEPS} training steps each) ==")
    models = {kind: build(kind) for kind in
              ["one layer", "two layers, no activation", "two layers + ReLU"]}
    print(f"  {'model':<27} {'parameters':>10} {'final loss':>11}")
    for label, model in models.items():
        final_loss = train(model)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {label:<27} {n_params:>10} {final_loss:>11.4f}")

    print("\n== 4. Two layers with nothing between them are one straight line ==")
    first, second = models["two layers, no activation"]
    with torch.no_grad():
        slope = round((second.weight @ first.weight).item(), 4) + 0.0
        start = round((second.weight @ first.bias + second.bias).item(), 4) + 0.0
    print(f"  all {HIDDEN * 2 + HIDDEN + 1} weights and biases act like: prediction = {slope:.4f} * x + {start:.4f}")

    draw_chart(models)
    print(f"\nchart saved to {CHART_PATH}")


if __name__ == "__main__":
    main()