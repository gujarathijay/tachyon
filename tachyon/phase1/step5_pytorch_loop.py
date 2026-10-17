"""Phase 1, step 5a: the training step you did by hand, written as PyTorch code.

Run from the repo root:

    uv run python -m tachyon.phase1.step5_pytorch_loop
"""

import torch

X, Y = 2.0, 12.0  # the practice problem: x = 2, y = 12
START_W1, START_W2 = 2.0, 0.5
STEPS = 8

# Your answers from the practice problem, to compare against torch.
BY_HAND = {
    "h": 4.0,
    "prediction": 2.0,
    "loss": 100.0,
    "slope for w1": -20.0,
    "slope for w2": -80.0,
    "new w1": 2.4,
    "new w2": 2.1,
    "new prediction": 10.08,
    "new loss": 3.6864,
}


def make_weights() -> tuple[torch.Tensor, torch.Tensor]:
    """Fresh weights at the starting values. requires_grad=True means: track their slopes."""
    w1 = torch.tensor(START_W1, requires_grad=True)
    w2 = torch.tensor(START_W2, requires_grad=True)
    return w1, w2


def forward(w1: torch.Tensor, w2: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The forward pass, written as plain math."""
    h = w1 * X
    prediction = w2 * h
    loss = (prediction - Y) ** 2
    return h, prediction, loss


def one_step_in_detail() -> None:
    """One training step, with torch's numbers next to yours."""
    w1, w2 = make_weights()
    optimizer = torch.optim.SGD([w1, w2], lr=0.02)

    h, prediction, loss = forward(w1, w2)  # 1. forward
    optimizer.zero_grad()  # clear any old slopes
    loss.backward()  # 2. backward: fills in w1.grad and w2.grad
    slope_w1, slope_w2 = w1.grad.item(), w2.grad.item()
    optimizer.step()  # 3. update: w = w - lr * grad, for every weight

    with torch.no_grad():  # just looking, so no notes for a backward pass
        _, new_prediction, new_loss = forward(w1, w2)

    torch_values = {
        "h": h.item(),
        "prediction": prediction.item(),
        "loss": loss.item(),
        "slope for w1": slope_w1,
        "slope for w2": slope_w2,
        "new w1": w1.item(),
        "new w2": w2.item(),
        "new prediction": new_prediction.item(),
        "new loss": new_loss.item(),
    }
    print(f"{'value':<16} {'torch':>10} {'by hand':>10}")
    for name, value in torch_values.items():
        print(f"{name:<16} {value:>10.4f} {BY_HAND[name]:>10.4f}")


def train(learning_rate: float, steps: int) -> list[tuple[int, float, float, float, float]]:
    """The full training loop. Returns one row per training step."""
    w1, w2 = make_weights()
    optimizer = torch.optim.SGD([w1, w2], lr=learning_rate)
    rows = []
    for step in range(steps + 1):
        _, prediction, loss = forward(w1, w2)
        rows.append((step, w1.item(), w2.item(), prediction.item(), loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return rows


def forget_zero_grad() -> None:
    """What goes wrong if you skip zero_grad: the slopes pile up."""
    w1, w2 = make_weights()
    print(f"{'backward call':>13} {'w1.grad':>9} {'w2.grad':>9}")
    for call in range(1, 4):
        _, _, loss = forward(w1, w2)
        loss.backward()  # no zero_grad and no update, so the weights never change
        print(f"{call:>13} {w1.grad.item():>9.1f} {w2.grad.item():>9.1f}")


def main() -> None:
    print("== 1. One training step: torch against your hand answers ==")
    one_step_in_detail()

    for rate in (0.02, 0.01):
        print(f"\n== 2. {STEPS} training steps, learning rate = {rate} ==")
        print(f"{'step':>4} {'w1':>8} {'w2':>8} {'prediction':>11} {'loss':>10}")
        for step, w1, w2, prediction, loss in train(rate, STEPS):
            print(f"{step:>4} {w1:>8.4f} {w2:>8.4f} {prediction:>11.4f} {loss:>10.4f}")

    print("\n== 3. Forget zero_grad, and the slopes pile up ==")
    forget_zero_grad()


if __name__ == "__main__":
    main()