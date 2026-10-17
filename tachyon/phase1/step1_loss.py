"""Phase 1, step 1: a model with two parameters, and how wrong it is.

Run from the repo root:

    uv run python -m tachyon.phase1.step1_loss
"""

# Made-up data: how long an answer took, by how long the answer was.
# x = answer length in tens of tokens, y = seconds.
XS = [1.0, 2.0, 3.0, 4.0]
YS = [0.7, 1.2, 1.7, 2.2]


def predict(w: float, b: float, x: float) -> float:
    """The whole model: one weight and one bias."""
    return w * x + b


def loss(w: float, b: float) -> float:
    """Mean squared error: the average of the squared misses."""
    total = 0.0
    for x, y in zip(XS, YS):
        miss = predict(w, b, x) - y
        total += miss**2
    return total / len(XS)


def show(w: float, b: float) -> None:
    """Print every step of the loss for one guess of w and b."""
    print(f"\nguess: w = {w}, b = {b}")
    print(f"{'x':>4} {'y (true)':>9} {'prediction':>11} {'miss':>6} {'miss squared':>13}")
    for x, y in zip(XS, YS):
        prediction = predict(w, b, x)
        miss = round(prediction - y, 10)
        print(f"{x:>4} {y:>9} {prediction:>11.2f} {miss:>6.2f} {miss**2:>13.4f}")
    print(f"loss = average of the last column = {loss(w, b):.4f}")


def show_loss_map() -> None:
    """Print the loss for a grid of guesses, to see where it is smallest."""
    ws = [i / 10 for i in range(0, 11)]  # 0.0, 0.1, ... 1.0
    bs = [i / 10 for i in range(0, 6)]  # 0.0, 0.1, ... 0.5

    print("\nloss map (rows: w, columns: b)")
    print("        " + "".join(f"b={b:<6}" for b in bs))
    for w in ws:
        print(f"w={w:<5} " + "".join(f"{loss(w, b):<8.3f}" for b in bs))

    best_loss, best_w, best_b = min((loss(w, b), w, b) for w in ws for b in bs)
    print(f"\nsmallest loss on the map: {best_loss:.3f} at w = {best_w}, b = {best_b}")


def main() -> None:
    show(0.0, 0.0)  # a model that knows nothing
    show(0.4, 0.3)  # a close guess
    show(0.5, 0.2)  # the exact answer
    show_loss_map()


if __name__ == "__main__":
    main()