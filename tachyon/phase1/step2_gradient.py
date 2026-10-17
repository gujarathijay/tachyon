"""Phase 1, step 2: the gradient, which way is downhill.

Run from the repo root:

    uv run python -m tachyon.phase1.step2_gradient
"""

from tachyon.phase1.step1_loss import XS, YS, loss, predict

LEARNING_RATE = 0.05
STEPS = 5


def gradient_by_nudge(w: float, b: float, tiny: float = 0.0001) -> tuple[float, float]:
    """Slope by trial: nudge one parameter and see how far the loss moves."""
    base = loss(w, b)
    slope_w = (loss(w + tiny, b) - base) / tiny
    slope_b = (loss(w, b + tiny) - base) / tiny
    return slope_w, slope_b


def gradient_by_formula(w: float, b: float) -> tuple[float, float]:
    """Slope by math: one pass over the data gives both slopes."""
    sum_w = 0.0
    sum_b = 0.0
    for x, y in zip(XS, YS):
        miss = predict(w, b, x) - y
        sum_w += 2 * miss * x
        sum_b += 2 * miss
    return sum_w / len(XS), sum_b / len(XS)


def show_formula_by_hand(w: float, b: float) -> None:
    """Print every step of the formula for one guess of w and b."""
    print(f"{'x':>4} {'y':>5} {'miss':>6} {'2*miss*x':>10} {'2*miss':>8}")
    for x, y in zip(XS, YS):
        miss = predict(w, b, x) - y
        print(f"{x:>4} {y:>5} {miss:>6.2f} {2 * miss * x:>10.2f} {2 * miss:>8.2f}")
    slope_w, slope_b = gradient_by_formula(w, b)
    print(f"{'average':>17} {slope_w:>10.2f} {slope_b:>8.2f}")


def main() -> None:
    w, b = 0.0, 0.0

    print(f"start: w = {w}, b = {b}, loss = {loss(w, b):.4f}")

    print("\n== 1. The slope by formula, every step shown ==")
    show_formula_by_hand(w, b)

    print("\n== 2. Two ways to get the slope ==")
    nudge_w, nudge_b = gradient_by_nudge(w, b)
    formula_w, formula_b = gradient_by_formula(w, b)
    print(f"{'':10} {'slope for w':>12} {'slope for b':>12}")
    print(f"{'nudge':10} {nudge_w:>12.4f} {nudge_b:>12.4f}")
    print(f"{'formula':10} {formula_w:>12.4f} {formula_b:>12.4f}")

    print(f"\n== 3. Walk downhill, learning rate = {LEARNING_RATE} ==")
    print(f"{'step':>4} {'w':>8} {'b':>8} {'loss':>8} {'slope w':>9} {'slope b':>9}")
    for step in range(STEPS + 1):
        slope_w, slope_b = gradient_by_formula(w, b)
        print(f"{step:>4} {w:>8.4f} {b:>8.4f} {loss(w, b):>8.4f} {slope_w:>9.4f} {slope_b:>9.4f}")
        w = w - LEARNING_RATE * slope_w
        b = b - LEARNING_RATE * slope_b


if __name__ == "__main__":
    main()