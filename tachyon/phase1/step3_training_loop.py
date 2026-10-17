"""Phase 1, step 3: the training loop and the learning rate.

Run from the repo root:

    uv run python -m tachyon.phase1.step3_training_loop
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to a file, never open a window
import matplotlib.pyplot as plt

from tachyon.phase1.step1_loss import loss
from tachyon.phase1.step2_gradient import gradient_by_formula

STEPS = 30
RATES = {"too small": 0.001, "good": 0.05, "too big": 0.13}
CHART_PATH = Path("docs/img/phase1_learning_rates.png")


def train(learning_rate: float, steps: int) -> tuple[list[float], float, float]:
    """The training loop. Returns the loss after every step, and the final w and b."""
    w, b = 0.0, 0.0
    losses = [loss(w, b)]
    for _ in range(steps):
        slope_w, slope_b = gradient_by_formula(w, b)  # 1. which way is downhill
        w = w - learning_rate * slope_w  # 2. step that way
        b = b - learning_rate * slope_b
        losses.append(loss(w, b))  # 3. how wrong are we now
    return losses, w, b


def compact(value: float) -> str:
    """Short text for a number: 0.0350, 2.42, 31K."""
    if value >= 1000:
        return f"{value / 1000:.0f}K"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def draw_chart(results: dict[str, list[float]]) -> None:
    """One small chart per learning rate, each with its own loss scale."""
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), facecolor="#fcfcfb")
    for axis, (label, losses) in zip(axes, results.items()):
        axis.plot(losses, color="#2a78d6", linewidth=2)
        axis.plot(len(losses) - 1, losses[-1], "o", color="#2a78d6", markersize=8)
        axis.annotate(
            compact(losses[-1]),
            (len(losses) - 1, losses[-1]),
            textcoords="offset points",
            xytext=(-8, 8),
            ha="right",
            color="#0b0b0b",
        )
        axis.set_title(f"learning rate {RATES[label]}: {label}", color="#0b0b0b", loc="left")
        axis.set_xlabel("step", color="#52514e")
        axis.set_ylim(bottom=0)
        axis.yaxis.set_major_formatter(
            lambda value, _: f"{value / 1000:.0f}K" if value >= 1000 else f"{value:g}"
        )
        axis.set_facecolor("#fcfcfb")
        axis.grid(axis="y", color="#e1e0d9", linewidth=1)
        axis.tick_params(colors="#898781", length=0)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color("#c3c2b7")
    axes[0].set_ylabel("loss", color="#52514e")
    figure.tight_layout()
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(CHART_PATH, dpi=150)


def main() -> None:
    results = {}
    finals = {}
    for label, rate in RATES.items():
        losses, w, b = train(rate, STEPS)
        results[label] = losses
        finals[label] = (w, b)

    print(f"== 1. Loss at each step, for three learning rates ({STEPS} steps) ==")
    print(f"{'step':>4}" + "".join(f"{f'{label} ({rate})':>20}" for label, rate in RATES.items()))
    for step in [0, 1, 2, 3, 5, 10, 20, 30]:
        print(f"{step:>4}" + "".join(f"{results[label][step]:>20,.4f}" for label in RATES))

    print(f"\n== 2. Where each run ended (the best answer is w = 0.5, b = 0.2) ==")
    print(f"{'run':<12} {'w':>10} {'b':>10}")
    for label, (w, b) in finals.items():
        print(f"{label:<12} {w:>10.4f} {b:>10.4f}")

    print("\n== 3. The good rate, run for longer ==")
    print(f"{'steps':>5} {'w':>8} {'b':>8} {'loss':>12}")
    for steps in [5, 30, 100, 300, 1000]:
        losses, w, b = train(RATES["good"], steps)
        print(f"{steps:>5} {w:>8.4f} {b:>8.4f} {losses[-1]:>12.8f}")

    draw_chart(results)
    print(f"\nchart saved to {CHART_PATH}")


if __name__ == "__main__":
    main()