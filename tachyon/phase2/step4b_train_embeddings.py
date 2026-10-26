"""Phase 2, step 4b: train an embedding table on many SQL sentences, then measure who ends up close.

Run from the repo root:

    uv run python -m tachyon.phase2.step4b_train_embeddings
"""

import itertools
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to a file, never open a window
import matplotlib.pyplot as plt
import torch
from torch import nn

COLUMNS = ["name", "email", "age"]
TABLES = ["users", "orders"]
VOCAB = ["SELECT", "FROM", "WHERE", "=", "1", ";"] + COLUMNS + TABLES
NUMBERS_PER_TOKEN = 2
LEARNING_RATE = 1.0
STEPS = 2000
SEED = 0
PAIRS_TO_COMPARE = [
    ("name", "email"),
    ("name", "age"),
    ("users", "orders"),
    ("SELECT", "WHERE"),
    ("name", "users"),
    ("FROM", "users"),
]
CHART_PATH = Path("docs/img/phase2_embeddings.png")


def make_sentences() -> list[str]:
    """24 tiny SQL queries: every column with every table, with and without a WHERE."""
    sentences = []
    for column, table in itertools.product(COLUMNS, TABLES):
        sentences.append(f"SELECT {column} FROM {table} ;")
        for where_column in COLUMNS:
            sentences.append(f"SELECT {column} FROM {table} WHERE {where_column} = 1 ;")
    return sentences


def make_pairs(sentences: list[str]) -> list[tuple[int, int]]:
    """Every (token, the token right after it) pair, as IDs."""
    pairs = []
    for sentence in sentences:
        tokens = sentence.split()
        for current, following in zip(tokens, tokens[1:]):
            pairs.append((VOCAB.index(current), VOCAB.index(following)))
    return pairs


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Direction match between two rows: 1 same way, 0 unrelated, -1 opposite."""
    return (a @ b / (a.norm() * b.norm())).item()


def draw_chart(before: torch.Tensor, after: torch.Tensor) -> None:
    """Each token's row as a point, before and after training."""
    groups = {
        "column names": (COLUMNS, "#2a78d6"),
        "table names": (TABLES, "#eb6834"),
        "SELECT, WHERE": (["SELECT", "WHERE"], "#1baf7a"),
        "others": (["FROM", "=", "1", ";"], "#898781"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#fcfcfb")
    for axis, table, title in [(axes[0], before, "before training"), (axes[1], after, "after training")]:
        for label, (tokens, color) in groups.items():
            points = table[[VOCAB.index(t) for t in tokens]]
            axis.scatter(points[:, 0], points[:, 1], s=60, color=color, label=label,
                         edgecolors="#fcfcfb", linewidths=2, zorder=3)
        labels = []  # tokens sitting on (almost) the same spot share one label
        for token, (x, y) in zip(VOCAB, table.tolist()):
            for spot in labels:
                if math.dist(spot["xy"], (x, y)) < 0.2:
                    spot["names"].append(token)
                    break
            else:
                labels.append({"xy": (x, y), "names": [token]})
        for spot in labels:
            axis.annotate(", ".join(spot["names"]), spot["xy"], textcoords="offset points",
                          xytext=(6, 6), color="#0b0b0b")
        axis.set_title(title, loc="left", color="#0b0b0b")
        axis.set_xlabel("1st number", color="#52514e")
        axis.set_ylabel("2nd number", color="#52514e")
        axis.set_facecolor("#fcfcfb")
        axis.grid(color="#e1e0d9", linewidth=1)
        axis.tick_params(colors="#898781", length=0)
        for side in axis.spines.values():
            side.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", frameon=False, ncol=4)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(CHART_PATH, dpi=150)


def main() -> None:
    sentences = make_sentences()
    pairs = make_pairs(sentences)
    inputs = torch.tensor([p[0] for p in pairs])
    answers = torch.tensor([p[1] for p in pairs])

    print("== 1. The training data ==")
    print(f"  {len(sentences)} sentences -> {len(pairs)} (token, next token) pairs")
    first = sentences[0].split()
    print(f"  e.g. '{sentences[0]}' gives: " + ", ".join(f"{a}->{b}" for a, b in zip(first, first[1:])))

    print("\n== 2. What really follows each token in the text ==")
    follows = defaultdict(Counter)
    for current, following in pairs:
        follows[current][following] += 1
    for token_id in sorted(follows, key=lambda i: VOCAB[i]):
        counts = follows[token_id]
        total = sum(counts.values())
        shares = ", ".join(f"{VOCAB[n]} {c / total:.2f}" for n, c in counts.most_common())
        print(f"  after {VOCAB[token_id]:<7} -> {shares}")
    best_loss = -sum(math.log(follows[a][b] / sum(follows[a].values())) for a, b in pairs) / len(pairs)
    print(f"  best loss any model could reach on this text: {best_loss:.4f}")

    print(f"\n== 3. Train: {STEPS} steps, all {len(pairs)} pairs every step, learning rate {LEARNING_RATE} ==")
    torch.manual_seed(SEED)
    embedding = nn.Embedding(len(VOCAB), NUMBERS_PER_TOKEN)
    output = nn.Linear(NUMBERS_PER_TOKEN, len(VOCAB), bias=False)
    before = embedding.weight.detach().clone()
    optimizer = torch.optim.SGD(list(embedding.parameters()) + list(output.parameters()), lr=LEARNING_RATE)
    for step in range(STEPS + 1):
        scores = output(embedding(inputs))  # one row of 11 scores per pair
        loss = nn.functional.cross_entropy(scores, answers)  # average over all pairs
        if step in (0, 10, 50, 100, 500, 1000, 2000):
            print(f"  step {step:>4}: loss {loss.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    after = embedding.weight.detach().clone()  # snapshot: plain numbers, no tracking, own copy

    print("\n== 4. What the model predicts now, next to the real shares ==")
    with torch.no_grad():
        probs = torch.softmax(output(after), dim=1)
    for token in ["SELECT", "name", "FROM", "users", "="]:
        top = probs[VOCAB.index(token)].topk(3)
        guess = ", ".join(f"{VOCAB[i]} {p:.2f}" for p, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"  after {token:<7} -> {guess}")

    print("\n== 5. Each row, before and after ==")
    print(f"  {'token':<8} {'before':>18} {'after':>18}")
    for i, token in enumerate(VOCAB):
        b, a = before[i].tolist(), after[i].tolist()
        print(f"  {token:<8} ({b[0]:>6.2f}, {b[1]:>6.2f})   ({a[0]:>6.2f}, {a[1]:>6.2f})")

    print("\n== 6. Cosine similarity: do two rows point the same way? ==")
    print(f"  {'pair':<18} {'before':>7} {'after':>7}")
    for x, y in PAIRS_TO_COMPARE:
        bx, by = before[VOCAB.index(x)], before[VOCAB.index(y)]
        ax, ay = after[VOCAB.index(x)], after[VOCAB.index(y)]
        print(f"  {x + ' / ' + y:<18} {cosine(bx, by):>7.3f} {cosine(ax, ay):>7.3f}")

    draw_chart(before, after)
    print(f"\nchart saved to {CHART_PATH}")


if __name__ == "__main__":
    main()