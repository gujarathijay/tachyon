"""Phase 2, step 1: build a tokenizer by hand with BPE (byte pair encoding).

Run from the repo root:

    uv run python -m tachyon.phase2.step1_bpe
"""

from collections import Counter

# A tiny "training text": each word and how many times it appears.
WORD_COUNTS = {
    "select": 6,
    "selected": 2,
    "count": 5,
    "counts": 2,
    "user": 4,
    "users": 4,
    "order": 3,
    "orders": 3,
}
NUM_MERGES = 10
NEW_WORDS = ["selects", "recount", "reorders", "customer"]  # words BPE never saw


def count_pairs(pieces: dict[str, list[str]]) -> Counter:
    """Count every pair of side-by-side pieces, weighted by how often the word appears."""
    pairs = Counter()
    for word, count in WORD_COUNTS.items():
        parts = pieces[word]
        # print(parts)
        for left, right in zip(parts, parts[1:]):
            pairs[(left, right)] += count
            # print(pairs)
    return pairs


def merge_pair(parts: list[str], pair: tuple[str, str]) -> list[str]:
    """Join every place where the two pieces of `pair` sit side by side."""
    merged = []
    i = 0
    # print("parts1--------", parts)
    while i < len(parts):
        if i < len(parts) - 1 and (parts[i], parts[i + 1]) == pair:
            merged.append(parts[i] + parts[i + 1])
            
            i += 2
        else:
            merged.append(parts[i])
            i += 1
    # print(merged)
    return merged


def total_tokens(pieces: dict[str, list[str]]) -> int:
    """How many tokens the whole training text takes right now."""
    # print("pieces", pieces)
    # print("sum", sum(len(pieces[word]) * count for word, count in WORD_COUNTS.items()))
    return sum(len(pieces[word]) * count for word, count in WORD_COUNTS.items())


def train_bpe(num_merges: int) -> tuple[list[str], list[tuple[str, str]]]:
    """Learn merges. Returns the final vocabulary and the merges in the order learned."""
    pieces = {word: list(word) for word in WORD_COUNTS}  # start: every word as single letters
    vocab = sorted({letter for word in WORD_COUNTS for letter in word})
    print(vocab)
    merges = []

    print(f"start: {len(vocab)} tokens in vocab {vocab}")
    print(f"       the training text takes {total_tokens(pieces)} tokens\n")

    print(f"{'merge':>5}  {'pair':<16} {'count':>5}  {'vocab size':>10}  {'text tokens':>11}  how 'selected' and 'users' look now")

    for step in range(1, num_merges + 1):
        pairs = count_pairs(pieces)
        # most frequent pair wins; a tie goes to the pair that comes first alphabetically
        # print("pairs1---------",pairs)
        best_pair, best_count = min(pairs.items(), key=lambda item: (-item[1], item[0]))
        # print("piecessssssss", pieces)

        for word, parts in pieces.items():
            # print("pieces-----", word, pieces[word])
            pieces[word]=merge_pair(parts, best_pair)
        vocab.append(best_pair[0] + best_pair[1])
        merges.append(best_pair)
        shown = f"{pieces['selected']}  {pieces['users']}"
        pair_text = f"{best_pair[0]} + {best_pair[1]}"
        print(f"{step:>5}  {pair_text:<16} {best_count:>5}  {len(vocab):>10}  {total_tokens(pieces):>11}  {shown}")

    return vocab, merges


def encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Tokenize a new word: start from letters, then apply the learned merges in order."""
    parts = list(word)
    # print(parts)
    for pair in merges:
        parts = merge_pair(parts, pair)
        # print(parts)
    return parts


def main() -> None:
    print("== 1.  ")
    vocab, merges = train_bpe(NUM_MERGES)

    print("\n== 2. The final vocabulary: every token gets an ID number ==")
    for token_id, token in enumerate(vocab):
        print(f"  {token_id:>2}: {token!r}")

    # print(merges)
    print("\n== 3. Tokenize words BPE never saw ==")
    for word in NEW_WORDS:
        parts = encode(word, merges)
        # print("word", word, "parts", parts)
        ids = [vocab.index(part) if part in vocab else None for part in parts]
        print(f"  {word:<10} -> {parts}  ids {ids}")


if __name__ == "__main__":
    main()