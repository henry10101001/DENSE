"""Utility to split combined_dataset.jsonl into train/test jsonl files.

Assumes the input file lives in ``synthetic_data_converted/combined_dataset.jsonl``.
Outputs ``train.jsonl`` and ``test.jsonl`` in the same directory with a 10% random test split.
"""

import json
import random
from pathlib import Path

INPUT_FILE = Path("synthetic_data_converted/combined_dataset.jsonl")
TRAIN_FILE = INPUT_FILE.parent / "train.jsonl"
TEST_FILE = INPUT_FILE.parent / "test.jsonl"
TEST_RATIO = 0.1


def main():
    data = []
    with INPUT_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    random.shuffle(data)
    split_idx = max(1, int(len(data) * TEST_RATIO))
    test_data = data[:split_idx]
    train_data = data[split_idx:]

    INPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_FILE.open("w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with TEST_FILE.open("w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    print(f"wrote {len(train_data)} train samples to {TRAIN_FILE}")
    print(f"wrote {len(test_data)} test samples to {TEST_FILE}")


if __name__ == "__main__":
    main()
