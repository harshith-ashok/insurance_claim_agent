import json
import os

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.jsonl")


def load_golden_set(path: str = GOLDEN_SET_PATH):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
