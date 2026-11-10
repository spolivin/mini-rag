"""Shared test helpers: deterministic fake models and sample documents."""

import hashlib
import uuid

import numpy as np

from rag import NoRelevantResultsError

PARAGRAPHS_A = [
    "Cats are small carnivorous mammals kept as pets.",
    "The Pacific Ocean is the largest body of water on Earth.",
    "Python is a popular programming language for data work.",
]
PARAGRAPHS_B = [
    "Mount Everest is the highest mountain above sea level.",
    "Bread is made by baking dough from flour and water.",
]


class FakeSentenceTransformer:
    """Stands in for SentenceTransformer: equal texts get equal 384-dim vectors."""

    def __init__(self, name):
        pass

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            vectors.append(np.random.default_rng(seed).standard_normal(384))
        return np.stack(vectors).astype(np.float32)


class FakeReRanker:
    """Stands in for ReRanker: keeps the incoming order and gives positive scores."""

    def __init__(self, model_name):
        pass

    def __call__(self, query, candidates, top_k):
        if not candidates:
            raise NoRelevantResultsError("No candidate chunks to rank.")
        ranked = list(candidates)[:top_k]
        return ranked, [1.0 - 0.01 * i for i in range(len(ranked))]


class FakeLLMWrapper:
    """Stands in for LLMWrapper: no API call, a deterministic reply from the best chunk."""

    def __init__(self, model_name, textgen_params):
        pass

    def generate(self, user_query, context):
        snippet = " ".join(context[0].split())[:200] if context else ""
        return f'[Simulated answer] Based on {len(context)} passage(s): "{snippet}"'


def write_doc(tmp_path, paragraphs):
    """Writes a text document with a unique name and content (so its hash is unique too)."""
    tag = uuid.uuid4().hex[:8]
    path = tmp_path / f"doc{tag}.txt"
    path.write_text("\n\n".join(paragraphs) + f"\n\nref {tag}")
    return path
