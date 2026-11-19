"""
Embedding backends for semantic retrieval.

An *embedder* is anything callable as ``embedder(list_of_texts) -> list_of_vectors``.
Optionally it can also have an ``embed_query(text) -> vector`` method, for models
that encode questions differently from documents (e5, bge, Voyage, Cohere, ...).

So any API works with a one-line wrapper, e.g. OpenAI::

    from openai import OpenAI
    client = OpenAI()
    def openai_embedder(texts):
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [d.embedding for d in resp.data]
"""

import math
from typing import Any, Callable, List, Optional, Sequence

Vector = List[float]
Embedder = Callable[[List[str]], Sequence[Sequence[float]]]

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    """
    Local embeddings with `sentence-transformers <https://sbert.net>`_.

    Install with ``pip install "midmind[embeddings]"``. The model is downloaded
    from Hugging Face on first use and cached locally afterwards.

    ``query_prefix`` / ``document_prefix`` support instruction-tuned models,
    e.g. ``query_prefix="query: ", document_prefix="passage: "`` for e5.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32,
                 query_prefix: str = "", document_prefix: str = "",
                 device: Optional[str] = None, **model_kwargs: Any):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(
                "SentenceTransformerEmbedder needs sentence-transformers. "
                "Install it with: pip install \"midmind[embeddings]\""
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.model = SentenceTransformer(model_name, device=device, **model_kwargs)

    def _encode(self, texts: List[str]) -> List[Vector]:
        vectors = self.model.encode(list(texts), batch_size=self.batch_size,
                                    normalize_embeddings=True,
                                    show_progress_bar=False)
        return vectors.tolist()

    def __call__(self, texts: List[str]) -> List[Vector]:
        return self._encode([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> Vector:
        return self._encode([self.query_prefix + text])[0]

    def __repr__(self) -> str:
        return "SentenceTransformerEmbedder(%r)" % self.model_name


def embed_documents(embedder: Embedder, texts: List[str]) -> List[Vector]:
    vectors = [list(map(float, v)) for v in embedder(texts)]
    if len(vectors) != len(texts):
        raise ValueError("Embedder returned %d vectors for %d texts"
                         % (len(vectors), len(texts)))
    return vectors


def embed_query(embedder: Embedder, text: str) -> Vector:
    fn = getattr(embedder, "embed_query", None)
    if callable(fn):
        return list(map(float, fn(text)))
    return embed_documents(embedder, [text])[0]


def normalize(vector: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else list(vector)


def cosine_scores(query: Sequence[float], matrix: Sequence[Sequence[float]]) -> List[float]:
    """Cosine similarity of ``query`` against each row (rows pre-normalized)."""
    q = normalize(query)
    try:
        import numpy as np
    except ImportError:
        return [sum(a * b for a, b in zip(q, row)) for row in matrix]
    if not len(matrix):
        return []
    return (np.asarray(matrix, dtype=float) @ np.asarray(q, dtype=float)).tolist()
