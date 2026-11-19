"""
RAGPromptBuilder: turn a folder of documents into grounded, cited LLM prompts.

Documents are split into overlapping chunks, ranked against the question, and
the best chunks are formatted into a prompt template that works with any LLM
(OpenAI, Claude, Ollama, ...).

Ranking is BM25 keyword search (standard library only) by default. Pass an
``embedder`` for semantic search, or combine both with ``retriever="hybrid"``.
"""

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

from .embeddings import Embedder, Vector, cosine_scores, embed_documents, embed_query, normalize

RETRIEVERS = ("bm25", "embedding", "hybrid")
RRF_K = 60  # reciprocal-rank-fusion constant (Cormack et al., 2009)

DEFAULT_EXTENSIONS = (".txt", ".md", ".rst", ".csv", ".json", ".html", ".py")

_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have how i if in into is it its
    of on or that the their there these this to was were what when where which
    who why will with you your""".split()
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n")
_HEADING_RE = re.compile(r"^(#{1,6}\s.*)$", re.MULTILINE)


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens with common English stopwords removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def estimate_tokens(text: str) -> int:
    """Rough LLM token estimate (~4 characters per token for English)."""
    return max(1, len(text) // 4) if text else 0


@dataclass
class Chunk:
    """A piece of a source document, optionally scored against a query."""

    text: str
    source: str
    index: int
    score: float = 0.0

    def __repr__(self) -> str:
        preview = self.text[:40].replace("\n", " ")
        return "Chunk(source=%r, index=%d, score=%.3f, text=%r...)" % (
            self.source, self.index, self.score, preview)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of about ``chunk_size`` characters, breaking on
    sentence/paragraph boundaries where possible. Consecutive chunks share
    about ``overlap`` characters of trailing sentences so context isn't lost
    at the seams.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = _HEADING_RE.sub(r"\n\1\n", text)  # Markdown headings stand alone
    sentences = [" ".join(s.split()) for s in _SENTENCE_RE.split(text) if s and s.strip()]
    # Hard-split any sentence that is longer than a whole chunk.
    pieces: List[str] = []
    for s in sentences:
        while len(s) > chunk_size:
            cut = s.rfind(" ", 0, chunk_size)
            cut = cut if cut > 0 else chunk_size
            pieces.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            pieces.append(s)

    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for piece in pieces:
        if current and _HEADING_RE.match(piece):
            # A heading starts a fresh chunk (no overlap across sections).
            chunks.append(" ".join(current))
            current, length = [], 0
        elif current and length + len(piece) + 1 > chunk_size:
            chunks.append(" ".join(current))
            # Carry trailing sentences forward as overlap.
            carried: List[str] = []
            carried_len = 0
            for prev in reversed(current):
                if carried_len + len(prev) + 1 > overlap:
                    break
                carried.insert(0, prev)
                carried_len += len(prev) + 1
            current, length = carried, carried_len
        current.append(piece)
        length += len(piece) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


class BM25Index:
    """Minimal Okapi BM25 ranking over a list of chunks."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._tfs = [Counter(tokenize(c.text)) for c in self.chunks]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        n = len(self.chunks)
        self._avgdl = (sum(self._lens) / n) if n else 0.0
        df: Counter = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query: str) -> List[float]:
        terms = tokenize(query)
        out = []
        for tf, dl in zip(self._tfs, self._lens):
            s = 0.0
            for term in terms:
                f = tf.get(term)
                if not f:
                    continue
                norm = 1 - self.b + self.b * (dl / self._avgdl if self._avgdl else 1)
                s += self._idf[term] * f * (self.k1 + 1) / (f + self.k1 * norm)
            out.append(s)
        return out


TEMPLATES: Dict[str, Dict[str, str]] = {
    "qa": {
        "system": (
            "You are a helpful assistant. Answer the question using ONLY the "
            "context provided. If the answer is not in the context, say "
            "\"I don't know based on the provided documents.\""
        ),
        "user": "Context:\n{context}\n\nQuestion: {question}\nAnswer:",
    },
    "citations": {
        "system": (
            "You are a precise research assistant. Answer using ONLY the "
            "numbered context passages. Cite every claim with its passage "
            "number in square brackets, e.g. [1] or [2][3]. If the context "
            "does not contain the answer, say so instead of guessing."
        ),
        "user": "Context:\n{context}\n\nQuestion: {question}\nAnswer (with citations):",
    },
    "summarize": {
        "system": (
            "You summarize documents faithfully. Use only the context given "
            "and do not add outside information."
        ),
        "user": (
            "Context:\n{context}\n\nSummarize what the context says about: "
            "{question}\nSummary:"
        ),
    },
    "strict": {
        "system": (
            "You answer strictly from the provided context. Quote the relevant "
            "sentence(s) verbatim, then give a one-line answer. If no passage "
            "answers the question, reply exactly: NOT_FOUND"
        ),
        "user": "Context:\n{context}\n\nQuestion: {question}",
    },
}


class RAGPromptBuilder:
    """
    Build retrieval-augmented prompts from your own documents.

    Example::

        rag = RAGPromptBuilder.from_folder("docs/")
        prompt = rag.build("What is AI?", top_k=3)
        # or, for chat APIs:
        messages = rag.build_messages("What is AI?")

        # semantic search (pip install "midmind[embeddings]")
        from midmind import SentenceTransformerEmbedder
        rag = RAGPromptBuilder.from_folder("docs/", embedder=SentenceTransformerEmbedder())

    ``retriever`` is ``"bm25"`` (keywords), ``"embedding"`` (meaning) or
    ``"hybrid"`` (both, merged with reciprocal rank fusion). It defaults to
    ``"hybrid"`` when an embedder is given and ``"bm25"`` otherwise.
    ``min_similarity`` drops embedding matches below that cosine similarity.
    """

    def __init__(self, chunks: Optional[Iterable[Chunk]] = None,
                 embedder: Optional[Embedder] = None,
                 retriever: Optional[str] = None,
                 min_similarity: Optional[float] = None):
        self.chunks: List[Chunk] = list(chunks or [])
        self.embedder = embedder
        self.retriever = retriever or ("hybrid" if embedder is not None else "bm25")
        self._check_retriever(self.retriever)
        self.min_similarity = min_similarity
        self._index: Optional[BM25Index] = None
        self._vectors: List[Vector] = []  # normalized, aligned with self.chunks

    def _check_retriever(self, retriever: str) -> None:
        if retriever not in RETRIEVERS:
            raise ValueError("retriever must be one of %s, got %r" % (RETRIEVERS, retriever))
        if retriever != "bm25" and self.embedder is None:
            raise ValueError("retriever=%r needs an embedder, e.g. "
                             "RAGPromptBuilder(embedder=SentenceTransformerEmbedder())"
                             % retriever)

    # ---- loading ---------------------------------------------------------

    @classmethod
    def from_folder(cls, path: str, chunk_size: int = 500, overlap: int = 50,
                    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
                    recursive: bool = True, **options) -> "RAGPromptBuilder":
        """
        Load every text file under ``path`` (a folder or a single file).
        ``options`` (``embedder``, ``retriever``, ``min_similarity``) go to the constructor.
        """
        builder = cls(**options)
        builder.add_folder(path, chunk_size, overlap, extensions, recursive)
        return builder

    @classmethod
    def from_texts(cls, texts: Union[Sequence[str], Dict[str, str]],
                   chunk_size: int = 500, overlap: int = 50,
                   **options) -> "RAGPromptBuilder":
        """Load in-memory strings: a list, or a ``{source_name: text}`` dict."""
        builder = cls(**options)
        items = texts.items() if isinstance(texts, dict) else (
            ("text_%d" % i, t) for i, t in enumerate(texts))
        for source, text in items:
            builder.add_text(text, source, chunk_size, overlap)
        return builder

    def add_text(self, text: str, source: str = "text",
                 chunk_size: int = 500, overlap: int = 50) -> "RAGPromptBuilder":
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            self.chunks.append(Chunk(piece, source, i))
        self._index = None
        return self

    def add_folder(self, path: str, chunk_size: int = 500, overlap: int = 50,
                   extensions: Sequence[str] = DEFAULT_EXTENSIONS,
                   recursive: bool = True) -> "RAGPromptBuilder":
        if not os.path.exists(path):
            raise FileNotFoundError("No such file or folder: %r" % path)
        exts = tuple(e.lower() for e in extensions)
        if os.path.isfile(path):
            files = [path]
        else:
            files = []
            for root, dirs, names in os.walk(path):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                for name in sorted(names):
                    if name.lower().endswith(exts) and not name.startswith("."):
                        files.append(os.path.join(root, name))
                if not recursive:
                    break
        base = path if os.path.isdir(path) else os.path.dirname(path)
        for fp in files:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                source = os.path.relpath(fp, base) if base else fp
                self.add_text(f.read(), source.replace(os.sep, "/"),
                              chunk_size, overlap)
        return self

    # ---- retrieval -------------------------------------------------------

    def _bm25_ranking(self, query: str) -> List[tuple]:
        if self._index is None:
            self._index = BM25Index(self.chunks)
        pairs = [(i, s) for i, s in enumerate(self._index.scores(query)) if s > 0]
        return sorted(pairs, key=lambda p: p[1], reverse=True)

    def embed_chunks(self) -> "RAGPromptBuilder":
        """Embed any chunks not embedded yet (runs automatically on first search)."""
        if self.embedder is None:
            raise ValueError("No embedder configured")
        todo = self.chunks[len(self._vectors):]
        if todo:
            vectors = embed_documents(self.embedder, [c.text for c in todo])
            self._vectors.extend(normalize(v) for v in vectors)
        return self

    def _embedding_ranking(self, query: str) -> List[tuple]:
        self.embed_chunks()
        sims = cosine_scores(embed_query(self.embedder, query), self._vectors)
        floor = self.min_similarity
        pairs = [(i, s) for i, s in enumerate(sims) if floor is None or s >= floor]
        return sorted(pairs, key=lambda p: p[1], reverse=True)

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.0,
                 retriever: Optional[str] = None) -> List[Chunk]:
        """
        Return the ``top_k`` most relevant chunks, best first.

        ``score`` is the BM25 score, the cosine similarity, or the fused
        reciprocal-rank score, depending on the retriever. ``retriever``
        overrides the builder's default for this call.
        """
        retriever = retriever or self.retriever
        self._check_retriever(retriever)
        if not self.chunks:
            return []
        if retriever == "bm25":
            ranking = self._bm25_ranking(query)
        elif retriever == "embedding":
            ranking = self._embedding_ranking(query)
        else:
            fused: Dict[int, float] = {}
            for ranked in (self._bm25_ranking(query), self._embedding_ranking(query)):
                for rank, (i, _) in enumerate(ranked):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            ranking = sorted(fused.items(), key=lambda p: p[1], reverse=True)
        results = []
        for i, score in ranking:
            if min_score and score < min_score:
                continue
            c = self.chunks[i]
            results.append(Chunk(c.text, c.source, c.index, score))
            if len(results) == top_k:
                break
        return results

    # ---- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        """Save chunks (and embeddings, if computed) to a JSON file."""
        if self.embedder is not None:
            self.embed_chunks()
        data = {
            "version": 1,
            "embedder": repr(self.embedder) if self.embedder is not None else None,
            "chunks": [{"text": c.text, "source": c.source, "index": c.index}
                       for c in self.chunks],
            "vectors": self._vectors,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str, embedder: Optional[Embedder] = None,
             **options) -> "RAGPromptBuilder":
        """
        Load a saved index. Pass the same ``embedder`` used when saving to
        reuse the stored embeddings without re-computing them.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = [Chunk(c["text"], c["source"], c["index"]) for c in data["chunks"]]
        builder = cls(chunks, embedder=embedder, **options)
        if embedder is not None and len(data.get("vectors", [])) == len(chunks):
            builder._vectors = data["vectors"]
        return builder

    # ---- prompt building -------------------------------------------------

    @staticmethod
    def format_context(chunks: Sequence[Chunk], max_tokens: Optional[int] = None) -> str:
        """Number and label chunks as ``[1] (source) text``, within a token budget."""
        parts: List[str] = []
        used = 0
        for i, c in enumerate(chunks, 1):
            part = "[%d] (%s) %s" % (i, c.source, c.text)
            cost = estimate_tokens(part)
            if max_tokens is not None and parts and used + cost > max_tokens:
                break
            parts.append(part)
            used += cost
        return "\n\n".join(parts) if parts else "(no relevant context found)"

    def _resolve(self, template: Union[str, Dict[str, str]]) -> Dict[str, str]:
        if isinstance(template, dict):
            return {"system": template.get("system", ""), "user": template["user"]}
        if template in TEMPLATES:
            return TEMPLATES[template]
        if "{context}" in template and "{question}" in template:
            return {"system": "", "user": template}
        raise ValueError(
            "Unknown template %r. Use one of %s, or a string containing "
            "{context} and {question}." % (template, sorted(TEMPLATES)))

    def build_messages(self, question: str, top_k: int = 3,
                       template: Union[str, Dict[str, str]] = "citations",
                       max_context_tokens: Optional[int] = None) -> List[Dict[str, str]]:
        """Chat-style ``[{"role": ..., "content": ...}]`` messages for any chat API."""
        tpl = self._resolve(template)
        context = self.format_context(self.retrieve(question, top_k), max_context_tokens)
        user = tpl["user"].format(context=context, question=question)
        messages = []
        if tpl["system"]:
            messages.append({"role": "system", "content": tpl["system"]})
        messages.append({"role": "user", "content": user})
        return messages

    def build(self, question: str, top_k: int = 3,
              template: Union[str, Dict[str, str]] = "citations",
              max_context_tokens: Optional[int] = None) -> str:
        """A single prompt string (system + user), ready to send to an LLM."""
        messages = self.build_messages(question, top_k, template, max_context_tokens)
        return "\n\n".join(m["content"] for m in messages)

    def ask(self, question: str, llm: Callable[[str], str], **kwargs) -> str:
        """Build the prompt and pass it to ``llm`` (any ``str -> str`` callable)."""
        return llm(self.build(question, **kwargs))

    def __len__(self) -> int:
        return len(self.chunks)
