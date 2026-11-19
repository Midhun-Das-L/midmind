import builtins

import pytest

from midmind import RAGPromptBuilder, SentenceTransformerEmbedder, autoragx
from midmind.embeddings import cosine_scores

# A tiny "semantic" embedder: words that mean the same thing share a dimension,
# so "AI" lands next to "artificial intelligence" even with no shared keyword.
CONCEPTS = {
    "ai": 0, "artificial": 0, "intelligence": 0, "machine": 0, "machines": 0,
    "bread": 1, "bake": 1, "baking": 1, "dough": 1, "loaf": 1, "flour": 1,
    "mars": 2, "planet": 2, "red": 2, "moons": 2, "sun": 2,
}


class ConceptEmbedder:
    def __init__(self):
        self.calls = 0
        self.texts_embedded = 0

    def __call__(self, texts):
        self.calls += 1
        self.texts_embedded += len(texts)
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(text):
        v = [0.0, 0.0, 0.0, 0.01]
        for word in text.lower().replace("?", " ").replace(".", " ").split():
            if word in CONCEPTS:
                v[CONCEPTS[word]] += 1.0
        return v


DOCS = {
    "ai.txt": "Artificial Intelligence is the simulation of human intelligence by machines.",
    "cooking.txt": "To bake bread, mix flour, water, salt and yeast, then knead the dough.",
    "space.txt": "Mars is the fourth planet from the Sun and has two small moons.",
}


def test_bm25_misses_synonym_but_embedding_finds_it():
    keyword = RAGPromptBuilder.from_texts(DOCS)
    assert keyword.retrieve("What is AI?") == []

    semantic = RAGPromptBuilder.from_texts(DOCS, embedder=ConceptEmbedder(),
                                           retriever="embedding")
    top = semantic.retrieve("What is AI?", top_k=1)
    assert top[0].source == "ai.txt"
    assert top[0].score == pytest.approx(1.0, abs=0.01)


def test_hybrid_is_default_with_embedder_and_ranks_relevant_first():
    rag = RAGPromptBuilder.from_texts(DOCS, embedder=ConceptEmbedder())
    assert rag.retriever == "hybrid"
    assert rag.retrieve("How do I make a loaf?")[0].source == "cooking.txt"
    # Matches both keyword and meaning -> highest fused score
    top = rag.retrieve("bread dough", top_k=3)
    assert top[0].source == "cooking.txt" and top[0].score > top[1].score


def test_per_call_retriever_override():
    rag = RAGPromptBuilder.from_texts(DOCS, embedder=ConceptEmbedder())
    assert rag.retrieve("AI", retriever="bm25") == []
    assert rag.retrieve("AI", retriever="embedding")[0].source == "ai.txt"


def test_min_similarity_filters_weak_matches():
    rag = RAGPromptBuilder.from_texts(DOCS, embedder=ConceptEmbedder(),
                                      retriever="embedding", min_similarity=0.5)
    results = rag.retrieve("red planet", top_k=3)
    assert [c.source for c in results] == ["space.txt"]


def test_chunks_are_embedded_once_and_incrementally():
    emb = ConceptEmbedder()
    rag = RAGPromptBuilder.from_texts(DOCS, embedder=emb)
    rag.retrieve("AI")
    rag.retrieve("bread")
    doc_texts = emb.texts_embedded - 2  # minus the two queries
    assert doc_texts == 3
    rag.add_text("A loaf of sourdough bread.", "sourdough.txt")
    rag.retrieve("bread")
    assert emb.texts_embedded - 3 == 4  # only the new chunk was embedded


def test_embed_query_method_is_used_for_queries():
    class Asymmetric(ConceptEmbedder):
        def embed_query(self, text):
            self.query_seen = text
            return self._vec(text)

    emb = Asymmetric()
    rag = RAGPromptBuilder.from_texts(DOCS, embedder=emb, retriever="embedding")
    rag.retrieve("Mars?")
    assert emb.query_seen == "Mars?"
    assert emb.texts_embedded == 3  # query did not go through __call__


def test_retriever_validation():
    with pytest.raises(ValueError):
        RAGPromptBuilder(retriever="embedding")  # no embedder
    with pytest.raises(ValueError):
        RAGPromptBuilder(embedder=ConceptEmbedder(), retriever="vector")


def test_embedder_returning_wrong_count_errors():
    rag = RAGPromptBuilder.from_texts(DOCS, embedder=lambda texts: [[1.0]],
                                      retriever="embedding")
    with pytest.raises(ValueError):
        rag.retrieve("x")


def test_save_and_load_reuses_embeddings(tmp_path):
    emb = ConceptEmbedder()
    rag = RAGPromptBuilder.from_texts(DOCS, embedder=emb)
    path = str(tmp_path / "index.json")
    rag.save(path)

    emb2 = ConceptEmbedder()
    loaded = RAGPromptBuilder.load(path, embedder=emb2)
    assert len(loaded) == 3
    assert loaded.retrieve("What is AI?", top_k=1)[0].source == "ai.txt"
    assert emb2.texts_embedded == 1  # only the query; chunk vectors came from disk

    keyword_only = RAGPromptBuilder.load(path)
    assert keyword_only.retriever == "bm25"
    assert keyword_only.retrieve("Mars")[0].source == "space.txt"


def test_autoragx_with_embedder(tmp_path):
    for name, text in DOCS.items():
        (tmp_path / name).write_text(text)
    prompt = autoragx(str(tmp_path), "Tell me about AI", embedder=ConceptEmbedder(), top_k=1)
    assert "[1] (ai.txt)" in prompt


def test_cosine_scores():
    rows = [[1.0, 0.0], [0.0, 1.0]]
    assert cosine_scores([2.0, 0.0], rows) == pytest.approx([1.0, 0.0])


def test_missing_sentence_transformers_gives_install_hint(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"midmind\[embeddings\]"):
        SentenceTransformerEmbedder()
