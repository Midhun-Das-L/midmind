import pytest

from midmind import RAGPromptBuilder, autoragx, chunk_text

DOCS = {
    "ai.txt": "Artificial Intelligence is the simulation of human intelligence by machines.",
    "cooking.txt": "To bake bread, mix flour, water, salt and yeast, then knead the dough.",
    "space.txt": "Mars is the fourth planet from the Sun and has two small moons.",
}


def test_chunk_text_respects_size_and_overlap():
    text = " ".join("Sentence number %d is here." % i for i in range(50))
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # Overlap: the last sentence of one chunk starts the next.
    assert chunks[0].split(". ")[-1].rstrip(".") in chunks[1]


def test_chunk_text_splits_very_long_sentence():
    chunks = chunk_text("word " * 300, chunk_size=50, overlap=0)
    assert all(len(c) <= 50 for c in chunks)


def test_chunk_text_validates_args():
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=10, overlap=10)


def test_retrieve_ranks_relevant_chunk_first():
    rag = RAGPromptBuilder.from_texts(DOCS)
    top = rag.retrieve("How do I bake bread?", top_k=2)
    assert top[0].source == "cooking.txt"
    assert top[0].score > 0


def test_retrieve_skips_irrelevant():
    rag = RAGPromptBuilder.from_texts(DOCS)
    assert rag.retrieve("quantum chromodynamics") == []


def test_build_contains_context_question_and_citation_labels():
    rag = RAGPromptBuilder.from_texts(DOCS)
    prompt = rag.build("What is artificial intelligence?", top_k=1)
    assert "[1] (ai.txt)" in prompt
    assert "Question: What is artificial intelligence?" in prompt
    assert "Mars" not in prompt


def test_build_messages_shape():
    rag = RAGPromptBuilder.from_texts(DOCS)
    msgs = rag.build_messages("Mars moons", template="qa")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "Mars" in msgs[1]["content"]


def test_custom_template_and_unknown_template():
    rag = RAGPromptBuilder.from_texts(DOCS)
    out = rag.build("Mars", template="CTX={context} Q={question}")
    assert out.startswith("CTX=[1] (space.txt)") and out.endswith("Q=Mars")
    with pytest.raises(ValueError):
        rag.build("Mars", template="nope")


def test_no_match_has_placeholder_context():
    rag = RAGPromptBuilder.from_texts(DOCS)
    assert "(no relevant context found)" in rag.build("zzz")


def test_max_context_tokens_limits_chunks():
    rag = RAGPromptBuilder.from_texts({"a": "alpha beta. " * 40, "b": "alpha gamma. " * 40},
                                      chunk_size=100, overlap=0)
    user = rag.build_messages("alpha", top_k=10, max_context_tokens=30)[-1]["content"]
    assert "[1] (" in user and "[2] (" not in user


def test_from_folder_recursive_and_filters(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("Python is a programming language.")
    (tmp_path / "sub" / "b.md").write_text("Rust is a systems language.")
    (tmp_path / "image.png").write_bytes(b"\x89PNG binary")
    rag = RAGPromptBuilder.from_folder(str(tmp_path))
    assert sorted({c.source for c in rag.chunks}) == ["a.txt", "sub/b.md"]
    assert rag.retrieve("Rust")[0].source == "sub/b.md"


def test_from_folder_missing_path():
    with pytest.raises(FileNotFoundError):
        RAGPromptBuilder.from_folder("/does/not/exist")


def test_autoragx_returns_prompt_or_llm_answer(tmp_path):
    (tmp_path / "ai.txt").write_text(DOCS["ai.txt"])
    prompt = autoragx(str(tmp_path), "What is AI intelligence?")
    assert "[1] (ai.txt)" in prompt
    assert autoragx(str(tmp_path), "intelligence", llm=lambda p: "ANSWER") == "ANSWER"


def test_markdown_headings_are_their_own_chunk_boundary():
    chunks = chunk_text("# Title\nFirst line\ncontinues here.\n## Next\nMore.", 200, 0)
    assert chunks == ["# Title First line continues here.", "## Next More."]


def test_example_rag_runs():
    import subprocess, sys, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=root)
    result = subprocess.run([sys.executable, os.path.join(root, "examples", "rag_example.py")],
                            capture_output=True, text=True, cwd=root, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[2] (billing.md)" in result.stdout or "[1] (billing.md)" in result.stdout
