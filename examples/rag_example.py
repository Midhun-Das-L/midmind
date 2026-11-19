"""
Example: a support-bot RAG over a small product knowledge base.

    python examples/rag_example.py                 # keyword search (BM25), no extra installs
    python examples/rag_example.py --embeddings    # hybrid search, needs: pip install "midmind[embeddings]"
    python examples/rag_example.py --ask           # also send the prompt to Claude (needs anthropic + ANTHROPIC_API_KEY)

Each question lists the file that should be retrieved, so the script doubles as a
smoke test: it exits with status 1 if any question retrieves the wrong file first.
"""

import argparse
import os
import sys

from midmind import RAGPromptBuilder

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

# (question, expected source, needs_meaning). Questions marked True are
# paraphrases that share almost no words with the docs ("money back" vs
# "refund"), so keyword search misses them; only --embeddings must get them.
QUESTIONS = [
    ("How do I reset my password?", "account.md", False),
    ("Can I export my notes as PDF?", "features.md", False),
    ("How much does the Team plan cost?", "billing.md", False),
    ("How do I turn on two-factor authentication?", "account.md", False),
    ("Does the app work without internet?", "features.md", True),
    ("Can I get my money back if I paid for a year?", "billing.md", True),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", action="store_true", help="use hybrid BM25 + embedding search")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--ask", action="store_true", help="send the last prompt to Claude")
    args = parser.parse_args()

    embedder = None
    if args.embeddings:
        from midmind import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder(args.model)

    rag = RAGPromptBuilder.from_folder(DOCS, chunk_size=300, overlap=60, embedder=embedder)
    print("Loaded %d chunks from %s using %s retrieval\n" % (len(rag), DOCS, rag.retriever))

    misses = required_misses = 0
    for question, expected, needs_meaning in QUESTIONS:
        hits = rag.retrieve(question, top_k=2)
        got = hits[0].source if hits else None
        ok = got == expected
        misses += not ok
        required = embedder is not None or not needs_meaning
        required_misses += required and not ok
        label = "PASS" if ok else ("MISS" if required else "MISS (paraphrase: try --embeddings)")
        print("%s %s" % (label, question))
        for c in hits:
            print("       %.3f  %-12s %s..." % (c.score, c.source, c.text[:60].replace("\n", " ")))
        if not ok:
            print("       expected %s, got %s" % (expected, got))

    question = QUESTIONS[-1][0]  # the "money back" paraphrase
    messages = rag.build_messages(question, top_k=2, template="citations", max_context_tokens=400)
    print("\n--- prompt for: %s ---" % question)
    for m in messages:
        print("[%s]\n%s\n" % (m["role"], m["content"]))

    if args.ask:
        import anthropic
        reply = anthropic.Anthropic().messages.create(
            model="claude-sonnet-5-5", max_tokens=400,
            system=messages[0]["content"], messages=messages[1:])
        print("--- Claude ---\n" + reply.content[0].text)

    print("%d/%d questions retrieved the right document" % (len(QUESTIONS) - misses, len(QUESTIONS)))
    sys.exit(1 if required_misses else 0)


if __name__ == "__main__":
    main()
