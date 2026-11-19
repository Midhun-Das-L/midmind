# MidMind

**Turn a folder of documents into a grounded, cited LLM prompt — in one line, with zero dependencies.**

MidMind does the "R" in RAG for you: it loads your files, splits them into overlapping chunks,
ranks them against the question with BM25, and formats the best ones into a prompt that tells
the model to answer only from your documents and cite its sources. Send the result to any LLM.
Add an embedding model when you need search by meaning, not just keywords.

```bash
pip install midmind
```

## Quickstart

```python
from midmind import autoragx

prompt = autoragx("docs/", "What is artificial intelligence?")
print(prompt)
```

```text
You are a precise research assistant. Answer using ONLY the numbered context passages.
Cite every claim with its passage number in square brackets, e.g. [1] or [2][3]. ...

Context:
[1] (sample.txt) Artificial Intelligence is the simulation of human intelligence by machines.

Question: What is artificial intelligence?
Answer (with citations):
```

Pass any `str -> str` function as `llm=` to get an answer instead of a prompt:

```python
answer = autoragx("docs/", "What is AI?", llm=my_llm)
```

## RAGPromptBuilder

```python
from midmind import RAGPromptBuilder

rag = RAGPromptBuilder.from_folder("docs/", chunk_size=500, overlap=50)
# or: RAGPromptBuilder.from_texts({"faq.md": faq_text, "notes.txt": notes})

rag.retrieve("refund policy", top_k=3)        # -> [Chunk(source, index, score, text), ...]
rag.build("What is the refund policy?")        # -> one prompt string
rag.build_messages("What is the refund policy?")  # -> [{"role": "system", ...}, {"role": "user", ...}]
```

| Option | Default | What it does |
| --- | --- | --- |
| `top_k` | `3` | Number of chunks put into the prompt |
| `template` | `"citations"` | `"qa"`, `"citations"`, `"summarize"`, `"strict"`, or your own string with `{context}` and `{question}` |
| `max_context_tokens` | `None` | Stop adding chunks once this (estimated) token budget is reached |
| `chunk_size` / `overlap` | `500` / `50` | Chunk length and overlap in characters, split on sentence boundaries |
| `extensions` | `.txt .md .rst .csv .json .html .py` | File types loaded by `from_folder` (searches subfolders) |
| `embedder` | `None` | Embedding model for semantic search (see below) |
| `retriever` | `"bm25"`, or `"hybrid"` with an embedder | `"bm25"`, `"embedding"` or `"hybrid"`; can also be passed per `retrieve()` call |
| `min_similarity` | `None` | Drop embedding matches below this cosine similarity |

Custom template:

```python
rag.build("Who wrote it?", template="Use these notes:\n{context}\n\nQ: {question}\nA:")
```

## Semantic search with embeddings

Keyword search can't connect "Can I get my money back?" to a doc that says "refund".
Embeddings can:

```bash
pip install "midmind[embeddings]"
```

```python
from midmind import RAGPromptBuilder, SentenceTransformerEmbedder

rag = RAGPromptBuilder.from_folder("docs/", embedder=SentenceTransformerEmbedder())
rag.build("Can I get my money back?")   # hybrid: BM25 + embeddings, merged by rank
```

`retriever="hybrid"` (the default with an embedder) combines both rankings with
reciprocal rank fusion, so exact terms like error codes and product names still match.
Use `retriever="embedding"` for meaning only.

**Any embedding model works.** An embedder is just a function from a list of texts to a list
of vectors. Give it an `embed_query(text)` method if the model encodes questions differently.

```python
from openai import OpenAI
client = OpenAI()

def openai_embedder(texts):
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]

rag = RAGPromptBuilder.from_folder("docs/", embedder=openai_embedder)
```

For instruction-tuned models such as e5:
`SentenceTransformerEmbedder("intfloat/e5-small-v2", query_prefix="query: ", document_prefix="passage: ")`.

**Save the index** so documents aren't re-embedded on every run:

```python
rag.save("index.json")
rag = RAGPromptBuilder.load("index.json", embedder=SentenceTransformerEmbedder())
```

## Example

[`examples/rag_example.py`](examples/rag_example.py) is a support bot over a small product
knowledge base in [`examples/docs/`](examples/docs/):

```bash
python examples/rag_example.py               # keyword search
python examples/rag_example.py --embeddings  # hybrid search
python examples/rag_example.py --ask         # also ask Claude (needs ANTHROPIC_API_KEY)
```

## Use with your LLM

**Claude**

```python
import anthropic
client = anthropic.Anthropic()
system, user = rag.build_messages("What is AI?")
reply = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024,
                               system=system["content"], messages=[user])
print(reply.content[0].text)
```

**OpenAI**

```python
from openai import OpenAI
reply = OpenAI().chat.completions.create(model="gpt-4o-mini",
                                         messages=rag.build_messages("What is AI?"))
print(reply.choices[0].message.content)
```

**Ollama (local)**

```python
import ollama
reply = ollama.chat(model="llama3", messages=rag.build_messages("What is AI?"))
print(reply["message"]["content"])
```

## Prompt toolkit

```python
from midmind import PromptBuilder, PromptTemplates, PromptValidator

pb = PromptBuilder()
pb.set_instructions("You are an AI assistant.")
pb.set_context("User wants to learn about prompt engineering.")
pb.set_task(PromptTemplates.reasoning("Explain reinforcement learning step by step."))
prompt = pb.build()

PromptValidator.validate(prompt)  # {'length_ok': True, 'has_instruction': True}
```

## Limitations

Without an embedder, retrieval is keyword-based (BM25): it matches words, not meaning, so
"AI" will not match "artificial intelligence". Add an embedder to fix that. Embeddings are kept
in memory and searched exhaustively, which is fast up to tens of thousands of chunks.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
