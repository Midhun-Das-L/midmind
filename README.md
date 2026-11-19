# MidMind

**Turn a folder of documents into a grounded, cited LLM prompt — in one line, with zero dependencies.**

MidMind does the "R" in RAG for you: it loads your files, splits them into overlapping chunks,
ranks them against the question with BM25, and formats the best ones into a prompt that tells
the model to answer only from your documents and cite its sources. Send the result to any LLM.

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

Custom template:

```python
rag.build("Who wrote it?", template="Use these notes:\n{context}\n\nQ: {question}\nA:")
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

Retrieval is keyword-based (BM25): it matches words, not meaning, so "AI" will not match
"artificial intelligence". Embedding-based search is on the roadmap.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
