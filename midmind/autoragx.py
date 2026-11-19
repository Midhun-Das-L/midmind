from typing import Callable, Optional

from .rag import RAGPromptBuilder


def autoragx(data_path: str, query: str, llm: Optional[Callable[[str], str]] = None,
             top_k: int = 3, template: str = "citations") -> str:
    """
    RAG in one line.

    Loads the documents in ``data_path``, retrieves the chunks most relevant
    to ``query`` and builds a grounded prompt. If ``llm`` (any ``str -> str``
    callable) is given, the prompt is sent to it and its answer is returned;
    otherwise the prompt itself is returned.
    """
    rag = RAGPromptBuilder.from_folder(data_path)
    prompt = rag.build(query, top_k=top_k, template=template)
    return llm(prompt) if llm else prompt
