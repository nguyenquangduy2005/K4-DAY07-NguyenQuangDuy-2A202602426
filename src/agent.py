from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)

        if not results:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức."

        context_parts = []

        for i, result in enumerate(results, start=1):
            source = result["metadata"].get("source", "unknown")
            content = result["content"]

            context_parts.append(
                f"[{i}] Source: {source}\n{content}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""Chỉ sử dụng thông tin trong CONTEXT để trả lời câu hỏi.
    Nếu CONTEXT không chứa thông tin cần thiết, hãy nói rằng không tìm thấy thông tin.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    Hãy trả lời và nêu nguồn tham khảo bằng số [1], [2], ... nếu có thể.
    """

        return self.llm_fn(prompt)