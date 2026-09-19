from pathlib import Path

from src.chunking import SentenceChunker
from src.models import Document
from src.store import EmbeddingStore


DATA_DIR = Path("data/library-policy")


QUERIES = [
    {
        "question": "Quy định sử dụng thư viện như thế nào?",
        "filter": None,
        "expected_doc": "stu-library-usage",
    },
    {
        "question": "Sinh viên được mượn tài liệu trong thư viện trong thời gian bao lâu?",
        "filter": None,
        "expected_doc": "stu-library-usage",
    },
    {
        "question": "Chính sách học bổng tuyển sinh năm 2026 của STU có những mức nào?",
        "filter": {"audience": "student"},
        "expected_doc": "stu-scholarship-2026",
    },
    {
        "question": "Học phí đào tạo khóa 2026 tại STU là bao nhiêu?",
        "filter": {"audience": "student"},
        "expected_doc": "stu-tuition-2026",
    },
    {
        "question": "Sinh viên cần tuân thủ những quy định nào khi sử dụng thư viện?",
        "filter": None,
        "expected_doc": "stu-library-usage",
    },
]


def parse_frontmatter(text: str):
    """
    Tách YAML frontmatter ở đầu file Markdown.

    Trả về:
        metadata: dict
        body: phần nội dung Markdown sau frontmatter
    """
    metadata = {}
    body = text

    if not text.startswith("---"):
        return metadata, body

    parts = text.split("---", 2)

    if len(parts) < 3:
        return metadata, body

    frontmatter = parts[1].strip()
    body = parts[2].strip()

    for line in frontmatter.splitlines():
        line = line.strip()

        if not line or ":" not in line:
            continue

        key, value = line.split(":", 1)

        key = key.strip()
        value = value.strip().strip('"').strip("'")

        metadata[key] = value

    return metadata, body


def load_documents():
    """
    Đọc corpus, chunk bằng SentenceChunker,
    sau đó tạo Document cho từng chunk.
    """
    chunker = SentenceChunker(max_sentences_per_chunk=3)

    documents = []

    files = sorted(DATA_DIR.glob("*.md"))

    print(f"Found {len(files)} Markdown files.")

    for path in files:
        text = path.read_text(encoding="utf-8")

        metadata, body = parse_frontmatter(text)

        chunks = chunker.chunk(body)

        print(f"{path.name}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "doc_id": path.stem,
            }

            document = Document(
                id=f"{path.stem}#{i}",
                content=chunk,
                metadata=chunk_metadata,
            )

            documents.append(document)

    return documents


def print_results(question, expected_doc, results):
    print()
    print("=" * 80)
    print(f"QUERY: {question}")
    print(f"EXPECTED DOC: {expected_doc}")
    print("=" * 80)

    if not results:
        print("No results.")
        return False

    hit = False

    for rank, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        doc_id = metadata.get("doc_id")

        if doc_id == expected_doc:
            hit = True

        print(
            f"\n#{rank}"
            f" | score={result['score']:.4f}"
            f" | id={result['id']}"
            f" | doc_id={doc_id}"
        )

        print(result["content"])

    print()
    print(f"Hit@3: {'YES' if hit else 'NO'}")

    return hit


def main():
    documents = load_documents()

    print()
    print(f"Total chunks: {len(documents)}")

    store = EmbeddingStore()
    store.add_documents(documents)

    print(f"Store size: {store.get_collection_size()}")

    hit_count = 0

    for item in QUERIES:
        results = store.search_with_filter(
            item["question"],
            top_k=3,
            metadata_filter=item["filter"],
        )

        hit = print_results(
            item["question"],
            item["expected_doc"],
            results,
        )

        if hit:
            hit_count += 1

    total = len(QUERIES)
    hit_rate = hit_count / total * 100

    print()
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total queries: {total}")
    print(f"Hit@3: {hit_count}/{total} = {hit_rate:.1f}%")


if __name__ == "__main__":
    main()