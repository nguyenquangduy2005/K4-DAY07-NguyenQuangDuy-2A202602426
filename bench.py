from pathlib import Path

from src.chunking import SentenceChunker
from src.models import Document
from src.store import EmbeddingStore


DATA_DIR = Path("data/library-policy")
OUTPUT_FILE = Path("ket_qua_benchmark.txt")


QUERIES = [
    {
        "question": "Quy định sử dụng thư viện như thế nào?",
        "filter": None,
        "expected_doc": "stu-library-usage",
        "answer_marker": "không ăn uống trong thư viện",
    },
    {
        "question": "Sinh viên được mượn tài liệu trong thư viện trong thời gian bao lâu?",
        "filter": None,
        "expected_doc": "stu-library-usage",
        "answer_marker": "gia hạn thêm 07 ngày",
    },
    {
        "question": "Chính sách học bổng tuyển sinh năm 2026 của STU có những mức nào?",
        "filter": {"audience": "student"},
        "expected_doc": "stu-scholarship-2026",
        "answer_marker": "50% học phí 04 năm",
    },
    {
        "question": "Học phí đào tạo khóa 2026 tại STU là bao nhiêu?",
        "filter": {"audience": "student"},
        "expected_doc": "stu-tuition-2026",
        "answer_marker": "21.025.000 đồng/học kỳ",
    },
    {
        "question": "Sinh viên cần tuân thủ những quy định nào khi sử dụng thư viện?",
        "filter": None,
        "expected_doc": "stu-library-usage",
        "answer_marker": "không ăn uống trong thư viện",
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


def evaluate_result(item, results):
    """
    Chấm theo rubric CP6:

    2 điểm:
        - expected_doc ở top-1
        - answer_marker xuất hiện trong context top-3

    1 điểm:
        - expected_doc ở top-2 hoặc top-3

    0 điểm:
        - expected_doc không xuất hiện
        - hoặc context không chứa answer_marker
          khi expected_doc chỉ xuất hiện ở top-1.
    """
    expected_doc = item["expected_doc"]
    answer_marker = item["answer_marker"]

    if not results:
        return 0, None, False

    gold_rank = None
    answer_found = False

    for rank, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        doc_id = metadata.get("doc_id")

        if doc_id == expected_doc and gold_rank is None:
            gold_rank = rank

        content = result.get("content", "")

        if answer_marker.lower() in content.lower():
            answer_found = True

    if gold_rank is None:
        return 0, None, answer_found

    if gold_rank == 1 and answer_found:
        return 2, gold_rank, answer_found

    if gold_rank in (2, 3):
        return 1, gold_rank, answer_found

    return 0, gold_rank, answer_found


def format_results(results):
    lines = []

    if not results:
        lines.append("No results.")
        return lines

    for rank, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        doc_id = metadata.get("doc_id")

        lines.append(
            f"#{rank}"
            f" | score={result['score']:.4f}"
            f" | id={result['id']}"
            f" | doc_id={doc_id}"
        )

        lines.append(f"CONTENT: {result['content']}")

    return lines


def run_query(store, item, metadata_filter):
    return store.search_with_filter(
        item["question"],
        top_k=3,
        metadata_filter=metadata_filter,
    )


def main():
    documents = load_documents()

    print()
    print(f"Total chunks: {len(documents)}")

    store = EmbeddingStore()
    store.add_documents(documents)

    print(f"Store size: {store.get_collection_size()}")

    output_lines = []

    output_lines.append("=" * 80)
    output_lines.append("CP6 BENCHMARK - SENTENCE-BASED CHUNKING")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("Embedding backend: MockEmbedder")
    output_lines.append("Chunking: SentenceChunker")
    output_lines.append("max_sentences_per_chunk: 3")
    output_lines.append(f"Total chunks: {len(documents)}")
    output_lines.append("")

    total_score = 0
    max_score = len(QUERIES) * 2

    for index, item in enumerate(QUERIES, start=1):
        question = item["question"]
        expected_doc = item["expected_doc"]
        answer_marker = item["answer_marker"]
        metadata_filter = item["filter"]

        print()
        print("=" * 80)
        print(f"QUERY {index}: {question}")
        print(f"EXPECTED DOC: {expected_doc}")
        print(f"ANSWER MARKER: {answer_marker}")
        print("=" * 80)

        output_lines.append("=" * 80)
        output_lines.append(f"QUERY {index}: {question}")
        output_lines.append(f"EXPECTED DOC: {expected_doc}")
        output_lines.append(f"ANSWER MARKER: {answer_marker}")
        output_lines.append("=" * 80)

        # ---------------------------------------------------------
        # A/B TEST CHỈ CHO CÂU HỎI CÓ METADATA FILTER
        # ---------------------------------------------------------
        if metadata_filter is not None:
            print()
            print("[A] WITHOUT METADATA FILTER")

            output_lines.append("")
            output_lines.append("[A] WITHOUT METADATA FILTER")

            results_without_filter = run_query(
                store,
                item,
                metadata_filter=None,
            )

            score_a, rank_a, answer_a = evaluate_result(
                item,
                results_without_filter,
            )

            result_lines = format_results(results_without_filter)

            for line in result_lines:
                print(line)
                output_lines.append(line)

            print(
                f"Score A: {score_a}/2"
                f" | Gold rank: {rank_a}"
                f" | Answer marker: {'YES' if answer_a else 'NO'}"
            )

            output_lines.append(
                f"Score A: {score_a}/2"
                f" | Gold rank: {rank_a}"
                f" | Answer marker: {'YES' if answer_a else 'NO'}"
            )

            print()
            print(f"[B] WITH METADATA FILTER: {metadata_filter}")

            output_lines.append("")
            output_lines.append(
                f"[B] WITH METADATA FILTER: {metadata_filter}"
            )

            results_with_filter = run_query(
                store,
                item,
                metadata_filter=metadata_filter,
            )

            score_b, rank_b, answer_b = evaluate_result(
                item,
                results_with_filter,
            )

            result_lines = format_results(results_with_filter)

            for line in result_lines:
                print(line)
                output_lines.append(line)

            print(
                f"Score B: {score_b}/2"
                f" | Gold rank: {rank_b}"
                f" | Answer marker: {'YES' if answer_b else 'NO'}"
            )

            output_lines.append(
                f"Score B: {score_b}/2"
                f" | Gold rank: {rank_b}"
                f" | Answer marker: {'YES' if answer_b else 'NO'}"
            )

            same_results = [
                result["id"] for result in results_without_filter
            ] == [
                result["id"] for result in results_with_filter
            ]

            print(
                f"A/B identical Top-3: "
                f"{'YES' if same_results else 'NO'}"
            )

            output_lines.append(
                f"A/B identical Top-3: "
                f"{'YES' if same_results else 'NO'}"
            )

            # Với tổng điểm, dùng kết quả CÓ filter
            # vì đây là cấu hình benchmark chính.
            total_score += score_b

        else:
            print()
            print("[BENCHMARK] WITHOUT METADATA FILTER")

            output_lines.append("")
            output_lines.append("[BENCHMARK] WITHOUT METADATA FILTER")

            results = run_query(
                store,
                item,
                metadata_filter=None,
            )

            score, gold_rank, answer_found = evaluate_result(
                item,
                results,
            )

            result_lines = format_results(results)

            for line in result_lines:
                print(line)
                output_lines.append(line)

            print(
                f"Score: {score}/2"
                f" | Gold rank: {gold_rank}"
                f" | Answer marker: "
                f"{'YES' if answer_found else 'NO'}"
            )

            output_lines.append(
                f"Score: {score}/2"
                f" | Gold rank: {gold_rank}"
                f" | Answer marker: "
                f"{'YES' if answer_found else 'NO'}"
            )

            total_score += score

        print()
        output_lines.append("")

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    score_percent = total_score / max_score * 100

    print()
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total queries: {len(QUERIES)}")
    print(f"Total score: {total_score}/{max_score}")
    print(f"Score: {score_percent:.1f}%")

    output_lines.append("=" * 80)
    output_lines.append("BENCHMARK SUMMARY")
    output_lines.append("=" * 80)
    output_lines.append(f"Total queries: {len(QUERIES)}")
    output_lines.append(f"Total score: {total_score}/{max_score}")
    output_lines.append(f"Score: {score_percent:.1f}%")

    OUTPUT_FILE.write_text(
        "\n".join(output_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Saved benchmark result to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()