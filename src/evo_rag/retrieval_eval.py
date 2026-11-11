"""Score retrieval quality against the gold doc_ids of the QA set (no LLM needed).

Metrics, averaged over answerable dev questions:
  recall@k  fraction of a question's gold documents found among the top-k chunks
  all@k     share of questions where *every* gold document was found (matters for multi-hop)
  mrr       1 / rank of the first chunk coming from a gold document (0 if none)

Run:  uv run python -m evo_rag.retrieval_eval
"""

import json
import time
from collections import defaultdict
from pathlib import Path

from evo_rag.config import PipelineConfig
from evo_rag.retrieval import build_retriever, load_corpus

QA_PATH = Path("data/processed/qa.jsonl")


def load_questions(split: str = "dev") -> list[dict]:
    with QA_PATH.open(encoding="utf-8") as f:
        qa = [json.loads(line) for line in f]
    return [q for q in qa if q["split"] == split and q["type"] != "unanswerable"]


def score_question(retrieved_doc_ids: list[str], gold: set[str]) -> dict:
    found = gold & set(retrieved_doc_ids)
    first_hit = next((rank for rank, d in enumerate(retrieved_doc_ids, 1) if d in gold), None)
    return {
        "recall": len(found) / len(gold),
        "all": float(found == gold),
        "mrr": 1 / first_hit if first_hit else 0.0,
    }


def evaluate_retrieval(config: PipelineConfig, questions: list[dict], docs: list[dict] | None = None, **kwargs) -> dict:
    retriever = build_retriever(config, docs, **kwargs)
    per_type = defaultdict(list)
    start = time.perf_counter()
    for q in questions:
        retrieved = [chunk.doc_id for chunk, _ in retriever.search(q["question"])]
        per_type[q["type"]].append(score_question(retrieved, set(q["doc_ids"])))
    seconds_per_query = (time.perf_counter() - start) / len(questions)

    all_scores = [s for scores in per_type.values() for s in scores]
    mean = lambda scores, key: sum(s[key] for s in scores) / len(scores)  # noqa: E731
    result = {m: mean(all_scores, m) for m in ("recall", "all", "mrr")}
    result |= {f"recall_{t}": mean(s, "recall") for t, s in per_type.items()}
    result["n_chunks"] = len(retriever.chunks)
    result["s_per_query"] = seconds_per_query
    return result


EXPERIMENTS = {
    "bm25, recursive 800": PipelineConfig(retrieval="bm25"),
    "bm25, no header": PipelineConfig(retrieval="bm25", chunk_header=False),
    "bm25, fixed 800": PipelineConfig(retrieval="bm25", chunker="fixed"),
    "bm25, recursive 300": PipelineConfig(retrieval="bm25", chunk_size=300),
    "bm25, recursive 1600": PipelineConfig(retrieval="bm25", chunk_size=1600),
    "dense bge-small": PipelineConfig(retrieval="dense"),
    "dense bge-small, no header": PipelineConfig(retrieval="dense", chunk_header=False),
    "dense e5-base": PipelineConfig(retrieval="dense", embedder="e5-base"),
    "hybrid bge-small, a=0.5": PipelineConfig(retrieval="hybrid"),
}


def main() -> None:
    questions = load_questions("dev")
    docs = load_corpus()
    print(f"{len(questions)} answerable dev questions, top_k=5\n")
    header = f"{'config':28} {'recall':>7} {'all':>6} {'mrr':>6} {'single':>7} {'multi':>6} {'num':>6} {'chunks':>7}"
    print(header)
    print("-" * len(header))
    for name, config in EXPERIMENTS.items():
        r = evaluate_retrieval(config, questions, docs)
        print(
            f"{name:28} {r['recall']:7.3f} {r['all']:6.3f} {r['mrr']:6.3f} "
            f"{r['recall_single_hop']:7.3f} {r['recall_multi_hop']:6.3f} {r['recall_numerical']:6.3f} {r['n_chunks']:7}"
        )


if __name__ == "__main__":
    main()
