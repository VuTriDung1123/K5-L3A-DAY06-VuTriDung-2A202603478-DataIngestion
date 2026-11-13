"""End-to-end Day 06 data ingestion, pgvector indexing, and benchmark runner."""
from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import OpenRouterEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .models import Chunk
from .store import PgVectorStore


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip()


def _chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    lens = sorted(len(c.text) for c in chunks)
    if not lens:
        return {"count": 0}
    n = len(lens)
    total = sum(lens)
    return {
        "count": n,
        "total_chars": total,
        "min_chars": lens[0],
        "p50_chars": lens[n // 2],
        "p90_chars": lens[int(n * 0.9)],
        "max_chars": lens[-1],
        "mean_chars": round(total / n, 1),
    }


def _verify_idempotency_and_pruning(
    store: PgVectorStore,
    strategy: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> dict[str, Any]:
    """Verify re-ingesting identical version creates no duplicates and changed docs prune stale chunks."""
    before_count = store.count_chunks(strategy)
    store.upsert(strategy, chunks, vectors)
    after_reingest_count = store.count_chunks(strategy)

    # Isolated stale-chunk cleanup verification on a temporary partition
    temp_strategy = "_cleanup_check"
    dim = store.dimensions
    unit_vec = [1.0 / (dim ** 0.5)] * dim
    c1 = Chunk("tmp:c0", "tmp-doc", "tmp.pdf", 1, "chunk zero", {"chapter": "Chapter 1", "section": "1.1"})
    c2 = Chunk("tmp:c1", "tmp-doc", "tmp.pdf", 1, "chunk one", {"chapter": "Chapter 1", "section": "1.1"})
    store.upsert(temp_strategy, [c1, c2], [unit_vec, unit_vec])
    cnt_two = store.count_chunks(temp_strategy, "tmp-doc")
    store.upsert(temp_strategy, [c1], [unit_vec])
    pruned = store.last_upsert_stats.get("deleted_obsolete", 0)
    cnt_one = store.count_chunks(temp_strategy, "tmp-doc")
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE strategy = %s;", (temp_strategy,))
        conn.commit()

    return {
        "strategy_checked": strategy,
        "count_before_reingest": before_count,
        "count_after_reingest": after_reingest_count,
        "zero_duplicates": before_count == after_reingest_count,
        "stale_cleanup_test_initial": cnt_two,
        "stale_cleanup_test_after_shrink": cnt_one,
        "stale_cleanup_deleted_rows": pruned,
    }


def _collect_coherence_samples(
    store: PgVectorStore,
    anchors: list[dict[str, Any]],
    strategies: list[str],
) -> list[dict[str, Any]]:
    """Locate chunks covering each pre-selected anchor position across strategies."""
    samples: list[dict[str, Any]] = []
    for anchor in anchors:
        page = int(anchor["page"])
        norm_anchor = _norm_ws(anchor["anchor_text"])
        row: dict[str, Any] = {
            "anchor_id": anchor["anchor_id"],
            "page": page,
            "chapter": anchor["chapter"],
            "section": anchor["section"],
            "anchor_text": anchor["anchor_text"],
            "strategies": {},
        }
        for strat in strategies:
            page_chunks = store.fetch_chunks_by_page(strat, page)
            matched = [c for c in page_chunks if norm_anchor in _norm_ws(c["text"])]
            if not matched:
                words = norm_anchor.split()
                prefix = " ".join(words[: max(3, len(words) // 2)])
                matched = [c for c in page_chunks if prefix in _norm_ws(c["text"])]
            row["strategies"][strat] = [
                {
                    "chunk_id": m["chunk_id"],
                    "chars": len(m["text"]),
                    "chapter": m["chapter"],
                    "section": m["section"],
                    "text": m["text"],
                }
                for m in matched
            ]
        samples.append(row)
    return samples


def main() -> None:
    starter_dir = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(starter_dir / ".env")

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL missing in starter/.env.")

    model_name = os.getenv("EMBEDDING_MODEL", "").strip() or "openai/text-embedding-3-small"
    pdf_path = starter_dir / "data" / "Principles-of-Data-Science-WEB.pdf"
    doc_id = "openstax-pds-2025"
    source_rel = "starter/data/Principles-of-Data-Science-WEB.pdf"

    # 1. Load & normalize corpus (107 content pages >= 30 required)
    t_ingest_start = time.perf_counter()
    ingestion_log: list[dict[str, Any]] = []
    pages = load_pdf(pdf_path, document_id=doc_id, source=source_rel, ingestion_log=ingestion_log)
    pdf_extract_seconds = time.perf_counter() - t_ingest_start
    if len(pages) < 30:
        raise RuntimeError(f"Corpus must contain at least 30 content pages, got {len(pages)}.")

    # 2. Load pre-registered benchmark and coherence anchors
    bench_data = json.loads((repo_root / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    questions = bench_data["questions"]
    dev_qs = [q for q in questions if q["split"] == "dev"]
    test_qs = [q for q in questions if q["split"] == "test"]
    filter_qs = [q for q in questions if q.get("filter")]
    assert len(questions) >= 20 and len(dev_qs) >= 15 and len(test_qs) >= 5
    assert len(filter_qs) >= 3 and any(q["split"] == "test" for q in filter_qs)

    anchors_data = json.loads((repo_root / "benchmark" / "coherence_anchors.json").read_text(encoding="utf-8"))
    anchors = anchors_data["anchors"]

    # 3. Initialize real embedder and pgvector store
    embedder = OpenRouterEmbedder(model_name=model_name, dimensions=1536, batch_size=64)
    store = PgVectorStore(database_url=database_url, dimensions=embedder.dimensions)
    server_info = store.server_info()
    store.create_schema()

    # 4. Chunking for A, B, and C (C shares identical chunks & vectors with B)
    t0 = time.perf_counter()
    chunks_a = [c for p in pages for c in fixed_size(p, size=500, overlap=50)]
    chunk_time_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    chunks_b = [c for p in pages for c in by_paragraph(p, max_size=800)]
    chunk_time_b = time.perf_counter() - t0
    chunks_c = chunks_b

    # 5. Embed & persist each strategy in pgvector
    print(f"Embedding Strategy A ({len(chunks_a)} chunks)...", flush=True)
    vecs_a, usage_a = embedder.embed_documents_with_usage([c.text for c in chunks_a])
    t0 = time.perf_counter()
    store.upsert("A", chunks_a, vecs_a)
    upsert_time_a = time.perf_counter() - t0

    print(f"Embedding Strategy B ({len(chunks_b)} chunks)...", flush=True)
    vecs_b, usage_b = embedder.embed_documents_with_usage([c.text for c in chunks_b])
    t0 = time.perf_counter()
    store.upsert("B", chunks_b, vecs_b)
    upsert_time_b = time.perf_counter() - t0

    print("Persisting Strategy C (identical chunks & vectors to B)...", flush=True)
    t0 = time.perf_counter()
    store.upsert("C", chunks_c, vecs_b)
    upsert_time_c = time.perf_counter() - t0

    # 6. Verify idempotent re-ingestion & stale chunk deletion
    idempotency_report = _verify_idempotency_and_pruning(store, "B", chunks_b, vecs_b)

    # 7. Embed all 20 benchmark questions ONCE so queries are identical across A, B, C
    print(f"Embedding {len(questions)} benchmark queries...", flush=True)
    query_vectors: dict[str, list[float]] = {}
    t_qembed_start = time.perf_counter()
    for q in questions:
        query_vectors[q["id"]] = embedder.embed_query(q["question"])
    query_embed_seconds = time.perf_counter() - t_qembed_start

    # 8. Execute retrieval & trace logging for A, B, C
    all_traces: list[dict[str, Any]] = []
    search_timings: dict[str, float] = {}
    for strategy in ("A", "B", "C"):
        t0 = time.perf_counter()
        for q in questions:
            filters = q.get("filter") if strategy == "C" else None
            hits = store.search(strategy, query_vectors[q["id"]], top_k=3, filters=filters)
            trace = evaluate_query(q, hits)
            trace["strategy"] = strategy
            trace["filter_applied"] = filters
            all_traces.append(trace)
        search_timings[strategy] = round(time.perf_counter() - t0, 4)

    retrieval_summary = summarize(all_traces)
    coherence_samples = _collect_coherence_samples(store, anchors, ["A", "B", "C"])

    # 9. Compute exact timing & API cost metrics from OpenRouter billing telemetry
    stats_a = _chunk_stats(chunks_a)
    stats_b = _chunk_stats(chunks_b)
    stats_c = _chunk_stats(chunks_c)

    strategy_metrics = {
        "A": {
            "config": {"chunking": "fixed_size", "size": 500, "overlap": 50, "filter": "none"},
            "chunk_stats": stats_a,
            "persisted_count": store.count_chunks("A"),
            "ingest_seconds": round(pdf_extract_seconds + chunk_time_a + usage_a["seconds"] + upsert_time_a, 3),
            "embed_seconds": usage_a["seconds"],
            "embed_tokens": usage_a["tokens"],
            "doc_embed_cost_usd": usage_a["cost_usd"],
            "query_embed_cost_usd": round(embedder.query_cost_usd, 8),
            "total_cost_usd": round(usage_a["cost_usd"] + embedder.query_cost_usd, 8),
            "search_seconds_20q": search_timings["A"],
        },
        "B": {
            "config": {"chunking": "by_paragraph", "max_size": 800, "overlap": 0, "filter": "none"},
            "chunk_stats": stats_b,
            "persisted_count": store.count_chunks("B"),
            "ingest_seconds": round(pdf_extract_seconds + chunk_time_b + usage_b["seconds"] + upsert_time_b, 3),
            "embed_seconds": usage_b["seconds"],
            "embed_tokens": usage_b["tokens"],
            "doc_embed_cost_usd": usage_b["cost_usd"],
            "query_embed_cost_usd": round(embedder.query_cost_usd, 8),
            "total_cost_usd": round(usage_b["cost_usd"] + embedder.query_cost_usd, 8),
            "search_seconds_20q": search_timings["B"],
        },
        "C": {
            "config": {"chunking": "by_paragraph", "max_size": 800, "overlap": 0, "filter": "pre-supplied metadata filter when provided"},
            "chunk_stats": stats_c,
            "persisted_count": store.count_chunks("C"),
            "ingest_seconds_incremental_reuse_b": round(upsert_time_c, 3),
            "ingest_seconds_standalone_equivalent": round(pdf_extract_seconds + chunk_time_b + usage_b["seconds"] + upsert_time_c, 3),
            "embed_seconds_standalone_equivalent": usage_b["seconds"],
            "embed_tokens_standalone_equivalent": usage_b["tokens"],
            "doc_embed_cost_usd_incremental": 0.0,
            "doc_embed_cost_usd_standalone_equivalent": usage_b["cost_usd"],
            "query_embed_cost_usd": round(embedder.query_cost_usd, 8),
            "total_cost_usd_standalone_equivalent": round(usage_b["cost_usd"] + embedder.query_cost_usd, 8),
            "search_seconds_20q": search_timings["C"],
        },
    }

    # 10. Save artifacts to results/ and starter/results/
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    starter_results_dir = starter_dir / "results"
    starter_results_dir.mkdir(parents=True, exist_ok=True)

    provider_fallback_log = [
        {
            "attempted_provider": "openai-direct (text-embedding-3-small)",
            "status": "failed_quota",
            "error": "HTTP 429 Too Many Requests on direct OpenAI key prior to experiment.",
        },
        {
            "attempted_provider": "google-genai (gemini-embedding-001, Free Tier)",
            "status": "failed_rate_limit",
            "error": "429 RESOURCE_EXHAUSTED: Free Tier limit 100 embed_content requests/min hit during Strategy A batch 3.",
        },
        {
            "selected_provider": "openrouter (openai/text-embedding-3-small, 1536d)",
            "status": "succeeded",
            "note": "Switched prior to completing any strategy run; all three strategies A, B, and C use this identical model.",
        },
    ]

    run_metadata = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_pages_loaded": len(pages),
        "embedding": embedder.stats(),
        "query_embed_seconds_20q": round(query_embed_seconds, 3),
        "provider_fallback_log": provider_fallback_log,
        "vector_store": {
            "engine": "PostgreSQL + pgvector",
            **server_info,
            "metric": store.metric,
            "search_mode": store.search_mode,
        },
        "packages": {
            pkg: importlib.metadata.version(pkg)
            for pkg in ("psycopg", "pypdf", "python-dotenv", "requests")
        },
        "idempotency_and_pruning": idempotency_report,
    }

    (results_dir / "ingestion_log.json").write_text(
        json.dumps(
            {
                "pages_loaded": len(pages),
                "pdf_extract_seconds": round(pdf_extract_seconds, 3),
                "provider_fallback_log": provider_fallback_log,
                "warnings_and_errors": [e for e in ingestion_log if e["status"] != "ok"],
                "page_logs": ingestion_log,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (results_dir / "traces_dev_test.json").write_text(
        json.dumps({"run_metadata": run_metadata, "traces": all_traces}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_payload = {
        "run_metadata": run_metadata,
        "strategy_metrics": strategy_metrics,
        "retrieval_summary": retrieval_summary,
    }
    (results_dir / "metrics_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (starter_results_dir / "metrics_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (results_dir / "coherence_samples.json").write_text(
        json.dumps({"samples": coherence_samples}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    test_audit_rows = [tr for tr in all_traces if tr["split"] == "test"]
    (results_dir / "test_audit_samples.json").write_text(
        json.dumps({"test_traces": test_audit_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (results_dir / "retrieval_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "strategy", "split", "n", "hit_at_1_ratio", "hit_at_1_pct",
                "evidence_hit_at_3_ratio", "evidence_hit_at_3_pct",
                "multi_chunk_union_hit3_ratio", "evidence_split_errors", "filter_compliance",
            ],
        )
        writer.writeheader()
        for val in retrieval_summary.values():
            writer.writerow({k: val[k] for k in writer.fieldnames})

    print("\n=== RETRIEVAL SUMMARY ===")
    for key, val in retrieval_summary.items():
        print(
            f"{key:10s} | Hit@1={val['hit_at_1_ratio']} ({val['hit_at_1_pct']}%) | "
            f"Evidence Hit@3={val['evidence_hit_at_3_ratio']} ({val['evidence_hit_at_3_pct']}%) | "
            f"Union@3={val['multi_chunk_union_hit3_ratio']} | SplitErr={val['evidence_split_errors']} | "
            f"Filter={val['filter_compliance']}"
        )
    print(f"Total API cost billed (A+B+queries): ${embedder.doc_cost_usd + embedder.query_cost_usd:.6f}")
    print("Saved outputs to results/ and starter/results/.")


if __name__ == "__main__":
    main()
