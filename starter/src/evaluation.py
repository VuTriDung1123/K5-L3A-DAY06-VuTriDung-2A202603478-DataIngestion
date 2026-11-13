"""Evaluate retrieved evidence and aggregate Hit@1 / Evidence Hit@3 metrics."""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def _norm_match(text: str) -> str:
    """Normalize Unicode and collapse whitespace for verbatim evidence substring checks."""
    s = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", s).strip()


def _hit_contains_all_clauses(hit_text: str, clauses: list[str]) -> bool:
    norm_hit = _norm_match(hit_text)
    return all(_norm_match(cl) in norm_hit for cl in clauses)


def _hit_contains_any_clause(hit_text: str, clauses: list[str]) -> bool:
    norm_hit = _norm_match(hit_text)
    return any(_norm_match(cl) in norm_hit for cl in clauses)


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check gold source/page and full evidence presence for top-1 and top-3."""
    gold_doc = question["gold_source_id"]
    valid_pages = {int(question["gold_page"])}
    for alt_p in question.get("alternative_gold_pages", []):
        valid_pages.add(int(alt_p))

    clauses = question.get("evidence_clauses") or [question["evidence_quote"]]
    applied_filter = question.get("filter")

    hit_evals: list[dict[str, Any]] = []
    for h in hits:
        same_source = h.get("document_id") == gold_doc
        same_page = int(h.get("page", -1)) in valid_pages
        full_evidence = _hit_contains_all_clauses(h.get("text", ""), clauses) if (same_source and same_page) else False
        partial_evidence = _hit_contains_any_clause(h.get("text", ""), clauses) if (same_source and same_page) else False
        hit_evals.append(
            {
                "rank": h.get("rank"),
                "chunk_id": h.get("chunk_id"),
                "document_id": h.get("document_id"),
                "source": h.get("source"),
                "page": h.get("page"),
                "chapter": h.get("chapter"),
                "section": h.get("section"),
                "score": h.get("score"),
                "cosine_distance": h.get("cosine_distance"),
                "same_source": same_source,
                "same_page": same_page,
                "full_evidence_match": full_evidence,
                "partial_evidence_match": partial_evidence,
                "metadata": h.get("metadata"),
                "text": h.get("text"),
            }
        )

    hit_at_1 = bool(hit_evals and hit_evals[0]["full_evidence_match"])
    evidence_hit_at_3 = any(item["full_evidence_match"] for item in hit_evals[:3])

    # Check if multi-chunk union across top-3 on gold source/page recovers split evidence
    gold_top3_text = " ".join(
        item["text"] for item in hit_evals[:3] if item["same_source"] and item["same_page"]
    )
    multi_chunk_union_hit3 = _hit_contains_all_clauses(gold_top3_text, clauses) if gold_top3_text else False
    evidence_split_error = (not evidence_hit_at_3) and multi_chunk_union_hit3

    filter_compliant: bool | None = None
    if applied_filter:
        filter_compliant = bool(hit_evals) and all(
            str((item.get("metadata") or {}).get(k)) == str(v)
            for item in hit_evals[:3]
            for k, v in applied_filter.items()
        )

    return {
        "question_id": question["id"],
        "type": question.get("type"),
        "split": question["split"],
        "question": question["question"],
        "gold_source_id": gold_doc,
        "gold_page": question["gold_page"],
        "gold_section": question.get("gold_section"),
        "filter_requested": question.get("filter"),
        "hit_at_1": hit_at_1,
        "evidence_hit_at_3": evidence_hit_at_3,
        "multi_chunk_union_hit3": multi_chunk_union_hit3,
        "evidence_split_error": evidence_split_error,
        "filter_compliant": filter_compliant,
        "hits": hit_evals,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1, Evidence Hit@3, and filter compliance by strategy and split."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for tr in traces:
        grouped.setdefault((tr["strategy"], tr["split"]), []).append(tr)

    summary: dict[str, Any] = {}
    for (strategy, split), items in sorted(grouped.items()):
        total = len(items)
        h1 = sum(1 for it in items if it["hit_at_1"])
        h3 = sum(1 for it in items if it["evidence_hit_at_3"])
        union_h3 = sum(1 for it in items if it["multi_chunk_union_hit3"])
        split_errs = sum(1 for it in items if it["evidence_split_error"])

        filtered_items = [it for it in items if it["filter_compliant"] is not None]
        if filtered_items:
            f_ok = sum(1 for it in filtered_items if it["filter_compliant"])
            filter_summary = f"{f_ok}/{len(filtered_items)}"
        else:
            filter_summary = "N/A"

        summary[f"{strategy}:{split}"] = {
            "strategy": strategy,
            "split": split,
            "n": total,
            "hit_at_1_count": h1,
            "hit_at_1_ratio": f"{h1}/{total}",
            "hit_at_1_pct": round(100.0 * h1 / total, 1) if total else 0.0,
            "evidence_hit_at_3_count": h3,
            "evidence_hit_at_3_ratio": f"{h3}/{total}",
            "evidence_hit_at_3_pct": round(100.0 * h3 / total, 1) if total else 0.0,
            "multi_chunk_union_hit3_ratio": f"{union_h3}/{total}",
            "evidence_split_errors": split_errs,
            "filter_compliance": filter_summary,
        }
    return summary
