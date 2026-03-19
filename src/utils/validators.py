import json


def parse_json_object(raw_response: str) -> dict[str, object] | None:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def validate_rank_payload(payload: dict[str, object]) -> list[dict[str, object]]:
    rankings = payload.get("rankings", [])
    if not isinstance(rankings, list):
        return []

    valid_rankings = []
    for item in rankings:
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id")
        score = item.get("score")
        if isinstance(chunk_id, str) and isinstance(score, int | float):
            valid_rankings.append(
                {
                    "chunk_id": chunk_id,
                    "score": float(score),
                }
            )
    return valid_rankings


def validate_reflect_payload(payload: dict[str, object]) -> dict[str, object] | None:
    confidence = payload.get("confidence")
    needs_retry = payload.get("needs_retry")
    reason = payload.get("reason")

    if confidence not in {"high", "medium", "low"}:
        return None
    if not isinstance(needs_retry, bool):
        return None
    if not isinstance(reason, str):
        return None

    return {
        "confidence": confidence,
        "needs_retry": needs_retry,
        "reason": reason.strip(),
    }


def validate_citations(citations: object) -> list[str]:
    valid_citations: list[str] = []
    if not isinstance(citations, list):
        return valid_citations

    for citation in citations:
        if (
            isinstance(citation, str)
            and citation.startswith("(")
            and ")["
            in citation
            and citation.endswith("]")
        ):
            valid_citations.append(citation)
    return valid_citations


def validate_answer_payload(payload: dict[str, object]) -> dict[str, object] | None:
    answer = payload.get("answer")
    reasoning = payload.get("reasoning")
    confidence_score = payload.get("confidence_score")
    needs_retry = payload.get("needs_retry")
    used_chunk_ids = payload.get("used_chunk_ids")
    citations = validate_citations(payload.get("citations"))

    if not isinstance(answer, str):
        return None
    if not isinstance(reasoning, str):
        return None
    if not isinstance(confidence_score, int):
        return None
    if not isinstance(needs_retry, bool):
        return None
    if not isinstance(used_chunk_ids, list):
        return None

    return {
        "answer": answer,
        "reasoning": reasoning,
        "confidence_score": max(0, min(100, confidence_score)),
        "needs_retry": needs_retry,
        "used_chunk_ids": used_chunk_ids,
        "citations": citations,
    }


def validate_judge_payload(payload: dict[str, object]) -> dict[str, object] | None:
    alignment_score = payload.get("alignment_score")
    alignment_reason = payload.get("alignment_reason")
    retrieval_assessment = payload.get("retrieval_assessment")
    answer_assessment = payload.get("answer_assessment")
    overall_note = payload.get("overall_note")

    if alignment_score is not None and not isinstance(alignment_score, int):
        return None
    if not isinstance(alignment_reason, str):
        return None
    if not isinstance(retrieval_assessment, str):
        return None
    if not isinstance(answer_assessment, str):
        return None
    if not isinstance(overall_note, str):
        return None

    normalized_score = None
    if isinstance(alignment_score, int):
        normalized_score = max(1, min(5, alignment_score))

    return {
        "alignment_score": normalized_score,
        "alignment_reason": alignment_reason.strip(),
        "retrieval_assessment": retrieval_assessment.strip(),
        "answer_assessment": answer_assessment.strip(),
        "overall_note": overall_note.strip(),
    }
