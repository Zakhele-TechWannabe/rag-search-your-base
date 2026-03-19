import json
from pathlib import Path

from core.generate import generate_answer
from utils.clients import GeminiJudgeClient
from utils.config import Settings, load_settings
from utils.logger import get_logger
from utils.retriever import Retriever
from utils.validators import parse_json_object, validate_judge_payload


evaluate_logger = get_logger("evaluate", folder="evaluations")
DEFAULT_EVAL_SET_PATH = Path("docs/eval_set.json")
DEFAULT_ASK_EVAL_SET_PATH = Path("docs/ask_eval_set.json")


def load_eval_items(eval_set_path: Path) -> list[dict[str, object]]:
    payload = json.loads(eval_set_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation set must be a JSON list.")
    return payload


def normalize_question(question: str) -> str:
    return " ".join(question.lower().strip().split())


def find_expected_item(
    question: str,
    eval_set_path: Path,
) -> dict[str, object] | None:
    resolved_path = eval_set_path if eval_set_path.is_absolute() else load_settings().project_root / eval_set_path
    if not resolved_path.exists():
        return None
    normalized_question = normalize_question(question)
    for item in load_eval_items(resolved_path):
        if normalize_question(item.get("question", "")) == normalized_question:
            return item
    return None


def build_results_table(rows: list[dict[str, object]], top_k: int) -> str:
    header = [
        "| Question | Expected Source | Retrieved Top "
        f"{top_k} Sources | Recall@{top_k} |",
        "| --- | --- | --- | --- |",
    ]
    lines = []
    for row in rows:
        retrieval = row.get("retrieval", {})
        question = str(row["question"]).replace("|", "\\|")
        expected_slugs = ", ".join(str(slug) for slug in retrieval.get("expected_slugs", []))
        if not expected_slugs:
            expected_slugs = "unavailable"
        expected_slugs = expected_slugs.replace("|", "\\|")
        retrieved = ", ".join(str(slug) for slug in retrieval.get("retrieved_slugs", []))
        retrieved = retrieved.replace("|", "\\|")
        recall = retrieval.get("recall_at_k", "N/A")
        lines.append(f"| {question} | {expected_slugs} | {retrieved} | {recall} |")
    return "\n".join(header + lines)


def unavailable_judge(reason: str, expected_answer_summary: str) -> dict[str, object]:
    return {
        "answer_evaluation": {
            "expected_answer_summary": expected_answer_summary,
            "alignment_score": None,
            "alignment_reason": reason,
        },
        "judge": {
            "retrieval_assessment": reason,
            "answer_assessment": reason,
            "overall_note": reason,
        },
    }


def safe_generate_answer_text(question: str, settings: Settings) -> str:
    if not settings.openai_api_key:
        return "Answer generation is unavailable because OPENAI_API_KEY is not set."
    try:
        answer_result = generate_answer(question, settings)
        return str(answer_result.get("answer", "")).strip()
    except Exception as error:  # pragma: no cover - external service call
        evaluate_logger.warning("Answer generation failed for evaluation: %s", error)
        return f"Answer generation failed: {error}"


def build_judge_prompt(
    question: str,
    expected_slugs: list[str] | None,
    expected_answer_summary: str | None,
    retrieved_slugs: list[str],
    generated_answer: str,
) -> str:
    return f"""
You are reviewing a RAG evaluation result.

You are given:
- question: {question}
- expected source essay slugs: {json.dumps(expected_slugs or []) if expected_slugs else "unavailable"}
- expected answer summary: {expected_answer_summary or "unavailable"}
- retrieved source essay slugs: {json.dumps(retrieved_slugs)}
- generated answer: {generated_answer}

Recall@K is already computed elsewhere. Do not recalculate or replace that metric.

Return JSON with exactly these fields:
- alignment_score: integer from 1 to 5, or null if the expected answer summary is unavailable
- alignment_reason: short explanation of the alignment score
- retrieval_assessment: short comment on whether the retrieved sources look appropriate for the question
- answer_assessment: short comment on whether the generated answer aligns with the expected answer summary
- overall_note: short interpretation of the result as strong, weak, or ambiguous

Keep the commentary concise and specific to this example.
""".strip()


def judge_result(
    settings: Settings,
    question: str,
    expected_slugs: list[str] | None,
    expected_answer_summary: str | None,
    retrieved_slugs: list[str],
    generated_answer: str,
) -> dict[str, object]:
    if not settings.gemini_api_key:
        return unavailable_judge(
            "Gemini evaluation is unavailable because GEMINI_API_KEY is not set.",
            expected_answer_summary or "unavailable",
        )

    try:
        judge_client = GeminiJudgeClient(api_key=settings.gemini_api_key)
    except RuntimeError as error:
        return unavailable_judge(
            str(error),
            expected_answer_summary or "unavailable",
        )

    prompt = build_judge_prompt(
        question=question,
        expected_slugs=expected_slugs,
        expected_answer_summary=expected_answer_summary,
        retrieved_slugs=retrieved_slugs,
        generated_answer=generated_answer,
    )

    try:
        raw_response = judge_client.generate_json(
            prompt=prompt,
            model=settings.gemini_judge_model,
        )
    except Exception as error:  # pragma: no cover - external service call
        evaluate_logger.warning("Gemini judge failed: %s", error)
        return unavailable_judge(
            f"Gemini judge failed: {error}",
            expected_answer_summary or "unavailable",
        )

    payload = parse_json_object(raw_response)
    if payload is None:
        return unavailable_judge(
            "Gemini judge returned invalid JSON.",
            expected_answer_summary or "unavailable",
        )

    validated_payload = validate_judge_payload(payload)
    if validated_payload is None:
        return unavailable_judge(
            "Gemini judge response failed validation.",
            expected_answer_summary or "unavailable",
        )

    return {
        "answer_evaluation": {
            "expected_answer_summary": expected_answer_summary or "unavailable",
            "alignment_score": validated_payload["alignment_score"],
            "alignment_reason": validated_payload["alignment_reason"],
        },
        "judge": {
            "retrieval_assessment": validated_payload["retrieval_assessment"],
            "answer_assessment": validated_payload["answer_assessment"],
            "overall_note": validated_payload["overall_note"],
        },
    }


def evaluate_single_result(
    *,
    settings: Settings,
    question: str,
    expected_item: dict[str, object] | None,
    retrieved_slugs: list[str],
    generated_answer: str,
) -> dict[str, object]:
    expected_slugs = []
    if expected_item:
        raw_expected_slugs = expected_item.get("expected_slugs")
        if isinstance(raw_expected_slugs, list):
            expected_slugs = [str(slug) for slug in raw_expected_slugs]
        elif isinstance(expected_item.get("expected_slug"), str):
            expected_slugs = [str(expected_item["expected_slug"])]
    expected_answer_summary = (
        expected_item.get("expected_answer_summary") if expected_item else None
    )
    relevant_retrieved_slugs = [
        slug for slug in retrieved_slugs if slug in expected_slugs
    ]
    relevant_retrieved_count = len(set(relevant_retrieved_slugs))
    expected_count = len(set(expected_slugs))
    recall_at_k = (
        round(relevant_retrieved_count / expected_count, 4)
        if expected_count
        else None
    )

    fused_result = {
        "retrieval": {
            "expected_slugs": expected_slugs,
            "retrieved_slugs": retrieved_slugs,
            "relevant_retrieved_slugs": list(dict.fromkeys(relevant_retrieved_slugs)),
            "relevant_retrieved_count": relevant_retrieved_count,
            "expected_count": expected_count,
            "recall_at_k": recall_at_k,
        }
    }
    fused_result.update(
        judge_result(
            settings=settings,
            question=question,
            expected_slugs=expected_slugs or None,
            expected_answer_summary=expected_answer_summary,
            retrieved_slugs=retrieved_slugs,
            generated_answer=generated_answer,
        )
    )
    return fused_result


def evaluate_request_record(
    result_path: Path,
    ask_eval_set_path: Path = DEFAULT_ASK_EVAL_SET_PATH,
) -> dict[str, object]:
    settings = load_settings()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    request_record = payload.get("request_record", {})
    internal = request_record.get("internal", {})
    question = str(request_record.get("question", ""))
    answer = str(internal.get("answer", "")).strip()
    retrieved_matches = internal.get("retrieved_matches") or internal.get("ranked_matches") or internal.get("matches", [])
    retrieved_slugs = [str(match.get("doc_id", "")) for match in retrieved_matches]

    expected_item = find_expected_item(question, ask_eval_set_path)
    evaluation_result = evaluate_single_result(
        settings=settings,
        question=question,
        expected_item=expected_item,
        retrieved_slugs=retrieved_slugs,
        generated_answer=answer,
    )

    payload["evaluation_result"] = evaluation_result
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    update_ask_summary(result_path, evaluation_result)
    evaluate_logger.info("Updated request evaluation at %s", result_path)
    return evaluation_result


def update_ask_summary(
    result_path: Path,
    evaluation_result: dict[str, object],
) -> None:
    ask_day_dir = result_path.parent.parent
    summary_path = ask_day_dir / "summary.json"
    if not summary_path.exists():
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    request_timestamp = result_path.stem
    requests = summary.get("requests", [])

    for item in requests:
        if item.get("timestamp") == request_timestamp:
            item["evaluation_summary"] = {
                "recall_at_k": evaluation_result.get("retrieval", {}).get("recall_at_k"),
                "alignment_score": evaluation_result.get("answer_evaluation", {}).get("alignment_score"),
                "overall_note": evaluation_result.get("judge", {}).get("overall_note", ""),
            }
            break

    evaluated_requests = [
        item for item in requests if isinstance(item.get("evaluation_summary"), dict)
    ]
    recall_values = [
        item["evaluation_summary"].get("recall_at_k")
        for item in evaluated_requests
        if isinstance(item["evaluation_summary"].get("recall_at_k"), int | float)
    ]
    alignment_values = [
        item["evaluation_summary"].get("alignment_score")
        for item in evaluated_requests
        if isinstance(item["evaluation_summary"].get("alignment_score"), int)
    ]

    summary["evaluated_requests"] = len(evaluated_requests)
    summary["average_recall_at_k"] = (
        round(sum(recall_values) / len(recall_values), 4) if recall_values else None
    )
    summary["average_alignment_score"] = (
        round(sum(alignment_values) / len(alignment_values), 2)
        if alignment_values
        else None
    )

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def run_evaluation(
    eval_set_path: Path = DEFAULT_EVAL_SET_PATH,
    top_k: int = 3,
) -> dict[str, object]:
    settings = load_settings()
    retriever = Retriever(settings)
    resolved_eval_set_path = (
        eval_set_path if eval_set_path.is_absolute() else settings.project_root / eval_set_path
    )
    items = load_eval_items(resolved_eval_set_path)

    evaluate_logger.info(
        "Running fused evaluation with %s items from %s at top_k=%s",
        len(items),
        resolved_eval_set_path,
        top_k,
    )

    rows: list[dict[str, object]] = []
    total_hits = 0

    for item in items:
        question = item["question"]
        matches = retriever.search(question, top_k=top_k)
        retrieved_slugs = [str(match.get("doc_id", "")) for match in matches]
        generated_answer = safe_generate_answer_text(question, settings)

        row = {
            "question": question,
            "generated_answer": generated_answer,
        }
        row.update(
            evaluate_single_result(
                settings=settings,
                question=question,
                expected_item=item,
                retrieved_slugs=retrieved_slugs,
                generated_answer=generated_answer,
            )
        )
        rows.append(row)

        row_recall = row["retrieval"]["recall_at_k"]
        if isinstance(row_recall, int | float):
            total_hits += row["retrieval"]["relevant_retrieved_count"]
        evaluate_logger.info(
            "Question evaluated expected=%s retrieved=%s recall_at_k=%s",
            item.get("expected_slugs", []),
            retrieved_slugs,
            row_recall,
        )

    total_questions = len(rows)
    total_relevant_documents = 0
    for row in rows:
        total_relevant_documents += int(row["retrieval"].get("expected_count", 0))
    recall_at_k = round(total_hits / total_relevant_documents, 4) if total_relevant_documents else 0.0
    table = build_results_table(rows, top_k=top_k)

    result = {
        "metric": f"recall@{top_k}",
        "total_questions": total_questions,
        "total_relevant_documents": total_relevant_documents,
        "total_relevant_retrieved": total_hits,
        "recall_at_k": recall_at_k,
        "results": rows,
        "markdown_table": table,
    }

    output_path = settings.results_dir / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    evaluate_logger.info("Evaluation results saved to %s", output_path)
    return result


def main() -> None:
    result = run_evaluation()
    print(
        f"Recall@3: {result['total_relevant_retrieved']}/{result['total_relevant_documents']} = "
        f"{result['recall_at_k']:.4f}"
    )


if __name__ == "__main__":
    main()
