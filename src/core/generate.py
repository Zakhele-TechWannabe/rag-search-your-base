from utils.clients import LLMClient
from utils.config import Settings
from utils.logger import get_logger
from utils.prompts import PromptManager, PromptState
from utils.retriever import Retriever
from utils.validators import (
    parse_json_object,
    validate_answer_payload,
    validate_rank_payload,
    validate_reflect_payload,
)


generate_logger = get_logger("generate", folder="pipeline")


class AnswerGenerator:
    def __init__(
        self,
        settings: Settings,
        max_iterations: int = 3,
        retrieval_k: int = 8,
        context_k: int = 8,
        model: str = "gpt-5-mini",
    ) -> None:
        self.settings = settings
        self.max_iterations = max_iterations
        self.retrieval_k = retrieval_k
        self.context_k = context_k
        self.model = model
        self.llm_client = LLMClient(api_key=settings.openai_api_key)
        self.prompt_manager = PromptManager()
        self.retriever = Retriever(settings)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, str | int | float]]:
        limit = top_k or self.retrieval_k
        return self.retriever.search(query, top_k=limit)

    def rank(
        self,
        query: str,
        matches: list[dict[str, str | int | float]],
    ) -> list[dict[str, str | int | float]]:
        if not matches:
            generate_logger.warning("No matches available for ranking")
            return []
        state = PromptState.RANK
        prompt = self.prompt_manager.build(state=state, query=query, matches=matches)

        raw_response = self.llm_client.generate_text(
            prompt,
            model=self.model,
            json_mode=True,
        )
        generate_logger.debug("Raw %s response: %s", state, raw_response)

        payload = parse_json_object(raw_response)
        if payload is None:
            generate_logger.warning("Ranking response was not valid JSON; using retrieval order")
            return matches[: self.context_k]

        ranked_items = validate_rank_payload(payload)
        score_by_chunk = {}
        for item in ranked_items:
            chunk_id = item.get("chunk_id")
            score = item.get("score")
            if isinstance(chunk_id, str) and isinstance(score, int | float):
                score_by_chunk[chunk_id] = float(score)

        if not score_by_chunk:
            generate_logger.warning("Ranking response had no usable scores; using retrieval order")
            return matches[: self.context_k]

        ranked_matches = sorted(
            matches,
            key=lambda match: score_by_chunk.get(str(match["chunk_id"]), 0.0),
            reverse=True,
        )
        scored_matches = []
        for match in ranked_matches[: self.context_k]:
            scored_match = dict(match)
            scored_match["rank_score"] = score_by_chunk.get(str(match["chunk_id"]), 0.0)
            scored_matches.append(scored_match)

        generate_logger.info("Ranked %s matches", len(matches))
        generate_logger.debug("Ranked matches with scores: %s", scored_matches)
        return scored_matches

    def select_context(
        self,
        ranked_matches: list[dict[str, str | int | float]],
    ) -> list[dict[str, str | int | float]]:
        if not ranked_matches:
            return []

        selected_matches = [
            match for match in ranked_matches if float(match.get("rank_score", 0.0)) >= 7.0
        ]

        if not selected_matches and float(ranked_matches[0].get("rank_score", 0.0)) >= 6.0:
            selected_matches = [ranked_matches[0]]

        if not selected_matches and len(ranked_matches) > 1:
            top_score = float(ranked_matches[0].get("rank_score", 0.0))
            second_score = float(ranked_matches[1].get("rank_score", 0.0))
            if top_score >= 5.0 and second_score >= 5.0:
                selected_matches = ranked_matches[:2]

        if not selected_matches and len(ranked_matches) > 0:
            top_score = float(ranked_matches[0].get("rank_score", 0.0))
            if top_score >= 4.0:
                selected_matches = [ranked_matches[0]]

        selected_matches = selected_matches[: self.context_k]
        generate_logger.info(
            "Selected %s ranked matches for answer context (max=%s)",
            len(selected_matches),
            self.context_k,
        )
        generate_logger.debug("Selected answer context: %s", selected_matches)
        return selected_matches

    def reflect(
        self,
        query: str,
        matches: list[dict[str, str | int | float]],
    ) -> dict[str, object]:
        if not matches:
            return {
                "confidence": "low",
                "needs_retry": True,
                "reason": "No relevant retrieved matches were selected for answering.",
            }

        state = PromptState.REFLECT
        prompt = self.prompt_manager.build(state=state, query=query, matches=matches)
        raw_response = self.llm_client.generate_text(
            prompt,
            model=self.model,
            json_mode=True,
        )
        generate_logger.debug("Raw %s response: %s", state, raw_response)

        payload = parse_json_object(raw_response)
        if payload is None:
            generate_logger.warning("Reflect response was not valid JSON")
            return {
                "confidence": "low",
                "needs_retry": True,
                "reason": "Reflect response was not valid JSON.",
            }

        validated_payload = validate_reflect_payload(payload)
        if validated_payload is None:
            generate_logger.warning("Reflect response failed validation")
            return {
                "confidence": "low",
                "needs_retry": True,
                "reason": "Reflect response failed validation.",
            }

        return validated_payload

    def answer(
        self,
        query: str,
        matches: list[dict[str, str | int | float]],
    ) -> dict[str, object]:
        if not matches:
            return {
                "answer": "I don't know based on the provided context.",
                "reasoning": "No relevant context was selected for answering.",
                "confidence_score": 0,
                "needs_retry": True,
                "used_chunk_ids": [],
                "citations": [],
            }
        state = PromptState.ANSWER
        prompt = self.prompt_manager.build(state=state, query=query, matches=matches)

        raw_response = self.llm_client.generate_text(
            prompt,
            model=self.model,
            json_mode=True,
        )
        generate_logger.debug("Raw %s response: %s", state, raw_response)

        payload = parse_json_object(raw_response)
        if payload is None:
            generate_logger.warning("Answer response was not valid JSON")
            return {
                "answer": raw_response,
                "reasoning": "The model did not return valid JSON, so the raw response was preserved.",
                "confidence_score": 0,
                "needs_retry": True,
                "used_chunk_ids": [],
                "citations": [],
            }

        validated_payload = validate_answer_payload(payload)
        if validated_payload is None:
            generate_logger.warning("Answer response failed validation")
            return {
                "answer": payload.get("answer", "") if isinstance(payload.get("answer"), str) else "",
                "reasoning": payload.get("reasoning", "") if isinstance(payload.get("reasoning"), str) else "",
                "confidence_score": 0,
                "needs_retry": True,
                "used_chunk_ids": payload.get("used_chunk_ids", []) if isinstance(payload.get("used_chunk_ids"), list) else [],
                "citations": [],
            }

        valid_citations = validated_payload["citations"]
        if not valid_citations:
            generate_logger.warning("Answer response did not include valid citations")
            return {
                "answer": validated_payload["answer"],
                "reasoning": validated_payload["reasoning"],
                "confidence_score": 0,
                "needs_retry": True,
                "used_chunk_ids": validated_payload["used_chunk_ids"],
                "citations": [],
            }

        return {
            "answer": validated_payload["answer"],
            "reasoning": validated_payload["reasoning"],
            "confidence_score": validated_payload["confidence_score"],
            "needs_retry": validated_payload["needs_retry"],
            "used_chunk_ids": validated_payload["used_chunk_ids"],
            "citations": valid_citations,
        }

    def execute(self, query: str) -> dict[str, object]:
        generate_logger.info("Starting answer generation for query=%s", query)

        last_result: dict[str, object] = {
            "answer": "",
            "reasoning": "",
            "confidence_score": 0,
            "needs_retry": True,
            "used_chunk_ids": [],
            "citations": [],
            "retrieved_matches": [],
            "ranked_matches": [],
            "matches": [],
        }

        for iteration in range(1, self.max_iterations + 1):
            retrieval_k = self.retrieval_k
            generate_logger.info("Generation iteration %s with retrieval_k=%s", iteration, retrieval_k)
            matches = self.retrieve(query, top_k=retrieval_k)
            ranked_matches = self.rank(query, matches)
            selected_matches = self.select_context(ranked_matches)
            reflection = self.reflect(query, selected_matches)
            confidence = str(reflection.get("confidence", "low")).lower()
            needs_retry = bool(reflection.get("needs_retry", True))

            if needs_retry and iteration < self.max_iterations:
                generate_logger.info(
                    "Reflection requested retry on iteration %s with confidence=%s",
                    iteration,
                    confidence,
                )
                last_result = {
                    "answer": "",
                    "reasoning": "",
                    "confidence_score": 0,
                    "needs_retry": needs_retry,
                    "used_chunk_ids": [],
                    "citations": [],
                    "retrieved_matches": matches,
                    "ranked_matches": ranked_matches,
                    "matches": selected_matches,
                    "reflection": reflection,
                    "iteration": iteration,
                }
                continue

            result = self.answer(query, selected_matches)
            result["reflection"] = reflection
            result["retrieved_matches"] = matches
            result["ranked_matches"] = ranked_matches
            result["matches"] = selected_matches
            result["iteration"] = iteration
            last_result = result

            confidence_score = int(result.get("confidence_score", 0))
            needs_retry = bool(result.get("needs_retry", False))
            generate_logger.info(
                "Iteration %s finished with confidence_score=%s needs_retry=%s",
                iteration,
                confidence_score,
                needs_retry,
            )

            if confidence_score >= 60 and not needs_retry:
                break

        return last_result


def generate_answer(question: str, settings: Settings) -> dict[str, object]:
    generator = AnswerGenerator(settings=settings)
    return generator.execute(question)
