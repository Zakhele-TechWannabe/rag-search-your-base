import json


class PromptState:
    RANK = "rank"
    REFLECT = "reflect"
    ANSWER = "answer"


class PromptManager:
    def build(
        self,
        state: str,
        query: str,
        matches: list[dict[str, str | int | float]],
    ) -> str:
        if state == PromptState.RANK:
            return self._build_rank_prompt(query, matches)
        if state == PromptState.REFLECT:
            return self._build_reflect_prompt(query, matches)
        if state == PromptState.ANSWER:
            return self._build_answer_prompt(query, matches)
        raise ValueError(f"Unknown prompt state: {state}")

    def _build_rank_prompt(
        self,
        query: str,
        matches: list[dict[str, str | int | float]],
    ) -> str:
        candidates = []
        for match in matches:
            candidates.append(
                {
                    "chunk_id": match["chunk_id"],
                    "title": match["title"],
                    "url": match["url"],
                    "text": match["text"],
                    "distance": match["distance"],
                }
            )

        return (
            "You are ranking retrieved context for relevance.\n"
            "Given a user question and candidate chunks, return JSON only.\n"
            "Return a JSON object with one key: rankings.\n"
            "rankings must be an array of objects with keys: chunk_id, score.\n"
            "Scores must be integers from 1 to 10, where 10 is most relevant.\n"
            "Sort the rankings array from most relevant to least relevant.\n\n"
            f"Question:\n{query}\n\n"
            f"Candidates:\n{json.dumps(candidates, ensure_ascii=True)}"
        )

    def _build_answer_prompt(
        self,
        query: str,
        matches: list[dict[str, str | int | float]],
    ) -> str:
        context_parts = []
        for match in matches:
            context_parts.append(
                f"Title: {match['title']}\nURL: {match['url']}\nText: {match['text']}"
            )

        return (
            "You are answering questions using retrieved source material.\n"
            "Use only the provided context.\n"
            "Return JSON only with keys: answer, reasoning, confidence_score, needs_retry, used_chunk_ids, citations.\n"
            "reasoning must be a short explanation of why the answer is supported by the context.\n"
            "confidence_score must be an integer from 0 to 100.\n"
            "needs_retry must be true if the context looks insufficient.\n\n"
            "citations are compulsory.\n"
            "citations must be an array with at least one item when you provide an answer.\n"
            "Each citation must be a string in exactly this format: (title)[url].\n"
            "Only cite sources that appear in the provided context.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n\n{chr(10).join(context_parts)}"
        )

    def _build_reflect_prompt(
        self,
        query: str,
        matches: list[dict[str, str | int | float]],
    ) -> str:
        context_parts = []
        for match in matches:
            context_parts.append(
                f"[{match['chunk_id']}] {match['title']}\n{match['text']}"
            )

        return (
            "You are checking whether the retrieved context is sufficient to answer the question.\n"
            "Return JSON only with keys: confidence, needs_retry, reason.\n"
            "confidence must be one of: high, medium, low.\n"
            "needs_retry must be true when the context is weak, incomplete, or off-topic.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n\n{chr(10).join(context_parts)}"
        )
