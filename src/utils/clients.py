import chromadb
from openai import OpenAI
from google import genai



class LLMClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.client = OpenAI(api_key=api_key)

    def generate_text(
        self,
        prompt: str,
        model: str = "gpt-5-mini",
        json_mode: bool = False,
    ) -> str:
        request = {
            "model": model,
            "input": prompt,
        }
        if json_mode:
            request["text"] = {"format": {"type": "json_object"}}

        response = self.client.responses.create(**request)
        return response.output_text

    def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small",
    ) -> list[float]:
        response = self.client.embeddings.create(
            input=text,
            model=model,
        )
        return response.data[0].embedding


class ChromaClient:
    def __init__(self, path: str = "data/chroma") -> None:
        self.client = chromadb.PersistentClient(path=path)

    def get_or_create_collection(self, name: str):
        return self.client.get_or_create_collection(name=name)

    def heartbeat(self) -> int:
        return self.client.heartbeat()


class GeminiJudgeClient:
    def __init__(self, api_key: str | None = None) -> None:
        if genai is None:
            raise RuntimeError("google-genai is not installed.")
        self.client = genai.Client(api_key=api_key)

    def generate_json(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
    ) -> str:
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
            },
        )
        return response.text
