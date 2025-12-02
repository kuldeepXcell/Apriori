from openai import OpenAI
from app.core.config import settings
from typing import List

class LLMService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates a dense vector embedding for the given text.
        """
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text],
            model=settings.EMBEDDING_MODEL
        )
        return response.data[0].embedding

    def get_chat_completion(self, messages: List[dict], model: str = "gpt-4o") -> str:
        """
        Generates a chat completion response.
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content

llm_service = LLMService()
