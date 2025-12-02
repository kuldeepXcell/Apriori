from openai import OpenAI
from app.core.config import settings
from typing import List

class LLMService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def get_embedding(self, text: str, model: str = None) -> List[float]:
        """
        Generates a dense vector embedding for the given text.
        
        Args:
            text: Text to embed
            model: Embedding model name (defaults to settings.EMBEDDING_MODEL)
        """
        if model is None:
            model = settings.EMBEDDING_MODEL
            
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding

    def get_chat_completion(self, messages: List[dict], model: str = "gpt-4o", temperature: float = 0.0) -> str:
        """
        Generates a chat completion response.
        
        Args:
            messages: List of message dictionaries
            model: LLM model name
            temperature: Sampling temperature
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content

llm_service = LLMService()
