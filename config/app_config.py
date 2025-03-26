import os
from typing import Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


class AppConfig:
    _instance: Optional["AppConfig"] = None

    def __new__(cls) -> "AppConfig":
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._init_default_config()
        return cls._instance

    def _init_default_config(self) -> None:
        # Load environment variables from .env file
        load_dotenv()

        # Load configurations from environment variables with fallback defaults
        self.whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "base")
        self.llm_model_name: str = os.getenv("LLM_MODEL_NAME", "")
        self.llm_model_provider: str = os.getenv("LLM_MODEL_PROVIDER", "")
        self.temperature: float = float(os.getenv("TEMPERATURE", "0"))
        self.batch_size: int = int(os.getenv("BATCH_SIZE", "10"))
        self.concurrent_batches: int = int(os.getenv("CONCURRENT_BATCHES", "1"))

    def set_whisper_model(self, model_size: str) -> "AppConfig":
        self.whisper_model_size = model_size
        return self

    def set_llm_model_name(self, model_name: str) -> "AppConfig":
        self.llm_model_name = model_name
        return self

    def set_llm_model_provider(self, model_provider: str) -> "AppConfig":
        self.llm_model_provider = model_provider
        return self

    def set_temperature(self, temperature: float) -> "AppConfig":
        self.temperature = temperature
        return self

    def get_llm_model(self):
        if not self.llm_model_name:
            raise ValueError("LLM model name is not set")
        if not self.llm_model_provider:
            raise ValueError("LLM model provider is not set")

        return init_chat_model(
            model=self.llm_model_name,
            model_provider=self.llm_model_provider,
            temperature=self.temperature,
        )
