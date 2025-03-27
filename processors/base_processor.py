from abc import ABC, abstractmethod
from typing import Any

from langchain.prompts import PromptTemplate

from config.app_config import AppConfig
from utils.logging_utils import setup_logger
from workflow.state import State


class BaseProcessor(ABC):
    """Template Method Pattern for processors"""

    def __init__(self, node_name: str = "processor") -> None:
        self.config = AppConfig()
        self.node_name = node_name
        self.logger = setup_logger(f"{self.__class__.__name__}")

    def track_token_usage(self, state: State, response: Any) -> None:
        """Track token usage from LLM response metadata"""
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            state.add_token_usage_from_metadata(
                self.node_name,
                response.usage_metadata,
            )

    def process(self, state: State) -> State:
        """Template method that defines the outline of processing steps"""
        # Hook before processing
        state = self.before_process(state)

        # Main processing
        state = self._process_implementation(state)

        # Hook after processing
        state = self.after_process(state)

        return state

    @abstractmethod
    def _process_implementation(self, state: State) -> State:
        """Concrete implementations must override this method"""
        pass

    @property
    def _load_prompt_template(self) -> PromptTemplate:
        raise NotImplementedError("PromptTemplate")

    def before_process(self, state: State) -> State:
        """Hook called before processing"""
        return state

    def after_process(self, state: State) -> State:
        """Hook called after processing"""
        return state
