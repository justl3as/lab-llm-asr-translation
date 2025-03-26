from abc import ABC, abstractmethod

from langchain.prompts import PromptTemplate

from config.app_config import AppConfig
from workflow.state import State


class BaseProcessor(ABC):
    """Template Method Pattern for processors"""

    def __init__(self):
        self.config = AppConfig()

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
