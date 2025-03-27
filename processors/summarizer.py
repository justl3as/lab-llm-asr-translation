from langchain.prompts import PromptTemplate

from processors.base_processor import BaseProcessor
from utils import prompt_template
from workflow.state import State


class Summarizer(BaseProcessor):
    """Processor for text summarization"""

    @property
    def _load_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["context"],
            template=prompt_template.SUMMARIZATION_TEMPLATE,
        )

    def _process_implementation(self, state: State) -> State:
        self.logger.info("Creating context summary...")
        context = state.context or ""

        llm = self.config.get_llm_model()

        prompt_template = self._load_prompt_template.format(context=context)
        response = llm.invoke(prompt_template)
        summarized_context = response.content

        # Track token usage
        self.track_token_usage(state, response)

        self.logger.info("Context summary created successfully")

        return State(
            **{
                **state.model_dump(),
                "context": summarized_context,
                "metadata": {
                    **state.metadata,
                    "summarized_context": summarized_context,
                },
            }
        )
