from langchain.prompts import PromptTemplate

from processors.base_processor import BaseProcessor
from workflow.state import State


class Summarizer(BaseProcessor):
    """Processor for text summarization"""

    TEMPLATE = """
    You are an expert in summarizing content from video transcripts.

    Carefully read the following transcript and provide a concise summary that includes:
    - The main points discussed
    - The overall context or purpose of the conversation
    - Any key insights or noteworthy opinions

    Use clear and easy-to-understand language:

    Input:
    {context}
    """

    @property
    def _load_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["context"],
            template=self.TEMPLATE,
        )

    def _process_implementation(self, state: State) -> State:
        print("Creating context summary...")
        context = state.context or ""

        llm = self.config.get_llm_model()

        prompt_template = self._load_prompt_template.format(context=context)
        response = llm.invoke(prompt_template)
        summarized_context = response.content

        print("Context summary created successfully")

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
