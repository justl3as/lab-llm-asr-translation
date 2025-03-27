import asyncio
from asyncio import Semaphore

import nest_asyncio
from langchain.prompts import PromptTemplate

from processors.base_processor import BaseProcessor
from utils import prompt_template
from workflow.state import State


class WhisperTranslator(BaseProcessor):
    """Processor for text translation"""

    @property
    def _load_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["context"],
            template=prompt_template.WHISPER_TRANSLATOR_TEMPLATE,
        )

    async def _process_batch(
        self,
        batch,
        context,
        semaphore: Semaphore,
        batch_index: int,
        state: State,
        max_retries: int = 5,
    ):
        async with semaphore:
            self.logger.info(
                f"[Batch {batch_index}] Starting processing ({len(batch)} segments)"
            )

            # Prepare the input text
            segment_texts = self._prepare_batch_text(batch)

            # Get the prompt ready for LLM
            prompt = self._load_prompt_template.format(context=segment_texts)

            # Process with retries
            for attempt in range(max_retries):
                try:
                    translated_batch = await self._translate_segments(
                        prompt,
                        batch,
                        batch_index,
                        attempt,
                        max_retries,
                        state,
                    )

                    self.logger.info(
                        f"[Batch {batch_index}] Successfully processed all segments"
                    )
                    return translated_batch

                except ValueError as e:
                    if (
                        "Mismatch in translated batch size" in str(e)
                        and attempt < max_retries - 1
                    ):
                        self.logger.warning(
                            f"[Batch {batch_index}] {str(e)}. Retrying..."
                        )
                        await asyncio.sleep(2)
                        continue
                    raise
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(
                            f"[Batch {batch_index}] Error on attempt {attempt + 1}: {str(e)}. Retrying..."
                        )
                        await asyncio.sleep(2)
                        continue
                    self.logger.error(
                        f"[Batch {batch_index}] Failed after {max_retries} attempts"
                    )
                    raise

    def _prepare_batch_text(self, batch):
        """Extract and join segment texts with delimiter"""
        batch_texts = [segment["text"].strip() for segment in batch]
        return "\n[SSS]\n".join(batch_texts)

    async def _translate_segments(
        self,
        prompt,
        batch,
        batch_index,
        attempt,
        max_retries,
        state: State,
    ):
        """Send request to LLM and process the response"""
        llm = self.config.get_llm_model()

        self.logger.info(
            f"[Batch {batch_index}] Sending request to LLM "
            f"(attempt {attempt + 1}/{max_retries})"
        )
        # Determine the prompt to use based on the attempt number
        if attempt == 0:
            enhanced_prompt = prompt
        else:
            enhanced_prompt = prompt_template.RETRIES_WHISPER_TEMPLATE.format(
                prompt=prompt,
                batch_length=len(batch),
            )

        response = llm.invoke(enhanced_prompt)
        translated_segment_texts = str(response.content).strip()
        translated_batch_texts = translated_segment_texts.split("\n[SSS]\n")

        # Track token usage
        self.track_token_usage(state, response)

        # Validate response
        if len(translated_batch_texts) != len(batch):
            raise ValueError(
                f"Mismatch in translated batch size on attempt {attempt + 1}. "
                f"Expected {len(batch)}, got {len(translated_batch_texts)}"
            )

        # Format the results
        return [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": translated_text.strip(),
            }
            for segment, translated_text in zip(batch, translated_batch_texts)
        ]

    async def _process_implementation_async(self, state: State) -> State:
        self.logger.info("Translating context...")
        batch_size = self.config.batch_size
        concurrent_batches = self.config.concurrent_batches or 3
        context = state.context or ""

        segments = state.metadata["transcribed_segments"]
        segment_batches = [
            segments[i : i + batch_size] for i in range(0, len(segments), batch_size)
        ]

        self.logger.info(
            f"Processing {len(segment_batches)} batches with max "
            f"{concurrent_batches} concurrent tasks"
        )
        max_concurrent = concurrent_batches

        semaphore = Semaphore(max_concurrent)
        tasks = [
            self._process_batch(batch, context, semaphore, idx, state)
            for idx, batch in enumerate(segment_batches, 1)
        ]
        results = await asyncio.gather(*tasks)
        translated_segments = [
            item for sublist in results if sublist is not None for item in sublist
        ]

        self.logger.info("Translation segments completed.")

        return State(
            **{
                **state.model_dump(),
                "context": "Final translated context",
                "metadata": {
                    **state.metadata,
                    "translated_segments": translated_segments,
                },
            }
        )

    def _process_implementation(self, state: State) -> State:
        nest_asyncio.apply()

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._process_implementation_async(state))


class ContextTranslator(BaseProcessor):
    """Processor for text translation"""

    @property
    def _load_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["context"],
            template=prompt_template.CONTEXT_TRANSLATOR_TEMPLATE,
        )

    def _process_implementation(self, state: State) -> State:
        self.logger.info(
            f"Translating context using {self.config.llm_model_name} models..."
        )
        context = state.context or ""

        llm = self.config.get_llm_model()

        prompt_template = self._load_prompt_template.format(context=context)
        response = llm.invoke(prompt_template)
        translated_context = response.content

        # Track token usage
        self.track_token_usage(state, response)

        self.logger.info("Translation context completed.")

        return State(
            **{
                **state.model_dump(),
                "context": translated_context,
                "metadata": {
                    **state.metadata,
                    "translated_context": translated_context,
                },
            }
        )
