import asyncio
from asyncio import Semaphore

import nest_asyncio
from langchain.prompts import PromptTemplate

from processors.base_processor import BaseProcessor
from workflow.state import State


class WhisperTranslator(BaseProcessor):
    """Processor for text translation"""

    TEMPLATE = """
You are an expert in Software Engineer, specializing in translating subtitles.

Your task is to accurately translate English subtitles into Thai.

Instructions:
- Preserve the original meaning and flow of natural spoken Thai.
- Remove any unnecessary words while preserving the original meaning.
- The input is a string of subtitle segments, separated by the delimiter `[SSS]`.
- Ensure Output `[SSS]` is equal to Input `[SSS]`.
- DO NOT translate or modify the delimiter `[SSS]`. Keep it exactly as is between segments.
- Do NOT translate proper names (e.g., people's names, brand names) or technical terms (e.g., programming syntax, tool names, AI-related terms).
- Do NOT add any explanation, formatting, or commentary.
- Line Count: Limit the subtitle to a maximum of 2 lines.
- Character Limits: Adjust each line to have between 43 and 50 characters.

Example:
- data input
text1
[SSS]
text2
[SSS]
text3
- data output
translated_text1
[SSS]
translated_text2
[SSS]
translated_text3

your task is to translate the following segments:
{context}
"""

    @property
    def _load_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["context"],
            template=self.TEMPLATE,
        )

    async def _process_batch(
        self,
        batch,
        context,
        semaphore: Semaphore,
        batch_index: int,
        state: State,
        max_retries: int = 3,
    ):
        async with semaphore:
            print(f"[Batch {batch_index}] Starting processing ({len(batch)} segments)")

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

                    print(f"[Batch {batch_index}] Successfully processed all segments")
                    return translated_batch

                except ValueError as e:
                    if (
                        "Mismatch in translated batch size" in str(e)
                        and attempt < max_retries - 1
                    ):
                        print(f"[Batch {batch_index}] {str(e)}. Retrying...")
                        await asyncio.sleep(2)
                        continue
                    raise
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(
                            f"[Batch {batch_index}] Error on attempt {attempt + 1}: {str(e)}. Retrying..."
                        )
                        await asyncio.sleep(2)
                        continue
                    print(f"[Batch {batch_index}] Failed after {max_retries} attempts")
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

        print(
            f"[Batch {batch_index}] Sending request to LLM "
            f"(attempt {attempt + 1}/{max_retries})"
        )
        response = llm.invoke(prompt)
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
        print("Translating context...")
        batch_size = self.config.batch_size
        concurrent_batches = self.config.concurrent_batches or 3
        context = state.context or ""

        segments = state.metadata["transcribed_segments"]
        segment_batches = [
            segments[i : i + batch_size] for i in range(0, len(segments), batch_size)
        ]

        print(
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

        print("Translation segments completed.")

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

    TEMPLATE = """
You are an expert in communication, language editing, and translation.

Below is a transcript generated by a speech-to-text system (Whisper):
"{context}"

Instructions:
1. Correct any incorrect or misspelled words.
3. Preserve the original meaning as much as possible.
3. Translate the improved version into Thai.

Output Only the final improved and translated Thai version of the text.
"""

    @property
    def _load_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["context"],
            template=self.TEMPLATE,
        )

    def _process_implementation(self, state: State) -> State:
        print(f"Translating context using {self.config.llm_model_name} models...")
        context = state.context or ""

        llm = self.config.get_llm_model()

        prompt_template = self._load_prompt_template.format(context=context)
        response = llm.invoke(prompt_template)
        translated_context = response.content

        # Track token usage
        self.track_token_usage(state, response)

        print("Translation context completed.")

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
