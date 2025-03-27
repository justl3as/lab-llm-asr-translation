import os

import whisper

from processors.audio_extractor import AudioExtractor
from processors.base_processor import BaseProcessor
from workflow.state import State


class TranscribeAudio(BaseProcessor):
    """
    Transcribes audio files using OpenAI's Whisper model.
    """

    def before_process(self, state: State) -> State:
        if not state.audio_path or not os.path.exists(state.audio_path):
            print("No audio file found. Running AudioExtractor...")
            audio_extractor = AudioExtractor()
            state = audio_extractor.process(state)
        return state

    def extract_segments(self, transcript_segments):
        segments = []
        for segment in transcript_segments:
            segments.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": str(segment["text"]).strip(),
                }
            )
        return segments

    def _process_implementation(self, state: State) -> State:
        whisper_model_size = self.config.whisper_model_size
        print(f"Transcribing audio using Whisper {whisper_model_size} model...")

        audio_path = state.audio_path
        model = whisper.load_model(whisper_model_size)

        response = model.transcribe(audio_path, task="transcribe", fp16=False)
        transcribed_text = str(response["text"]).strip()
        transcribed_segments = self.extract_segments(response["segments"])

        print(
            f"Transcription completed successfully "
            f"({len(transcribed_segments)} segments, {len(transcribed_text)} characters)"
        )

        return State(
            **{
                **state.model_dump(),
                "context": transcribed_text,
                "metadata": {
                    **state.metadata,
                    "transcribed_text": transcribed_text,
                    "transcribed_segments": transcribed_segments,
                },
            }
        )
