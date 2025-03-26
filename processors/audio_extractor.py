import os
import tempfile

import ffmpeg

from processors.base_processor import BaseProcessor
from workflow.state import State


class AudioExtractor(BaseProcessor):
    """
    Extracts audio from video files using ffmpeg and converts it to WAV format.
    """

    def _process_implementation(self, state: State) -> State:
        print("Extracting audio from video...")
        video_path = state.video_path

        # Create a temporary file with .wav extension
        temp_dir = tempfile.gettempdir()
        temp_filename = os.path.basename(os.path.splitext(video_path)[0]) + ".wav"
        audio_path = os.path.join(temp_dir, temp_filename)

        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(
            stream,
            audio_path,
            acodec="pcm_s16le",  # Audio codec for WAV format
            ac=1,  # Mono channel for better compatibility
            ar="16k",  # Audio sample rate
            map="0:a",  # Select first audio stream from input
            loglevel="error",
        )
        ffmpeg.run(stream, overwrite_output=True)
        print(f"Audio extracted to {audio_path}")
        return State(**{**state.model_dump(), "audio_path": audio_path})
