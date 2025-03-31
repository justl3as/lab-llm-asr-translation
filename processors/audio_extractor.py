import os
import re
import tempfile
import uuid

import yt_dlp

from processors.base_processor import BaseProcessor
from utils.process_audio import extract_audio_file
from workflow.state import State


class AudioExtractor(BaseProcessor):
    """
    Extracts audio from video files using ffmpeg and converts it to WAV format.
    Also supports extracting audio directly from YouTube URLs.
    """

    def _is_youtube_url(self, url: str) -> bool:
        """Check if the given string is a YouTube URL."""
        youtube_regex = r"(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})"
        return bool(re.match(youtube_regex, url))

    def _extract_audio_from_youtube_video(self, youtube_url: str) -> str:
        """Download a YouTube video and return the path to the downloaded file."""
        self.logger.info(f"Downloading YouTube video: {youtube_url}")
        temp_dir = tempfile.gettempdir()
        uuid_str = str(uuid.uuid4())
        output_path = os.path.join(temp_dir, f"{uuid_str}")
        audio_path = os.path.join(temp_dir, uuid_str + ".wav")
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "postprocessor_args": [
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16k",
                "-ac",
                "1",
            ],
            "outtmpl": output_path,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        if not os.path.exists(audio_path):
            raise FileNotFoundError(
                f"Audio extraction failed. The expected audio file was not found at {audio_path}. "
                f"Please check if the YouTube URL is valid and accessible: {youtube_url}."
            )

        self.logger.info(f"YouTube audio extracted to {audio_path}")
        return audio_path

    def _extract_audio_from_video_file(self, file_path: str) -> str:
        """Extract audio from video file and save as WAV format."""
        self.logger.info("Extracting audio from video...")

        audio_path = extract_audio_file(file_path)

        self.logger.info(f"Audio extracted to {audio_path}")
        return audio_path

    def _process_implementation(self, state: State) -> State:
        video_path = state.video_path
        audio_path = state.audio_path

        if audio_path:
            self.logger.info("Audio already extracted. Skipping extraction step.")
            return State(**{**state.model_dump(), "audio_path": audio_path})
        elif self._is_youtube_url(video_path):
            audio_path = self._extract_audio_from_youtube_video(video_path)
        else:
            audio_path = self._extract_audio_from_video_file(video_path)

        return State(**{**state.model_dump(), "audio_path": audio_path})
