import os

from utils.logging import setup_logger
from utils.string import warp_text
from utils.time import format_timestamp


class SRTFormatter:
    """Formatter for SRT files"""

    def __init__(self):
        self.logger = setup_logger(f"{self.__class__.__name__}")

    def format_and_save(self, translated_segments, output_path):
        # Create a folder for the output file if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as srt_file:
            previous_end = 0  # Track the end time of the previous segment

            len_segments = len(translated_segments)
            for i, segment in enumerate(translated_segments, start=1):
                start = max(segment["start"], previous_end)
                end = max(segment["end"], start + 0.001)

                start_time = format_timestamp(start)
                end_time = format_timestamp(end)
                text = warp_text(segment["text"])

                previous_end = end

                srt_file.write(f"{i}\n")
                srt_file.write(f"{start_time} --> {end_time}\n")
                srt_file.write(f"{text}")

                if i < len_segments:
                    srt_file.write("\n\n")
        self.logger.info(f"Subtitles generated successfully: {output_path}")
        return output_path
