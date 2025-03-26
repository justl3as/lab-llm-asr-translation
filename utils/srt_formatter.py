import os

from utils.time_utils import format_timestamp


class SRTFormatter:
    """Formatter for SRT files"""

    def format_and_save(self, translated_segments, output_path):
        # Create a folder for the output file if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as srt_file:
            previous_end = 0  # Track the end time of the previous segment

            for i, segment in enumerate(translated_segments, start=1):
                start = max(segment["start"], previous_end)
                end = max(segment["end"], start + 0.001)

                start_time = format_timestamp(start)
                end_time = format_timestamp(end)
                text = segment["text"].strip()

                previous_end = end

                srt_file.write(f"{i}\n")
                srt_file.write(f"{start_time} --> {end_time}\n")
                srt_file.write(f"{text}\n\n")

        print(f"Subtitles generated successfully: {output_path}")
        return output_path
