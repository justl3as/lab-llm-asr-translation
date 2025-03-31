import os
import tempfile

import ffmpeg


def process_audio(input_bytes, format_in="wav", format_out="wav", **options):
    process = (
        ffmpeg.input("pipe:", format=format_in)
        .output("pipe:", format=format_out, **options)
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    output_bytes, err = process.communicate(input=input_bytes)
    return output_bytes


def extract_audio_file(input_file: str):
    """
    Process audio with English spoken with German accent for optimal Whisper.
    """
    with open(input_file, "rb") as f:
        mp4_bytes = f.read()

    # Process audio in a single pipeline
    wav_bytes = process_audio(mp4_bytes, format_in="mp4", acodec="pcm_s16le")
    clean_bytes = process_audio(wav_bytes, af="afftdn")
    normalized_bytes = process_audio(clean_bytes, filter_complex="loudnorm")
    filtered_bytes = process_audio(normalized_bytes, af="highpass=f=480")
    final_bytes = process_audio(filtered_bytes, ar=16000, ac=1)

    # Create a temporary file with .wav extension
    temp_dir = tempfile.gettempdir()
    temp_filename = os.path.basename(os.path.splitext(input_file)[0]) + ".wav"
    audio_path = os.path.join(temp_dir, temp_filename)

    # Write the final processed audio to the temporary file
    with open(audio_path, "wb") as f:
        f.write(final_bytes)

    return audio_path
