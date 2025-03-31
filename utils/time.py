import wave

import webrtcvad


def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds / 3600)
    seconds %= 3600
    minutes = int(seconds / 60)
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"


def fix_first_speech_timestamp(wav_path, frame_duration_ms=30) -> float:
    """
    Returns the timestamp (in seconds) of the first frame that contains speech.
    """
    vad = webrtcvad.Vad(3)

    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        frame_samples = int(sample_rate * frame_duration_ms / 1000)
        frame_bytes = frame_samples * 2

        timestamp = 0.0
        while True:
            frame = wf.readframes(frame_samples)
            if len(frame) < frame_bytes:
                break
            if vad.is_speech(frame, sample_rate):
                return timestamp
            timestamp += frame_duration_ms / 1000.0
    return timestamp
