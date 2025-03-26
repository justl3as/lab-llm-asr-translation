def format_timestamp(seconds):
    # """แปลงเวลาจากวินาทีเป็นรูปแบบ SRT (HH:MM:SS,mmm)"""
    # hours = int(seconds // 3600)
    # minutes = int((seconds % 3600) // 60)
    # secs = int(seconds % 60)
    # millisecs = int((seconds % 1) * 1000)
    # return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds / 3600)
    seconds %= 3600
    minutes = int(seconds / 60)
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
