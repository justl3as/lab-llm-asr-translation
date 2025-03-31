from pythainlp.tokenize import word_tokenize


def warp_text(text: str, min_length: int = 44, ratio: float = 2.0) -> str:
    """
    Wrap Thai text into multiple lines using word tokenization.
    """
    if not text:
        return ""

    tokens = word_tokenize(text, engine="newmm")
    if not tokens:
        return ""

    max_line_length = max(len(text) // ratio, min_length)
    lines = []
    current_line = ""

    for token in tokens:
        if len(current_line + token) <= max_line_length:
            current_line += token
        else:
            if current_line:
                lines.append(current_line)
            current_line = token

    if current_line:
        lines.append(current_line)

    # More efficient line joining
    if not lines:
        return ""
    elif len(lines) == 1:
        return lines[0]
    elif len(lines) == 2:
        if len(lines[len(lines) - 1]) < max_line_length // 2:
            return f"{''.join(lines[0:])}"
        else:
            return "\n".join(lines)
    else:
        return f"{lines[0]}\n{''.join(lines[1:])}"


def combine_texts(texts: list) -> str:
    """
    Combine a list of texts into a single string, ensuring that each text is separated by a newline character.
    """
    combined_text = " ".join(texts)
    return combined_text
