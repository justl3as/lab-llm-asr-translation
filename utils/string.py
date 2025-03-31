from pythainlp.tokenize import word_tokenize


def warp_text(text: str) -> str:
    # Tokenize the text into a list of words
    tokens = word_tokenize(text, engine="newmm")
    lines = []
    current_line = ""

    warp = len(text)
    BASE_LENGTH = 42
    ratio = warp / BASE_LENGTH

    if ratio <= 1.5:
        max_chars = warp
    elif ratio <= 2.5:
        max_chars = BASE_LENGTH
    elif ratio <= 3:
        max_chars = warp / 2
    else:
        max_chars = warp / 3

    for token in tokens:
        # Check if adding the new word exceeds the maximum character limit
        if len(current_line + token) <= max_chars:
            current_line += token
        else:
            # If the limit is reached, save the current line and start a new one
            lines.append(current_line)
            current_line = token
    # Save the last line if it exists
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)


def combine_texts(texts: list) -> str:
    """
    Combine a list of texts into a single string, ensuring that each text is separated by a newline character.
    """
    combined_text = " ".join(texts)
    return combined_text
