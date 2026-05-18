import math
import re

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)


def tokenize_text(text: str) -> list[str]:
    lowered = text.lower()
    base_tokens = TOKEN_PATTERN.findall(lowered)
    cjk_chars = [token for token in base_tokens if len(token) == 1 and "\u4e00" <= token <= "\u9fff"]
    bigrams = [left + right for left, right in zip(cjk_chars, cjk_chars[1:])]
    return list(dict.fromkeys(base_tokens + bigrams))


def score_lexical_match(
    query: str,
    query_tokens: set[str],
    chunk_text: str,
    chunk_tokens: set[str],
) -> float:
    if not chunk_tokens:
        return 0.0

    overlap = query_tokens & chunk_tokens
    if not overlap and query.lower() not in chunk_text.lower():
        return 0.0

    score = len(overlap) / math.sqrt(len(chunk_tokens))
    if query.lower() in chunk_text.lower():
        score += 1.5

    return round(score, 4)
