def should_retry(current_attempt: int, max_attempts: int) -> bool:
    return current_attempt < max_attempts
