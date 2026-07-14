import re

import jiwer


def compute_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    return {
        "wer": jiwer.wer(references, hypotheses),
        "cer": jiwer.cer(references, hypotheses),
    }


def normalize_text(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_repetition(text: str) -> str | None:
    """Flag Whisper-style hallucination: same short word/phrase repeated 3+ times in a row."""
    match = re.search(r"(\S{1,20})(?:\s+\1){2,}", text)
    return match.group(0) if match else None
