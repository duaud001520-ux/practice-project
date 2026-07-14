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
