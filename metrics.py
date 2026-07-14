import jiwer


def compute_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    return {
        "wer": jiwer.wer(references, hypotheses),
        "cer": jiwer.cer(references, hypotheses),
    }
