from metrics import compute_metrics


def test_compute_metrics_perfect_match_is_zero():
    result = compute_metrics(["안녕 하세요"], ["안녕 하세요"])
    assert result["wer"] == 0.0
    assert result["cer"] == 0.0


def test_compute_metrics_detects_errors():
    result = compute_metrics(["안녕 하세요"], ["안녕 하십니까"])
    assert result["wer"] > 0.0
    assert result["cer"] > 0.0
