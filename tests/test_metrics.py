from metrics import compute_metrics, normalize_text


def test_compute_metrics_perfect_match_is_zero():
    result = compute_metrics(["안녕 하세요"], ["안녕 하세요"])
    assert result["wer"] == 0.0
    assert result["cer"] == 0.0


def test_compute_metrics_detects_errors():
    result = compute_metrics(["안녕 하세요"], ["안녕 하십니까"])
    assert result["wer"] > 0.0
    assert result["cer"] > 0.0


def test_normalize_text_strips_punctuation():
    assert normalize_text("안녕하세요, 반갑습니다!") == "안녕하세요 반갑습니다"


def test_normalize_text_collapses_whitespace():
    assert normalize_text("안녕   하세요\n반갑습니다") == "안녕 하세요 반갑습니다"
