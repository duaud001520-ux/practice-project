from pathlib import Path

from eval import run_eval


class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModel:
    def __init__(self, hyp_by_stem: dict[str, str]) -> None:
        self.hyp_by_stem = hyp_by_stem

    def transcribe(self, path: str, language: str):
        stem = Path(path).stem
        return [_FakeSegment(self.hyp_by_stem[stem])], None


def test_run_eval_computes_metrics_from_pairs(tmp_path):
    wav1 = tmp_path / "a.wav"
    wav2 = tmp_path / "b.wav"
    wav1.write_bytes(b"x")
    wav2.write_bytes(b"x")
    pairs = [(wav1, "안녕 하세요"), (wav2, "반갑 습니다")]

    model = _FakeModel({"a": "안녕 하세요", "b": "반갑 습니다"})
    result = run_eval(pairs, model)

    assert result["wer"] == 0.0
    assert result["cer"] == 0.0
