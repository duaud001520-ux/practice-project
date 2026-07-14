import json
from pathlib import Path

from dataset import load_medical_pairs


def test_load_medical_pairs_matches_wav_to_transcript(tmp_path):
    wav_dir = tmp_path / "wav"
    label_dir = tmp_path / "labels"
    wav_dir.mkdir()
    label_dir.mkdir()

    (wav_dir / "SPK1-1-A.wav").write_bytes(b"dummy")
    (label_dir / "SPK1-1-A.json").write_text(
        json.dumps({"전사정보": {"LabelText": "안녕하세요"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    # 라벨 없는 wav는 건너뛴다
    (wav_dir / "SPK1-2-A.wav").write_bytes(b"dummy2")

    pairs = load_medical_pairs(tmp_path)

    assert len(pairs) == 1
    wav_path, transcript = pairs[0]
    assert wav_path.name == "SPK1-1-A.wav"
    assert transcript == "안녕하세요"
