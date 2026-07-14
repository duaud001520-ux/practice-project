import json
from pathlib import Path


def load_medical_pairs(data_dir: Path) -> list[tuple[Path, str]]:
    data_dir = Path(data_dir)
    wav_dir = data_dir / "wav"
    label_dir = data_dir / "labels"

    pairs: list[tuple[Path, str]] = []
    for wav_path in sorted(wav_dir.glob("*.wav")):
        json_path = label_dir / f"{wav_path.stem}.json"
        if not json_path.exists():
            continue
        with open(json_path, encoding="utf-8") as f:
            label = json.load(f)
        transcript = label["전사정보"]["LabelText"]
        pairs.append((wav_path, transcript))
    return pairs
