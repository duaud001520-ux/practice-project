import argparse
from pathlib import Path
import zipfile

DOWNLOADS = Path(
    "/mnt/c/Users/visionlab/Downloads/비대면 진료를 위한 의료진 및 환자 음성"
)

SPLITS = {
    "eval": {
        "source_zip": DOWNLOADS / "Validation" / "[V원천]의료진_간호사_1.zip",
        "label_zip": DOWNLOADS / "Validation" / "[V]라벨링데이터.zip",
        "label_prefix": "medv/간호사/",
    },
    "train": {
        "source_zip": DOWNLOADS / "Training" / "[T원천]의료진_간호사_1.zip",
        "label_zip": DOWNLOADS / "Training" / "[T]라벨링데이터.zip",
        "label_prefix": "medsub/간호사/",
    },
}


def extract_split(
    source_zip: Path, label_zip: Path, label_prefix: str, out_dir: Path
) -> tuple[int, int]:
    wav_dir = out_dir / "wav"
    label_dir = out_dir / "labels"
    wav_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    wav_count = 0
    with zipfile.ZipFile(source_zip) as zf:
        for info in zf.infolist():
            if info.filename.endswith(".wav"):
                target = wav_dir / Path(info.filename).name
                with zf.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                wav_count += 1

    label_count = 0
    with zipfile.ZipFile(label_zip) as zf:
        for info in zf.infolist():
            if info.filename.startswith(label_prefix) and info.filename.endswith(".json"):
                target = label_dir / Path(info.filename).name
                with zf.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                label_count += 1

    return wav_count, label_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["eval", "train", "both"], default="both")
    parser.add_argument("--out", default=str(Path.home() / "data" / "medical"))
    args = parser.parse_args()

    out_root = Path(args.out)
    splits = ["eval", "train"] if args.split == "both" else [args.split]
    for split in splits:
        cfg = SPLITS[split]
        wav_count, label_count = extract_split(
            cfg["source_zip"], cfg["label_zip"], cfg["label_prefix"], out_root / split
        )
        print(f"[{split}] wav={wav_count} label={label_count}")


if __name__ == "__main__":
    main()
