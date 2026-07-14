import argparse
from pathlib import Path
import tarfile

DOWNLOADS = Path("/mnt/c/Users/visionlab/Downloads/한국인 대화 음성/Training")

CATEGORIES = {
    "weather_03": {
        "source": DOWNLOADS / "[원천]5.날씨_weather_03.tar.gz",
        "label": DOWNLOADS / "[라벨]5.날씨_weather_03.tar.gz",
    },
    "hobby_01": {
        "source": DOWNLOADS / "[원천]2.취미_hobby_01.tar.gz",
        "label": DOWNLOADS / "[라벨]2.취미_hobby_01.tar.gz",
    },
}


def extract_category(source_tar: Path, label_tar: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for tar_path in (source_tar, label_tar):
        with tarfile.open(tar_path) as tf:
            tf.extractall(dest, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path.home() / "data" / "kspon"))
    args = parser.parse_args()

    out_root = Path(args.out)
    for name, cfg in CATEGORIES.items():
        extract_category(cfg["source"], cfg["label"], out_root / name)
        print(f"[{name}] extracted to {out_root / name}")


if __name__ == "__main__":
    main()
