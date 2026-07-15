import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd

from aihub186_data import DOMAINS


def select_validation_sample(
    orgtext_file: Path, category1: str, category2: str, category3: str, n: int, seed: int
) -> pd.DataFrame:
    df = pd.read_csv(orgtext_file, sep="\t", encoding="utf-8-sig")
    filtered = df[
        (df["category1"] == category1)
        & (df["category2"] == category2)
        & (df["category3"] == category3)
        & (df["dataset_type"] == "validation")
    ]
    if len(filtered) < n:
        raise ValueError(
            f"{category1}>{category2}>{category3} validation 표본이 {len(filtered)}개뿐입니다 (요청 {n}개)"
        )
    return filtered.sample(n=n, random_state=seed)[["file_label", "orgtext"]].reset_index(drop=True)


def build_relative_path_lookup(wav_info_file: Path, domain: str, dataset_type: str = "validation") -> dict[str, str]:
    df = pd.read_csv(wav_info_file, sep="\t", encoding="utf-8-sig")
    filtered = df[(df["domain"] == domain) & (df["dataset_type"] == dataset_type)]
    return dict(zip(filtered["file_id"], filtered["relative_path"]))


def extract_category_eval_set(
    sampled: pd.DataFrame,
    relative_path_lookup: dict[str, str],
    source_zip: Path,
    out_dir: Path,
) -> int:
    """load_callcenter_pairs 호환 형식(wav/ + labels/{"utterances": [...]})으로 추출."""
    wav_dir = out_dir / "wav"
    label_dir = out_dir / "labels"
    wav_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    with zipfile.ZipFile(source_zip) as zf:
        for _, row in sampled.iterrows():
            file_label = row["file_label"]
            relative_path = relative_path_lookup.get(file_label)
            if relative_path is None:
                raise KeyError(f"wav_info.tsv에서 file_id={file_label}를 찾을 수 없습니다")

            data = zf.read(relative_path)
            (wav_dir / f"{file_label}.wav").write_bytes(data)
            (label_dir / f"{file_label}.json").write_text(
                json.dumps({"utterances": [row["orgtext"]]}, ensure_ascii=False),
                encoding="utf-8",
            )
            extracted += 1

    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category1", required=True)
    parser.add_argument("--category2", required=True)
    parser.add_argument("--category3", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--orgtext", default=str(Path.home() / "data" / "callcenter" / "analysis" / "orgtext.tsv")
    )
    parser.add_argument(
        "--wav-info", default=str(Path.home() / "data" / "callcenter" / "analysis" / "wav_info.tsv")
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.category1 not in DOMAINS:
        raise SystemExit(f"알 수 없는 category1: {args.category1} (가능: {list(DOMAINS)})")

    sampled = select_validation_sample(
        Path(args.orgtext), args.category1, args.category2, args.category3, args.n, args.seed
    )
    lookup = build_relative_path_lookup(Path(args.wav_info), args.category1)
    source_zip = DOMAINS[args.category1]["validation"]["source_zip"]

    extracted = extract_category_eval_set(sampled, lookup, source_zip, Path(args.out))
    print(f"{args.category1}>{args.category2}>{args.category3}: {extracted}개 추출 완료 -> {args.out}")


if __name__ == "__main__":
    main()
