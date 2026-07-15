import argparse
from pathlib import Path

import pandas as pd


def build_category_hours(orgtext_file: Path, wav_info_file: Path) -> dict[str, pd.DataFrame]:
    text_df = pd.read_csv(orgtext_file, sep="\t", encoding="utf-8-sig")
    wav_df = pd.read_csv(wav_info_file, sep="\t", encoding="utf-8-sig")

    df = text_df.merge(
        wav_df[["file_id", "duration_sec"]],
        left_on="file_label",
        right_on="file_id",
        how="inner",
    )
    df = df.drop_duplicates(subset="file_label")

    unmatched_text = len(text_df) - text_df["file_label"].isin(wav_df["file_id"]).sum()
    unmatched_wav = len(wav_df) - wav_df["file_id"].isin(text_df["file_label"]).sum()
    print(
        f"매칭 결과: orgtext {len(text_df):,}건 중 미매칭 {unmatched_text:,}건, "
        f"wav_info {len(wav_df):,}건 중 미매칭 {unmatched_wav:,}건 (inner join에서 제외됨)"
    )

    tables = {}
    for key, cols in [
        ("category1", ["category1"]),
        ("category1_2", ["category1", "category2"]),
        ("category1_2_3", ["category1", "category2", "category3"]),
    ]:
        group_cols = cols + ["dataset_type"]
        result = (
            df.groupby(group_cols)["duration_sec"]
            .agg(count="count", sum_sec="sum")
            .reset_index()
        )
        result["sum_hour"] = result["sum_sec"] / 3600
        result = result.sort_values(cols + ["dataset_type"]).reset_index(drop=True)
        tables[key] = result

    return tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orgtext", default=str(Path.home() / "data" / "callcenter" / "analysis" / "orgtext.tsv")
    )
    parser.add_argument(
        "--wav-info", default=str(Path.home() / "data" / "callcenter" / "analysis" / "wav_info.tsv")
    )
    parser.add_argument(
        "--out-dir", default=str(Path.home() / "data" / "callcenter" / "analysis")
    )
    args = parser.parse_args()

    tables = build_category_hours(Path(args.orgtext), Path(args.wav_info))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, table in tables.items():
        out_path = out_dir / f"category_hours_{key}.tsv"
        table.to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")
        print(f"=== {key} ===")
        print(table.to_string(index=False))
        print(f"-> {out_path}")
        print()


if __name__ == "__main__":
    main()
