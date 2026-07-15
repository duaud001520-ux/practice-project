import argparse
import csv
from pathlib import Path

from aihub186_data import DOMAINS, iter_label_entries


def build_orgtext_tsv(output_file: Path, error_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    error_file.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    error_count = 0

    with (
        output_file.open("w", encoding="utf-8-sig", newline="") as out_f,
        error_file.open("w", encoding="utf-8-sig", newline="") as err_f,
    ):
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(
            ["file_label", "category1", "category2", "category3", "speaker_type", "orgtext", "dataset_type"]
        )
        error_writer = csv.writer(err_f, delimiter="\t")
        error_writer.writerow(["domain", "dataset_type", "json_name", "error_type", "error_message"])

        for domain, splits in DOMAINS.items():
            for dataset_type, paths in splits.items():
                label_zip = paths["label_zip"]
                print(f"[{domain}/{dataset_type}] 라벨 zip 처리 중: {label_zip.name}")
                domain_rows = 0
                domain_errors = 0
                for row, error in iter_label_entries(label_zip):
                    if row is not None:
                        writer.writerow(
                            [
                                row.file_label,
                                row.category1,
                                row.category2,
                                row.category3,
                                row.speaker_type,
                                row.orgtext,
                                dataset_type,
                            ]
                        )
                        row_count += 1
                        domain_rows += 1
                    else:
                        error_writer.writerow(
                            [domain, dataset_type, error.json_name, error.error_type, error.error_message]
                        )
                        error_count += 1
                        domain_errors += 1
                print(f"  -> 정상 {domain_rows:,}건, 오류 {domain_errors:,}건")

    print()
    print(f"총 저장 행 수: {row_count:,}")
    print(f"총 오류 건 수: {error_count:,}")
    print(f"출력 파일: {output_file}")
    print(f"오류 파일: {error_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default=str(Path.home() / "data" / "callcenter" / "analysis" / "orgtext.tsv")
    )
    parser.add_argument(
        "--errors", default=str(Path.home() / "data" / "callcenter" / "analysis" / "orgtext_errors.tsv")
    )
    args = parser.parse_args()
    build_orgtext_tsv(Path(args.out), Path(args.errors))


if __name__ == "__main__":
    main()
