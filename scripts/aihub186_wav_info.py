import argparse
import csv
from pathlib import Path

from aihub186_data import DOMAINS, iter_wav_entries, sample_verify_wav_format


def verify_all_domains(sample_n: int) -> bool:
    """모든 도메인/split의 원천 zip에서 표본 wav 헤더를 읽어 16kHz/16bit/mono+44바이트 헤더 가정을 검증."""
    all_ok = True
    for domain, splits in DOMAINS.items():
        for dataset_type, paths in splits.items():
            source_zip = paths["source_zip"]
            results = sample_verify_wav_format(source_zip, n=sample_n)
            if not results:
                all_ok = False
                print(f"[검증 실패] {domain}/{dataset_type}: wav 항목을 하나도 찾지 못했습니다 ({source_zip.name})")
                continue
            mismatches = [name for name, ok in results if not ok]
            if mismatches:
                all_ok = False
                print(f"[검증 실패] {domain}/{dataset_type}: {len(mismatches)}/{len(results)}건 형식 불일치")
                for name in mismatches[:5]:
                    print(f"    {name}")
            else:
                print(f"[검증 통과] {domain}/{dataset_type}: 표본 {len(results)}건 모두 16kHz/16bit/mono+44바이트 헤더 일치")
    return all_ok


def build_wav_info_tsv(output_file: Path, error_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    error_file.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    error_count = 0
    total_duration_sec = 0.0

    with (
        output_file.open("w", encoding="utf-8-sig", newline="") as out_f,
        error_file.open("w", encoding="utf-8-sig", newline="") as err_f,
    ):
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(
            [
                "file_id",
                "relative_path",
                "duration_sec",
                "sample_rate",
                "channels",
                "sample_width_bits",
                "frame_count",
                "file_size_bytes",
                "dataset_type",
                "domain",
            ]
        )
        error_writer = csv.writer(err_f, delimiter="\t")
        error_writer.writerow(["domain", "dataset_type", "error_message"])

        for domain, splits in DOMAINS.items():
            for dataset_type, paths in splits.items():
                source_zip = paths["source_zip"]
                print(f"[{domain}/{dataset_type}] 원천 zip 처리 중: {source_zip.name}")
                domain_rows = 0
                domain_errors = 0
                domain_duration = 0.0
                for row, error in iter_wav_entries(source_zip):
                    if row is not None:
                        writer.writerow(
                            [
                                row.file_id,
                                row.relative_path,
                                f"{row.duration_sec:.6f}",
                                row.sample_rate,
                                row.channels,
                                row.sample_width_bits,
                                row.frame_count,
                                row.file_size_bytes,
                                dataset_type,
                                domain,
                            ]
                        )
                        row_count += 1
                        domain_rows += 1
                        domain_duration += row.duration_sec
                    else:
                        error_writer.writerow([domain, dataset_type, error])
                        error_count += 1
                        domain_errors += 1
                total_duration_sec += domain_duration
                print(f"  -> {domain_rows:,}개 wav, {domain_duration / 3600:.2f}시간, 오류 {domain_errors:,}건")

    print()
    print(f"총 wav 수: {row_count:,}")
    print(f"총 오류 수: {error_count:,}")
    print(f"총 길이: {total_duration_sec / 3600:,.2f}시간")
    print(f"출력 파일: {output_file}")
    print(f"오류 파일: {error_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default=str(Path.home() / "data" / "callcenter" / "analysis" / "wav_info.tsv")
    )
    parser.add_argument(
        "--errors", default=str(Path.home() / "data" / "callcenter" / "analysis" / "wav_errors.tsv")
    )
    parser.add_argument(
        "--skip-verify", action="store_true", help="형식 표본 검증을 건너뛰고 바로 전체 처리"
    )
    parser.add_argument("--verify-sample-n", type=int, default=20)
    args = parser.parse_args()

    if not args.skip_verify:
        ok = verify_all_domains(args.verify_sample_n)
        if not ok:
            raise SystemExit(
                "형식 가정(16kHz/16bit/mono/44바이트 헤더)과 다른 wav가 발견되어 중단합니다. "
                "aihub186_data.py의 duration 계산 로직을 다시 검토하세요."
            )

    build_wav_info_tsv(Path(args.out), Path(args.errors))


if __name__ == "__main__":
    main()
