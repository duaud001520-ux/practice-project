import json
import zipfile
from pathlib import Path
from typing import Iterator, NamedTuple

from extract_callcenter import DOWNLOADS as DOWNLOADS_ROOT
from extract_callcenter import BYTES_PER_SEC, WAV_HEADER_BYTES

# domain -> {split: {"label_zip": ..., "source_zip": ...}}
DOMAINS: dict[str, dict[str, dict[str, Path]]] = {
    "대학병원": {
        "train": {
            "label_zip": DOWNLOADS_ROOT / "1.Training" / "라벨링데이터" / "TL1_01.대학병원.zip",
            "source_zip": DOWNLOADS_ROOT / "1.Training" / "원천데이터" / "TS1_01.대학병원.zip",
        },
        "validation": {
            "label_zip": DOWNLOADS_ROOT / "2.Validation" / "라벨링데이터" / "VL1_01.대학병원.zip",
            "source_zip": DOWNLOADS_ROOT / "2.Validation" / "원천데이터" / "VS1_01.대학병원.zip",
        },
    },
    "광역이동지원센터": {
        "train": {
            "label_zip": DOWNLOADS_ROOT / "1.Training" / "라벨링데이터" / "TL2_02.광역이동지원센터.zip",
            "source_zip": DOWNLOADS_ROOT / "1.Training" / "원천데이터" / "TS2_02.광역이동지원센터.zip",
        },
        "validation": {
            "label_zip": DOWNLOADS_ROOT / "2.Validation" / "라벨링데이터" / "VL2_02.광역이동지원센터.zip",
            "source_zip": DOWNLOADS_ROOT / "2.Validation" / "원천데이터" / "VS2_02.광역이동지원센터.zip",
        },
    },
    "정신건강복지센터": {
        "train": {
            "label_zip": DOWNLOADS_ROOT / "1.Training" / "라벨링데이터" / "TL3_03.정신건강복지센터.zip",
            "source_zip": DOWNLOADS_ROOT / "1.Training" / "원천데이터" / "TS3_03.정신건강복지센터.zip",
        },
        "validation": {
            "label_zip": DOWNLOADS_ROOT / "2.Validation" / "라벨링데이터" / "VL3_03.정신건강복지센터.zip",
            "source_zip": DOWNLOADS_ROOT / "2.Validation" / "원천데이터" / "VS3_03.정신건강복지센터.zip",
        },
    },
}


class LabelRow(NamedTuple):
    file_label: str
    category1: str
    category2: str
    category3: str
    speaker_type: str
    orgtext: str


class LabelError(NamedTuple):
    json_name: str
    error_type: str
    error_message: str


def parse_label_entry(json_name: str, raw: bytes) -> LabelRow:
    """json_name 항목 하나를 파싱해 LabelRow를 반환. 문제가 있으면 ValueError."""
    try:
        data = json.loads(raw)
    except Exception as e:
        raise ValueError(f"JSON 파싱 실패: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"최상위 구조가 객체가 아닙니다: {type(data).__name__}")

    input_texts = data.get("inputText")
    if not isinstance(input_texts, list) or not input_texts:
        raise ValueError("inputText가 없거나 배열 형식이 아니거나 비어 있습니다.")

    orgtext = input_texts[0].get("orgtext") if isinstance(input_texts[0], dict) else None
    if not isinstance(orgtext, str) or not orgtext.strip():
        raise ValueError("inputText[0].orgtext가 없거나 비어 있습니다.")

    info_list = data.get("info")
    if not isinstance(info_list, list) or not info_list:
        raise ValueError("info가 없거나 비어 있거나 배열 형식이 아닙니다.")

    metadata = info_list[0].get("metadata") if isinstance(info_list[0], dict) else None
    if not isinstance(metadata, dict):
        raise ValueError("info[0].metadata가 없거나 객체 형식이 아닙니다.")

    required = {}
    for field in ("category1", "category2", "category3", "speaker_type"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"metadata.{field} 값이 없거나 비어 있습니다.")
        required[field] = value.strip()

    file_label = Path(json_name).stem
    cleaned_text = orgtext.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
    if not cleaned_text:
        raise ValueError("orgtext 정리 후 빈 문자열입니다.")

    return LabelRow(
        file_label=file_label,
        category1=required["category1"],
        category2=required["category2"],
        category3=required["category3"],
        speaker_type=required["speaker_type"],
        orgtext=cleaned_text,
    )


def iter_label_entries(label_zip: Path) -> Iterator[tuple[LabelRow | None, LabelError | None]]:
    """label_zip 내 모든 .json을 순회하며 (LabelRow, None) 또는 (None, LabelError)를 yield."""
    with zipfile.ZipFile(label_zip) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".json"):
                continue
            try:
                raw = zf.read(name)
                row = parse_label_entry(name, raw)
                yield row, None
            except (ValueError, zipfile.BadZipFile, OSError) as e:
                yield None, LabelError(json_name=name, error_type=type(e).__name__, error_message=str(e))


class WavInfoRow(NamedTuple):
    file_id: str
    relative_path: str
    duration_sec: float
    sample_rate: int
    channels: int
    sample_width_bits: int
    frame_count: int
    file_size_bytes: int


def wav_info_from_zipinfo(info: zipfile.ZipInfo) -> WavInfoRow:
    """zip 중앙 디렉터리 메타데이터만으로 wav 정보를 계산 (압축 해제 없음).

    16kHz/16bit/mono 고정 헤더(44바이트) 가정 — 실제 데이터로 검증됨
    (scripts/aihub186_wav_info.py의 표본 검증 단계 참고). file_size가 헤더보다
    작으면(손상/잘림 파일) ValueError.
    """
    file_size = info.file_size
    if file_size < WAV_HEADER_BYTES:
        raise ValueError(
            f"wav 파일 크기({file_size}바이트)가 헤더 크기({WAV_HEADER_BYTES}바이트)보다 작습니다: {info.filename}"
        )
    frame_count = (file_size - WAV_HEADER_BYTES) // 2
    duration_sec = frame_count / 16000
    return WavInfoRow(
        file_id=Path(info.filename).stem,
        relative_path=info.filename,
        duration_sec=duration_sec,
        sample_rate=16000,
        channels=1,
        sample_width_bits=16,
        frame_count=frame_count,
        file_size_bytes=file_size,
    )


def iter_wav_entries(source_zip: Path) -> Iterator[tuple[WavInfoRow | None, str | None]]:
    """source_zip 내 모든 .wav를 순회하며 (WavInfoRow, None) 또는 (None, 에러메시지)를 yield."""
    with zipfile.ZipFile(source_zip) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".wav"):
                continue
            try:
                yield wav_info_from_zipinfo(info), None
            except ValueError as e:
                yield None, str(e)


def sample_verify_wav_format(source_zip: Path, n: int = 20) -> list[tuple[str, bool]]:
    """zip 전체에 고르게 분포된 n개 wav 항목의 실제 헤더를 읽어 16kHz/16bit/mono +
    44바이트 헤더 가정을 모두 검증 (앞쪽 n개만 보면 뒤쪽에 몰린 이상치를 놓치므로
    전체 구간에서 등간격으로 표본을 뽑는다).

    반환: [(파일명, 가정과 일치 여부), ...]. wav가 하나도 없으면 빈 리스트.
    """
    import io
    import wave

    results = []
    with zipfile.ZipFile(source_zip) as zf:
        all_wav_names = [i.filename for i in zf.infolist() if i.filename.endswith(".wav")]
        if not all_wav_names:
            return results
        step = max(1, len(all_wav_names) // n)
        sample_names = all_wav_names[::step][:n]
        for name in sample_names:
            raw = zf.read(name)
            with wave.open(io.BytesIO(raw), "rb") as wf:
                data_bytes = wf.getnframes() * wf.getnchannels() * wf.getsampwidth()
                header_bytes = len(raw) - data_bytes
                matches = (
                    wf.getframerate() == 16000
                    and wf.getnchannels() == 1
                    and wf.getsampwidth() == 2
                    and header_bytes == WAV_HEADER_BYTES
                )
            results.append((name, matches))
    return results
