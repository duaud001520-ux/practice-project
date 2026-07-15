import json
import struct
import sys
import wave
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from aihub186_data import (
    iter_label_entries,
    iter_wav_entries,
    parse_label_entry,
    sample_verify_wav_format,
    wav_info_from_zipinfo,
)


def _make_label_json(orgtext="안녕하세요", category1="대학병원", category2="병원이용안내", category3="원무상담", speaker_type="고객"):
    return json.dumps(
        {
            "inputText": [{"orgtext": orgtext}],
            "info": [
                {
                    "metadata": {
                        "category1": category1,
                        "category2": category2,
                        "category3": category3,
                        "speaker_type": speaker_type,
                    }
                }
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")


def test_parse_label_entry_extracts_fields():
    raw = _make_label_json()
    row = parse_label_entry("HOS123A001.json", raw)

    assert row.file_label == "HOS123A001"
    assert row.category1 == "대학병원"
    assert row.category2 == "병원이용안내"
    assert row.category3 == "원무상담"
    assert row.speaker_type == "고객"
    assert row.orgtext == "안녕하세요"


def test_parse_label_entry_strips_newlines_and_tabs():
    raw = _make_label_json(orgtext="안녕\n하세요\t감사합니다\r")
    row = parse_label_entry("x.json", raw)
    assert "\n" not in row.orgtext
    assert "\t" not in row.orgtext
    assert "\r" not in row.orgtext


def test_parse_label_entry_rejects_missing_category():
    raw = json.dumps(
        {
            "inputText": [{"orgtext": "안녕"}],
            "info": [{"metadata": {"category1": "대학병원"}}],
        }
    ).encode("utf-8")
    with pytest.raises(ValueError, match="category2"):
        parse_label_entry("x.json", raw)


def test_parse_label_entry_rejects_empty_orgtext():
    raw = _make_label_json(orgtext="   ")
    with pytest.raises(ValueError):
        parse_label_entry("x.json", raw)


def test_parse_label_entry_rejects_malformed_json():
    with pytest.raises(ValueError, match="JSON 파싱 실패"):
        parse_label_entry("x.json", b"{not valid json")


def test_iter_label_entries_yields_rows_and_skips_errors(tmp_path):
    zip_path = tmp_path / "labels.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a/HOS1.json", _make_label_json(orgtext="첫번째"))
        zf.writestr("a/HOS2.json", b"{bad json")
        zf.writestr("a/HOS3.json", _make_label_json(orgtext="세번째"))

    rows = []
    errors = []
    for row, error in iter_label_entries(zip_path):
        if row is not None:
            rows.append(row)
        else:
            errors.append(error)

    assert len(rows) == 2
    assert {r.orgtext for r in rows} == {"첫번째", "세번째"}
    assert len(errors) == 1
    assert errors[0].json_name == "a/HOS2.json"
    assert errors[0].error_type == "ValueError"


def test_iter_label_entries_records_bad_zip_entry_instead_of_crashing(tmp_path, monkeypatch):
    zip_path = tmp_path / "labels.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a/HOS1.json", _make_label_json(orgtext="첫번째"))
        zf.writestr("a/HOS2.json", _make_label_json(orgtext="두번째"))

    import aihub186_data

    original_read = zipfile.ZipFile.read

    def _flaky_read(self, name, *args, **kwargs):
        if name == "a/HOS2.json":
            raise zipfile.BadZipFile("손상된 항목")
        return original_read(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", _flaky_read)

    rows = []
    errors = []
    for row, error in aihub186_data.iter_label_entries(zip_path):
        if row is not None:
            rows.append(row)
        else:
            errors.append(error)

    assert len(rows) == 1
    assert len(errors) == 1
    assert errors[0].error_type == "BadZipFile"


def _make_wav_bytes(frames: int, sample_rate: int = 16000, channels: int = 1, sampwidth: int = 2) -> bytes:
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{frames}h", *([0] * frames)))
    return buf.getvalue()


def test_wav_info_from_zipinfo_computes_duration_from_metadata_only(tmp_path):
    frames = 16000 * 3  # 3초 분량
    wav_bytes = _make_wav_bytes(frames)
    zip_path = tmp_path / "source.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dom/HOS1.wav", wav_bytes)

    with zipfile.ZipFile(zip_path) as zf:
        info = zf.getinfo("dom/HOS1.wav")
        row = wav_info_from_zipinfo(info)

    assert row.file_id == "HOS1"
    assert row.relative_path == "dom/HOS1.wav"
    assert row.sample_rate == 16000
    assert row.channels == 1
    assert row.sample_width_bits == 16
    assert row.frame_count == frames
    assert row.duration_sec == pytest.approx(3.0, abs=0.01)


def test_wav_info_from_zipinfo_rejects_file_smaller_than_header(tmp_path):
    zip_path = tmp_path / "source.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dom/broken.wav", b"short")  # 5바이트 < 44바이트 헤더

    with zipfile.ZipFile(zip_path) as zf:
        info = zf.getinfo("dom/broken.wav")
        with pytest.raises(ValueError, match="헤더 크기"):
            wav_info_from_zipinfo(info)


def test_iter_wav_entries_reads_all_wav_in_zip(tmp_path):
    zip_path = tmp_path / "source.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dom/HOS1.wav", _make_wav_bytes(16000))
        zf.writestr("dom/HOS2.wav", _make_wav_bytes(32000))
        zf.writestr("dom/readme.txt", b"not audio")

    rows = []
    errors = []
    for row, error in iter_wav_entries(zip_path):
        if row is not None:
            rows.append(row)
        else:
            errors.append(error)

    assert len(rows) == 2
    assert len(errors) == 0
    by_id = {r.file_id: r for r in rows}
    assert by_id["HOS1"].duration_sec == pytest.approx(1.0, abs=0.01)
    assert by_id["HOS2"].duration_sec == pytest.approx(2.0, abs=0.01)


def test_iter_wav_entries_yields_error_for_undersized_wav(tmp_path):
    zip_path = tmp_path / "source.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dom/HOS1.wav", _make_wav_bytes(16000))
        zf.writestr("dom/broken.wav", b"short")

    rows = []
    errors = []
    for row, error in iter_wav_entries(zip_path):
        if row is not None:
            rows.append(row)
        else:
            errors.append(error)

    assert len(rows) == 1
    assert len(errors) == 1
    assert "broken.wav" in errors[0]


def test_sample_verify_wav_format_spreads_across_whole_zip_not_just_first_n(tmp_path):
    zip_path = tmp_path / "source.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i in range(100):
            zf.writestr(f"dom/HOS{i:03d}.wav", _make_wav_bytes(1600))

    results = sample_verify_wav_format(zip_path, n=10)
    names = [name for name, _ in results]

    assert len(results) == 10
    assert all(ok for _, ok in results)
    # 앞쪽 10개(HOS000~HOS009)에만 몰려있지 않고 뒤쪽까지 퍼져 있어야 함
    assert any(name > "dom/HOS050.wav" for name in names)


def _make_wav_bytes_with_extra_chunk(
    frames: int, sample_rate: int = 16000, channels: int = 1, sampwidth: int = 2
) -> bytes:
    """fmt와 data 사이에 LIST 청크를 끼워 넣은 wav (wave 모듈로는 정상 포맷으로 읽히지만
    실제 헤더 길이는 표준 44바이트가 아님)."""
    audio_data = struct.pack(f"<{frames}h", *([0] * frames))
    byte_rate = sample_rate * channels * sampwidth
    block_align = channels * sampwidth

    fmt_chunk = (
        b"fmt "
        + struct.pack("<I", 16)
        + struct.pack("<HHIIHH", 1, channels, sample_rate, byte_rate, block_align, sampwidth * 8)
    )
    list_chunk = b"LIST" + struct.pack("<I", 4) + b"xxxx"
    data_chunk = b"data" + struct.pack("<I", len(audio_data)) + audio_data

    body = b"WAVE" + fmt_chunk + list_chunk + data_chunk
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_sample_verify_wav_format_flags_extra_header_bytes_as_mismatch(tmp_path):
    tampered = _make_wav_bytes_with_extra_chunk(1600)

    # wave 모듈은 LIST 청크를 건너뛰고 fmt/data를 정상적으로 읽는다 (형식 자체는 맞음)
    import io

    with wave.open(io.BytesIO(tampered), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2

    zip_path = tmp_path / "source.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dom/HOS1.wav", tampered)

    results = sample_verify_wav_format(zip_path, n=1)

    assert len(results) == 1
    name, ok = results[0]
    assert ok is False


def test_sample_verify_wav_format_returns_empty_list_when_no_wav(tmp_path):
    zip_path = tmp_path / "source.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dom/readme.txt", b"no audio here")

    assert sample_verify_wav_format(zip_path, n=10) == []
