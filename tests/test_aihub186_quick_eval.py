import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from aihub186_quick_eval import (
    build_relative_path_lookup,
    extract_category_eval_set,
    select_validation_sample,
)


def _write_orgtext(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8-sig")


def test_select_validation_sample_filters_category_and_split(tmp_path):
    orgtext_file = tmp_path / "orgtext.tsv"
    _write_orgtext(
        orgtext_file,
        [
            {"file_label": "A1", "category1": "대학병원", "category2": "진료안내", "category3": "외래", "speaker_type": "고객", "orgtext": "문장1", "dataset_type": "validation"},
            {"file_label": "A2", "category1": "대학병원", "category2": "진료안내", "category3": "외래", "speaker_type": "고객", "orgtext": "문장2", "dataset_type": "validation"},
            {"file_label": "A3", "category1": "대학병원", "category2": "진료안내", "category3": "외래", "speaker_type": "고객", "orgtext": "문장3", "dataset_type": "train"},
            {"file_label": "A4", "category1": "대학병원", "category2": "진료안내", "category3": "입원", "speaker_type": "고객", "orgtext": "문장4", "dataset_type": "validation"},
        ],
    )

    sampled = select_validation_sample(orgtext_file, "대학병원", "진료안내", "외래", n=2, seed=42)

    assert len(sampled) == 2
    assert set(sampled["file_label"]) == {"A1", "A2"}


def test_select_validation_sample_raises_when_not_enough_rows(tmp_path):
    orgtext_file = tmp_path / "orgtext.tsv"
    _write_orgtext(
        orgtext_file,
        [
            {"file_label": "A1", "category1": "대학병원", "category2": "진료안내", "category3": "외래", "speaker_type": "고객", "orgtext": "문장1", "dataset_type": "validation"},
        ],
    )

    with pytest.raises(ValueError, match="validation 표본이"):
        select_validation_sample(orgtext_file, "대학병원", "진료안내", "외래", n=5, seed=42)


def test_build_relative_path_lookup_filters_domain_and_split(tmp_path):
    wav_info_file = tmp_path / "wav_info.tsv"
    pd.DataFrame(
        [
            {"file_id": "A1", "relative_path": "dom/A1.wav", "duration_sec": 1.0, "dataset_type": "validation", "domain": "대학병원"},
            {"file_id": "A2", "relative_path": "dom/A2.wav", "duration_sec": 1.0, "dataset_type": "train", "domain": "대학병원"},
            {"file_id": "B1", "relative_path": "dom/B1.wav", "duration_sec": 1.0, "dataset_type": "validation", "domain": "광역이동지원센터"},
        ]
    ).to_csv(wav_info_file, sep="\t", index=False, encoding="utf-8-sig")

    lookup = build_relative_path_lookup(wav_info_file, "대학병원")

    assert lookup == {"A1": "dom/A1.wav"}


def test_extract_category_eval_set_writes_callcenter_compatible_layout(tmp_path):
    zip_path = tmp_path / "source.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dom/A1.wav", b"wavbytes1")
        zf.writestr("dom/A2.wav", b"wavbytes2")

    sampled = pd.DataFrame(
        [
            {"file_label": "A1", "orgtext": "첫번째 문장"},
            {"file_label": "A2", "orgtext": "두번째 문장"},
        ]
    )
    lookup = {"A1": "dom/A1.wav", "A2": "dom/A2.wav"}
    out_dir = tmp_path / "out"

    extracted = extract_category_eval_set(sampled, lookup, zip_path, out_dir)

    assert extracted == 2
    assert (out_dir / "wav" / "A1.wav").read_bytes() == b"wavbytes1"
    assert (out_dir / "wav" / "A2.wav").read_bytes() == b"wavbytes2"

    label = json.loads((out_dir / "labels" / "A1.json").read_text(encoding="utf-8"))
    assert label == {"utterances": ["첫번째 문장"]}


def test_extract_category_eval_set_raises_on_missing_lookup(tmp_path):
    zip_path = tmp_path / "source.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dom/A1.wav", b"wavbytes1")

    sampled = pd.DataFrame([{"file_label": "MISSING", "orgtext": "문장"}])
    out_dir = tmp_path / "out"

    with pytest.raises(KeyError):
        extract_category_eval_set(sampled, {}, zip_path, out_dir)
