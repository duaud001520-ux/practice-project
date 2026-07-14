import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_medical import extract_split


def _make_zip(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def test_extract_split_filters_by_prefix_and_extension(tmp_path):
    source_zip = tmp_path / "source.zip"
    label_zip = tmp_path / "label.zip"

    _make_zip(source_zip, {
        "nur/SPK1/SPK1-1-A.wav": b"wavdata1",
        "nur/SPK1/SPK1-2-A.wav": b"wavdata2",
        "nur/SPK1/readme.txt": b"not a wav",
    })
    _make_zip(label_zip, {
        "medv/간호사/SPK1/SPK1-1-A.json": b'{"a":1}',
        "medv/간호사/SPK1/SPK1-2-A.json": b'{"a":2}',
        "medv/의사/DOC1/DOC1-1-A.json": b'{"a":3}',
    })

    out_dir = tmp_path / "out"
    wav_count, label_count = extract_split(source_zip, label_zip, "medv/간호사/", out_dir)

    assert wav_count == 2
    assert label_count == 2
    assert (out_dir / "wav" / "SPK1-1-A.wav").read_bytes() == b"wavdata1"
    assert (out_dir / "wav" / "SPK1-2-A.wav").read_bytes() == b"wavdata2"
    assert (out_dir / "labels" / "SPK1-1-A.json").exists()
    assert not (out_dir / "labels" / "DOC1-1-A.json").exists()
