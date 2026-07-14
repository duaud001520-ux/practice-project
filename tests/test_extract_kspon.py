import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_kspon import extract_category


def _make_tar(path: Path, inner_name: str, data: bytes) -> None:
    inner_path = path.parent / inner_name
    inner_path.write_bytes(data)
    with tarfile.open(path, "w:gz") as tf:
        tf.add(inner_path, arcname=inner_name)
    inner_path.unlink()


def test_extract_category_unpacks_both_tarballs(tmp_path):
    source_tar = tmp_path / "source.tar.gz"
    label_tar = tmp_path / "label.tar.gz"
    _make_tar(source_tar, "clip1.wav", b"wavdata")
    _make_tar(label_tar, "clip1.txt", b"transcript")

    dest = tmp_path / "out"
    extract_category(source_tar, label_tar, dest)

    assert (dest / "clip1.wav").read_bytes() == b"wavdata"
    assert (dest / "clip1.txt").read_bytes() == b"transcript"
