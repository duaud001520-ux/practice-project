import io
import json
import struct
import sys
import wave
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_callcenter import call_duration_seconds, list_calls, select_calls


def _write_wav_bytes(seconds: float, framerate: int = 16000) -> bytes:
    nframes = int(seconds * framerate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(struct.pack("<%dh" % nframes, *([0] * nframes)))
    return buf.getvalue()


def _make_source_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        # call A: 2 utterances totalling 2s
        zf.writestr("dom/callA/callA001.wav", _write_wav_bytes(1.0))
        zf.writestr("dom/callA/callA002.wav", _write_wav_bytes(1.0))
        # call B: 1 utterance totalling 10s
        zf.writestr("dom/callB/callB001.wav", _write_wav_bytes(10.0))


def test_list_calls_groups_wav_by_folder(tmp_path):
    source_zip = tmp_path / "source.zip"
    _make_source_zip(source_zip)

    calls = list_calls(source_zip)

    assert set(calls.keys()) == {"dom/callA", "dom/callB"}
    assert calls["dom/callA"] == ["dom/callA/callA001.wav", "dom/callA/callA002.wav"]


def test_call_duration_seconds_sums_clip_lengths(tmp_path):
    source_zip = tmp_path / "source.zip"
    _make_source_zip(source_zip)

    calls = list_calls(source_zip)

    assert call_duration_seconds(source_zip, calls["dom/callA"]) == 2.0
    assert call_duration_seconds(source_zip, calls["dom/callB"]) == 10.0


def test_select_calls_filters_by_duration_range(tmp_path):
    source_zip = tmp_path / "source.zip"
    _make_source_zip(source_zip)

    selected = select_calls(source_zip, min_sec=5, max_sec=20, n_calls=10)

    assert selected == ["dom/callB"]
