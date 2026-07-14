import argparse
import io
import json
import wave
import zipfile
from pathlib import Path

DOWNLOADS = Path(
    "/mnt/c/Users/visionlab/Downloads/186.복지 분야 콜센터 상담데이터/01.데이터"
)
SOURCE_ZIP = DOWNLOADS / "2.Validation" / "원천데이터" / "VS1_01.대학병원.zip"
LABEL_ZIP = DOWNLOADS / "2.Validation" / "라벨링데이터" / "VL1_01.대학병원.zip"

BYTES_PER_SEC = 32000  # 16kHz, 16-bit, mono PCM
WAV_HEADER_BYTES = 44


def list_calls(source_zip: Path) -> dict[str, list[str]]:
    """Map call folder -> sorted wav member names within it (filename order = time order)."""
    calls: dict[str, list[str]] = {}
    with zipfile.ZipFile(source_zip) as zf:
        for info in zf.infolist():
            if info.filename.endswith(".wav"):
                folder = "/".join(info.filename.split("/")[:-1])
                calls.setdefault(folder, []).append(info.filename)
    for names in calls.values():
        names.sort()
    return calls


def call_duration_seconds(source_zip: Path, wav_names: list[str]) -> float:
    with zipfile.ZipFile(source_zip) as zf:
        total_bytes = sum(zf.getinfo(n).file_size - WAV_HEADER_BYTES for n in wav_names)
    return total_bytes / BYTES_PER_SEC


def select_calls(
    source_zip: Path, min_sec: float, max_sec: float, n_calls: int
) -> list[str]:
    calls = list_calls(source_zip)
    in_range = [
        (folder, call_duration_seconds(source_zip, names))
        for folder, names in calls.items()
    ]
    in_range = [(folder, d) for folder, d in in_range if min_sec <= d <= max_sec]
    in_range.sort(key=lambda item: item[1])
    return [folder for folder, _ in in_range[:n_calls]]


def extract_call(
    source_zip: Path, label_zip: Path, call_folder: str, out_dir: Path
) -> None:
    wav_dir = out_dir / "wav"
    label_dir = out_dir / "labels"
    wav_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    call_id = call_folder.split("/")[-1]

    with zipfile.ZipFile(source_zip) as zf:
        wav_names = sorted(
            n for n in zf.namelist()
            if n.startswith(call_folder + "/") and n.endswith(".wav")
        )
        frames = []
        params = None
        for name in wav_names:
            with wave.open(io.BytesIO(zf.read(name)), "rb") as wf:
                params = wf.getparams()
                frames.append(wf.readframes(wf.getnframes()))

    out_wav = wav_dir / f"{call_id}.wav"
    with wave.open(str(out_wav), "wb") as out:
        out.setparams(params)
        for chunk in frames:
            out.writeframes(chunk)

    utterances = []
    with zipfile.ZipFile(label_zip) as zf:
        for name in wav_names:
            json_name = f"{call_folder}/{Path(name).stem}.json"
            data = json.loads(zf.read(json_name))
            utterances.append(data["inputText"][0]["orgtext"])

    label_path = label_dir / f"{call_id}.json"
    label_path.write_text(
        json.dumps({"utterances": utterances}, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default=str(Path.home() / "data" / "callcenter" / "eval")
    )
    parser.add_argument("--min_sec", type=float, default=60)
    parser.add_argument("--max_sec", type=float, default=180)
    parser.add_argument("--n_calls", type=int, default=25)
    args = parser.parse_args()

    out_dir = Path(args.out)
    selected = select_calls(SOURCE_ZIP, args.min_sec, args.max_sec, args.n_calls)
    print(f"선택된 통화 수: {len(selected)}")
    for call_folder in selected:
        extract_call(SOURCE_ZIP, LABEL_ZIP, call_folder, out_dir)
        print(f"  {call_folder.split('/')[-1]} 추출 완료")


if __name__ == "__main__":
    main()
