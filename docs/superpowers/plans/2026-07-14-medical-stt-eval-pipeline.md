# 의료 STT 데이터 추출 + E0' 평가 파이프라인 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Windows 원본(zip/tar.gz)을 그대로 두고 필요한 의료(간호사)/kspon 서브셋만 `~/data`로 선택 추출한 뒤, whisper-small로 의료 100샘플 WER/CER을 측정해 `experiments.md`에 E0'로 기록한다.

**Architecture:** 추출은 `scripts/extract_medical.py`(zipfile 선택 추출), `scripts/extract_kspon.py`(tarfile 전체 추출, 카테고리 단위가 이미 작음) 두 개의 독립 CLI 스크립트로 분리. `dataset.py`는 wav↔전사문 매칭만 담당, `metrics.py`는 WER/CER 계산만 담당, `eval.py`가 이 둘과 faster-whisper를 조합하는 얇은 CLI. 각 모듈은 순수 함수 형태로 pytest 픽스처(작은 합성 zip/tar)로 검증하고, 실제 대용량 데이터에 대한 실행은 별도 단계로 분리한다.

**Tech Stack:** Python 3.10 (conda env `practice`), faster-whisper, jiwer, pytest, 표준 라이브러리 zipfile/tarfile

## Global Constraints

- 원본 zip/tar.gz는 `/mnt/c/Users/visionlab/Downloads/` 원위치 유지 — 이동·삭제 금지
- eval 서브셋(`~/data/medical/eval`)은 학습에 절대 사용하지 않는다 (leakage 금지)
- conda env `practice` (Python 3.10)를 사용해 모든 python/pip 명령 실행
- GPU: CUDA 12.1, RTX 3090 — faster-whisper는 `device="cuda"`로 실행

---

### Task 0: 의존성 설치

**Files:** 없음 (conda env `practice`에 패키지 설치)

**Interfaces:**
- Produces: `faster_whisper`, `jiwer`, `pytest` 가 `practice` env에서 import 가능한 상태

- [ ] **Step 1: 설치**

Run:
```bash
conda activate practice
pip install faster-whisper jiwer pytest
```

- [ ] **Step 2: 설치 확인**

Run:
```bash
python -c "import faster_whisper, jiwer, pytest; print('ok')"
```
Expected: `ok`

---

### Task 1: 의료 데이터 선택 추출 스크립트

**Files:**
- Create: `scripts/extract_medical.py`
- Test: `tests/test_extract_medical.py`

**Interfaces:**
- Produces: `extract_split(source_zip: Path, label_zip: Path, label_prefix: str, out_dir: Path) -> tuple[int, int]` (wav_count, label_count 반환). `out_dir` 아래 `wav/`, `labels/` 서브디렉터리를 만들고 채운다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_extract_medical.py`:
```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_extract_medical.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'extract_medical'`

- [ ] **Step 3: 최소 구현 작성**

`scripts/extract_medical.py`:
```python
import argparse
from pathlib import Path
import zipfile

DOWNLOADS = Path(
    "/mnt/c/Users/visionlab/Downloads/비대면 진료를 위한 의료진 및 환자 음성"
)

SPLITS = {
    "eval": {
        "source_zip": DOWNLOADS / "Validation" / "[V원천]의료진_간호사_1.zip",
        "label_zip": DOWNLOADS / "Validation" / "[V]라벨링데이터.zip",
        "label_prefix": "medv/간호사/",
    },
    "train": {
        "source_zip": DOWNLOADS / "Training" / "[T원천]의료진_간호사_1.zip",
        "label_zip": DOWNLOADS / "Training" / "[T]라벨링데이터.zip",
        "label_prefix": "medsub/간호사/",
    },
}


def extract_split(
    source_zip: Path, label_zip: Path, label_prefix: str, out_dir: Path
) -> tuple[int, int]:
    wav_dir = out_dir / "wav"
    label_dir = out_dir / "labels"
    wav_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    wav_count = 0
    with zipfile.ZipFile(source_zip) as zf:
        for info in zf.infolist():
            if info.filename.endswith(".wav"):
                target = wav_dir / Path(info.filename).name
                with zf.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                wav_count += 1

    label_count = 0
    with zipfile.ZipFile(label_zip) as zf:
        for info in zf.infolist():
            if info.filename.startswith(label_prefix) and info.filename.endswith(".json"):
                target = label_dir / Path(info.filename).name
                with zf.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                label_count += 1

    return wav_count, label_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["eval", "train", "both"], default="both")
    parser.add_argument("--out", default=str(Path.home() / "data" / "medical"))
    args = parser.parse_args()

    out_root = Path(args.out)
    splits = ["eval", "train"] if args.split == "both" else [args.split]
    for split in splits:
        cfg = SPLITS[split]
        wav_count, label_count = extract_split(
            cfg["source_zip"], cfg["label_zip"], cfg["label_prefix"], out_root / split
        )
        print(f"[{split}] wav={wav_count} label={label_count}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_extract_medical.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: 실제 데이터 추출 실행 (시간이 걸릴 수 있음, /mnt/c I/O)**

Run:
```bash
conda activate practice
python scripts/extract_medical.py --split both
```
Expected 출력 (개수는 정확히 일치해야 함):
```
[eval] wav=42284 label=42284
[train] wav=79905 label=79905
```
wav_count와 label_count가 다르면 멈추고 원인 파악 (라벨 누락 파일 존재 가능).

- [ ] **Step 6: 커밋**

```bash
git add scripts/extract_medical.py tests/test_extract_medical.py
git commit -m "add medical 간호사 subset extraction script"
```

---

### Task 2: kspon 카테고리 추출 스크립트

**Files:**
- Create: `scripts/extract_kspon.py`
- Test: `tests/test_extract_kspon.py`

**Interfaces:**
- Produces: `extract_category(source_tar: Path, label_tar: Path, dest: Path) -> None` — `dest` 아래에 두 tar.gz 내용을 모두 풀어놓는다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_extract_kspon.py`:
```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_extract_kspon.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'extract_kspon'`

- [ ] **Step 3: 최소 구현 작성**

`scripts/extract_kspon.py`:
```python
import argparse
from pathlib import Path
import tarfile

DOWNLOADS = Path("/mnt/c/Users/visionlab/Downloads/한국인 대화 음성/Training")

CATEGORIES = {
    "weather_03": {
        "source": DOWNLOADS / "[원천]5.날씨_weather_03.tar.gz",
        "label": DOWNLOADS / "[라벨]5.날씨_weather_03.tar.gz",
    },
    "hobby_01": {
        "source": DOWNLOADS / "[원천]2.취미_hobby_01.tar.gz",
        "label": DOWNLOADS / "[라벨]2.취미_hobby_01.tar.gz",
    },
}


def extract_category(source_tar: Path, label_tar: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for tar_path in (source_tar, label_tar):
        with tarfile.open(tar_path) as tf:
            tf.extractall(dest, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path.home() / "data" / "kspon"))
    args = parser.parse_args()

    out_root = Path(args.out)
    for name, cfg in CATEGORIES.items():
        extract_category(cfg["source"], cfg["label"], out_root / name)
        print(f"[{name}] extracted to {out_root / name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_extract_kspon.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: 실제 데이터 추출 실행**

Run:
```bash
conda activate practice
python scripts/extract_kspon.py
```
Expected 출력:
```
[weather_03] extracted to /home/minseo/data/kspon/weather_03
[hobby_01] extracted to /home/minseo/data/kspon/hobby_01
```

- [ ] **Step 6: 커밋**

```bash
git add scripts/extract_kspon.py tests/test_extract_kspon.py
git commit -m "add kspon weather_03/hobby_01 extraction script"
```

---

### Task 3: 의료 데이터 wav-전사 파싱 함수

**Files:**
- Create: `dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: Task 1이 만든 `~/data/medical/{eval,train}/wav/*.wav`, `.../labels/*.json` 레이아웃
- Produces: `load_medical_pairs(data_dir: Path) -> list[tuple[Path, str]]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_dataset.py`:
```python
import json
from pathlib import Path

from dataset import load_medical_pairs


def test_load_medical_pairs_matches_wav_to_transcript(tmp_path):
    wav_dir = tmp_path / "wav"
    label_dir = tmp_path / "labels"
    wav_dir.mkdir()
    label_dir.mkdir()

    (wav_dir / "SPK1-1-A.wav").write_bytes(b"dummy")
    (label_dir / "SPK1-1-A.json").write_text(
        json.dumps({"전사정보": {"LabelText": "안녕하세요"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    # 라벨 없는 wav는 건너뛴다
    (wav_dir / "SPK1-2-A.wav").write_bytes(b"dummy2")

    pairs = load_medical_pairs(tmp_path)

    assert len(pairs) == 1
    wav_path, transcript = pairs[0]
    assert wav_path.name == "SPK1-1-A.wav"
    assert transcript == "안녕하세요"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dataset'`

- [ ] **Step 3: 최소 구현 작성**

`dataset.py`:
```python
import json
from pathlib import Path


def load_medical_pairs(data_dir: Path) -> list[tuple[Path, str]]:
    data_dir = Path(data_dir)
    wav_dir = data_dir / "wav"
    label_dir = data_dir / "labels"

    pairs: list[tuple[Path, str]] = []
    for wav_path in sorted(wav_dir.glob("*.wav")):
        json_path = label_dir / f"{wav_path.stem}.json"
        if not json_path.exists():
            continue
        with open(json_path, encoding="utf-8") as f:
            label = json.load(f)
        transcript = label["전사정보"]["LabelText"]
        pairs.append((wav_path, transcript))
    return pairs
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: 커밋**

```bash
git add dataset.py tests/test_dataset.py
git commit -m "add medical wav-transcript pairing function"
```

---

### Task 4: WER/CER 계산 함수

**Files:**
- Create: `metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `compute_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]` (`{"wer": ..., "cer": ...}`)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_metrics.py`:
```python
from metrics import compute_metrics


def test_compute_metrics_perfect_match_is_zero():
    result = compute_metrics(["안녕 하세요"], ["안녕 하세요"])
    assert result["wer"] == 0.0
    assert result["cer"] == 0.0


def test_compute_metrics_detects_errors():
    result = compute_metrics(["안녕 하세요"], ["안녕 하십니까"])
    assert result["wer"] > 0.0
    assert result["cer"] > 0.0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'metrics'`

- [ ] **Step 3: 최소 구현 작성**

`metrics.py`:
```python
import jiwer


def compute_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    return {
        "wer": jiwer.wer(references, hypotheses),
        "cer": jiwer.cer(references, hypotheses),
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add metrics.py tests/test_metrics.py
git commit -m "add WER/CER computation via jiwer"
```

---

### Task 5: eval.py CLI + E0' 측정

**Files:**
- Create: `eval.py`
- Test: `tests/test_eval.py`
- Modify: `experiments.md` (신규 생성)

**Interfaces:**
- Consumes: `dataset.load_medical_pairs`, `metrics.compute_metrics`
- Produces: `run_eval(pairs: list[tuple[Path, str]], model) -> dict[str, float]` (model은 `.transcribe(path: str, language: str) -> tuple[segments, info]`를 갖는 객체 — faster-whisper `WhisperModel`과 동일 인터페이스)

- [ ] **Step 1: 실패하는 테스트 작성 (가짜 모델로 GPU 없이 검증)**

`tests/test_eval.py`:
```python
from pathlib import Path

from eval import run_eval


class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModel:
    def __init__(self, hyp_by_stem: dict[str, str]) -> None:
        self.hyp_by_stem = hyp_by_stem

    def transcribe(self, path: str, language: str):
        stem = Path(path).stem
        return [_FakeSegment(self.hyp_by_stem[stem])], None


def test_run_eval_computes_metrics_from_pairs(tmp_path):
    wav1 = tmp_path / "a.wav"
    wav2 = tmp_path / "b.wav"
    wav1.write_bytes(b"x")
    wav2.write_bytes(b"x")
    pairs = [(wav1, "안녕 하세요"), (wav2, "반갑 습니다")]

    model = _FakeModel({"a": "안녕 하세요", "b": "반갑 습니다"})
    result = run_eval(pairs, model)

    assert result["wer"] == 0.0
    assert result["cer"] == 0.0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval'`

- [ ] **Step 3: 최소 구현 작성**

`eval.py`:
```python
import argparse
import random
import time
from pathlib import Path

from dataset import load_medical_pairs
from metrics import compute_metrics


def run_eval(pairs: list[tuple[Path, str]], model) -> dict[str, float]:
    references: list[str] = []
    hypotheses: list[str] = []
    for wav_path, transcript in pairs:
        segments, _ = model.transcribe(str(wav_path), language="ko")
        hyp = "".join(seg.text for seg in segments).strip()
        references.append(transcript)
        hypotheses.append(hyp)
    return compute_metrics(references, hypotheses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument(
        "--data", default=str(Path.home() / "data" / "medical" / "eval")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    pairs = load_medical_pairs(args.data)
    random.Random(args.seed).shuffle(pairs)
    subset = pairs[: args.n_samples]

    model = WhisperModel(args.model, device="cuda", compute_type="float16")

    start = time.time()
    metrics = run_eval(subset, model)
    elapsed = time.time() - start

    print(
        f"model={args.model} n={len(subset)} "
        f"WER={metrics['wer'] * 100:.2f}% CER={metrics['cer'] * 100:.2f}% "
        f"elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_eval.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: experiments.md 생성**

`experiments.md`:
```markdown
# 실험 기록

| 실험ID | 변경점 | 학습데이터 | WER | CER | 소요시간 | 커밋해시 |
|---|---|---|---|---|---|---|
```

- [ ] **Step 6: E0' 실제 측정 실행**

Run:
```bash
conda activate practice
python eval.py --model small --n_samples 100 --data ~/data/medical/eval
```
Expected 출력 형식 (수치는 실측값으로 채움):
```
model=small n=100 WER=XX.XX% CER=XX.XX% elapsed=XX.Xs
```
(첫 실행 시 whisper-small 가중치를 HuggingFace에서 자동 다운로드 — 인터넷 필요)

- [ ] **Step 7: experiments.md에 E0' 기록**

Step 6 출력값과 `git rev-parse --short HEAD` 결과를 이용해 `experiments.md`에 행 추가:
```markdown
| E0' | whisper-small, 의료(간호사) eval 100샘플 무튜닝 | 없음(baseline) | XX.XX% | XX.XX% | XX.Xs | <커밋해시> |
```

- [ ] **Step 8: 커밋**

```bash
git add eval.py tests/test_eval.py experiments.md
git commit -m "add eval CLI and record E0' medical baseline"
```
