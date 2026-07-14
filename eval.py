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
