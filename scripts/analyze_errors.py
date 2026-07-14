import argparse
import random
from pathlib import Path

import jiwer

from dataset import load_medical_pairs
from metrics import compute_metrics, detect_repetition, normalize_text


def transcribe_all(pairs, model) -> list[tuple[Path, str, str]]:
    results = []
    for wav_path, transcript in pairs:
        segments, _ = model.transcribe(str(wav_path), language="ko")
        hyp = "".join(seg.text for seg in segments).strip()
        results.append((wav_path, transcript, hyp))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument(
        "--data", default=str(Path.home() / "data" / "medical" / "eval")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--metric", choices=["wer", "cer"], default="wer")
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    pairs = load_medical_pairs(args.data)
    random.Random(args.seed).shuffle(pairs)
    subset = pairs[: args.n_samples]

    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    results = transcribe_all(subset, model)

    score_fn = jiwer.cer if args.metric == "cer" else jiwer.wer
    scored = []
    for wav_path, ref, hyp in results:
        sample_score = score_fn(ref, hyp) if ref.strip() else 0.0
        scored.append((sample_score, wav_path, ref, hyp))
    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"=== {args.metric.upper()} 최악 {args.top_k}개 샘플 (n={len(subset)}) ===")
    for sample_score, wav_path, ref, hyp in scored[: args.top_k]:
        print(f"\n[{wav_path.name}] {args.metric.upper()}={sample_score * 100:.1f}%")
        print(f"  정답: {ref}")
        print(f"  예측: {hyp}")
        rep = detect_repetition(hyp)
        if rep is not None:
            print(f"  ⚠ 반복 의심: \"{rep}\"")
        if len(ref.strip()) > 0 and len(hyp) > 2 * len(ref):
            print(f"  ⚠ 예측 길이가 정답의 2배 초과 (환각 의심): 정답 {len(ref)}자 vs 예측 {len(hyp)}자")

    refs = [r for _, r, _ in results]
    hyps = [h for _, _, h in results]
    raw = compute_metrics(refs, hyps)
    norm_refs = [normalize_text(r) for r in refs]
    norm_hyps = [normalize_text(h) for h in hyps]
    normalized = compute_metrics(norm_refs, norm_hyps)

    print("\n=== 정규화 전/후 비교 (구두점 제거 + 띄어쓰기 통일) ===")
    print(f"{'':10s}{'WER':>10s}{'CER':>10s}")
    print(f"{'정규화 전':10s}{raw['wer'] * 100:9.2f}%{raw['cer'] * 100:9.2f}%")
    print(f"{'정규화 후':10s}{normalized['wer'] * 100:9.2f}%{normalized['cer'] * 100:9.2f}%")


if __name__ == "__main__":
    main()
