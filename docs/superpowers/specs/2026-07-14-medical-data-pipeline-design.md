# 의료 STT 데이터 파이프라인 + E0' 측정 (설계)

## 목표
Windows에 있는 AIHub 원본(zip/tar.gz)을 이동하지 않고, 필요한 부분만 선택 추출하여
`~/data`에 구성한 뒤, whisper-small로 의료 데이터 100샘플 WER/CER을 측정하고
`experiments.md`에 E0'로 기록한다.

## 배경 조사 결과
- **medical** (272GB): 전량 완전 다운로드, 분할압축 아님, 단일 zip들.
- **kspon** (279GB): 조사 시점엔 2개 카테고리(경제_04, 놀이_01)가 다운로드 중이었으나
  이후 완료 확인. 현재 전 카테고리 완전.
- **noise**: 전부 0바이트, `.irx` 임시파일에 극히 일부만 존재 — Windows 쪽 다운로드가
  멈춰있는 상태. 이번 범위에서 **보류**, 재다운로드 완료 후 재검토.
- WSL 홈(`/`) 여유 942GB, `/mnt/c` 여유 2.6TB → 전량 이동은 불필요하고 비효율적.
  zip 원본은 `/mnt/c`에 유지하고 필요한 부분만 `~/data`로 압축 해제하는 방식으로 결정.

## medical zip 내부 구조
- 원천 zip: `nur/{화자ID}/{화자ID}-{utt}-...-C.wav`
- 라벨 zip: Training은 `medsub/{간호사|의사|환자}/{화자ID}/*.json,*.txt`,
  Validation은 `medv/{간호사|의사|환자}/{화자ID}/*.json,*.txt`
- JSON 구조: `전사정보.LabelText`(전사문), `음성정보.SamplingRate`(48000, 16bit mono),
  `화자정보`, `파일정보.FileName`(wav 파일명과 매칭)
- `.txt`는 전사문만 담긴 단순 텍스트 (LabelText와 동일 내용)
- 파일명 basename으로 원천 wav ↔ 라벨 json/txt 1:1 매칭 가능

## 데이터 선택 범위 (사용자 확정)
- **medical eval**: `Validation/[V원천]의료진_간호사_1.zip` + `Validation/[V]라벨링데이터.zip`
  중 간호사분 → `~/data/medical/eval/` (약 13GB, 42,284클립). 학습에 사용 금지, eval 전용
- **medical train**: `Training/[T원천]의료진_간호사_1.zip` + `Training/[T]라벨링데이터.zip`
  중 간호사분 → `~/data/medical/train/` (약 22.5GB, 79,905클립, 약 60시간)
- **kspon**: `날씨_weather_03`(1.08GB) + `취미_hobby_01`(10.2GB) → `~/data/kspon/`
  (약 11GB). 일반 발화 성능 퇴화 방지용 혼합. 부족 시 추후 카테고리 추가
- **noise**: 보류

## 컴포넌트
1. **추출 스크립트** (`scripts/extract_medical.py`, `scripts/extract_kspon.py` 또는 일회성 스크립트)
   - zip은 python `zipfile`로 필요한 멤버만 선택 추출 (전체 압축 해제 불필요)
   - medical: 원천 zip의 `nur/`와 라벨 zip의 `medsub|medv/간호사/`만 추출
   - kspon: tar.gz 통째로 압축 해제 (카테고리 단위가 이미 작아서 선택 추출 불필요)
2. **파싱 함수** (`scripts/dataset.py` 등): wav 경로 ↔ json/txt 전사문 매칭해서
   `(wav_path, transcript)` 쌍 리스트 반환. medical train/eval 양쪽에서 재사용
3. **eval.py**: `--model`, `--n_samples` 인자로 faster-whisper 추론 → jiwer로 WER/CER 계산
4. **E0' 실행**: `python eval.py --model small --n_samples 100 --data medical/eval` →
   결과를 `experiments.md`에 기록 (실험ID/변경점/학습데이터/WER/CER/소요시간/커밋해시)

## 검증
- eval 샘플이 train 쪽 화자ID와 겹치지 않는지 확인 (Validation/Training 공식 분리라 자연히 보장되지만, 스크립트에서도 assert)
- 추출 후 wav 개수와 라벨 개수가 일치하는지 확인
- eval.py는 `~/data/medical/eval`에서만 읽고 train 경로 접근 금지
