# 프로젝트: SCI 부트캠프 — 핸즈프리 인수인계 (팀 Ai-CO)

## 목표
- 서울과기대 SCI 부트캠프 경진대회 1등 (7/31)
- Whisper 기반 STT 성능(ERR) 개선이 본체 (평가 배점: 성능 60 / 아이디어 20 / 발표 20)
- **ERR(%) = (Baseline − New) / Baseline × 100** — 절대 오류율이 아니라 baseline 대비 상대 개선율(강의자료 확인).
  한국어는 CER을 주 지표로 사용. "ERR 10% 이상 = 20점" → baseline CER 대비 10% 상대 개선이 만점 기준
- 서비스: 현장 녹음 업로드 → STT → LLM 후교정 → SBAR 인수인계 양식 자동 작성 → TTS 브리핑
- 대회 힌트: STT+LLM+TTS 셋 다 쓰면 가산점, 결과는 한 화면 데모, UI는 FastAPI

## 환경
- WSL2 Ubuntu, conda env 'practice' (Python 3.10)
- RTX 3090 x2 (각 24GB), CUDA 12.1, PyTorch GPU 확인 완료
- 부트캠프 A100 서버 별도 (JupyterLab 10.10.15.111:8013, 교내망 전용)
- alias: act(conda activate practice), gpu(nvidia-smi watch), gs/ga/gc/gp(git)

## 데이터
- 원본 zip/tar.gz는 Windows 다운로드 폴더(/mnt/c/Users/visionlab/Downloads/)에 그대로 둔다 (전부 이동하기엔 용량 과다, 삭제·이동 금지). 필요한 부분만 골라 ~/data로 압축 해제해서 사용
- ~/data/medical : AIHub 208 "비대면 진료를 위한 의료진 및 환자 음성" 중 간호사 발화만 선택 추출 — 파인튜닝 핵심
  - eval/ ← Validation/[V원천]의료진_간호사_1.zip + 라벨(간호사분) (~13GB, 42,177클립). eval 전용, 학습에 사용 금지
  - train/ ← Training/[T원천]의료진_간호사_1.zip + 라벨(간호사분) (~22.5GB, wav 79,905클립, 약 60시간).
    단, Training 라벨 zip은 간호사_1~4 화자 라벨을 통째로 담고 있어 label 폴더에 wav 없는 라벨(간호사_2~4분)까지
    같이 풀림(json 329,416개) — load_medical_pairs가 매칭되는 것만 쓰므로 기능엔 문제 없지만 디스크는 낭비 중, 정리는 보류
  - 의사/환자, 간호사_2~4 zip은 현재 미사용 (분량 부족 판단 시 추가)
- ~/data/kspon : 한국인 대화음성(KsponSpeech) 중 날씨_weather_03(1.08GB) + 취미_hobby_01(10.2GB)만 추출 (~11GB) — 파인튜닝 시 일반 발화 성능 퇴화 방지용 혼합. 부족하면 카테고리 추가
- ~/data/noise : 보류. Windows 쪽 다운로드가 중단된 상태(전부 0바이트, .irx 임시파일만 존재) — 재다운로드 완료 후 재검토
- ~/data/callcenter : AIHub 186 "복지 분야 콜센터 상담데이터" — **대회 데이터 유력 후보, 중요**. 다운로드 완전(210GB).
  3개 도메인(대학병원/광역이동지원센터/정신건강복지센터) × Training/Validation, 원천 zip과 라벨 zip 분리.
  통화(call) 폴더 하나 = 발화 단위로 쪼개진 wav+json 여러 개(파일명 정렬 = 시간순).
  wav는 16kHz/16bit/mono PCM. json은 `inputText[0].orgtext`(전사문), `info[0].metadata`(화자타입/나이/성별/녹음환경 등).
  통화 길이 분포(대학병원 Validation 기준): 최소 77s, 중앙값 365s, 최대 2138s — 실제로 이미 김. 발화 파일들을 파일명순으로 이어붙이면 긴 통화 오디오+전사문 재구성 가능
  - eval/ ← Validation/대학병원 zip에서 60~180초 통화 25건만 선택 추출 (전체 원천 7.5GB 중 일부)
  - Training(각 도메인 약 65~70GB)은 아직 미사용, 파인튜닝 단계에서 필요시 추가

## 기술 스택
- STT 추론: faster-whisper
- 파인튜닝: 수업 자료는 full fine-tuning(transformers Seq2SeqTrainer, LoRA 아님) 기준, 노트북 03→05→07→08 순서로 진행 예정.
  리허설(small)은 이 방식을 그대로 따르고, 본선(large-v3)은 메모리·시간 제약에 따라 full FT/PEFT(LoRA) 중 택1 — E2에서 결정
- 평가: jiwer (WER/CER, 대회 ERR은 CER 기준)
- LLM(후교정·SBAR 구조화): 한국어 오픈소스 8B급 (Qwen/EXAONE 계열, 비교 후 확정)
- TTS: 한국어 오픈소스 비교 후 확정 / 데모 UI: FastAPI 원페이지

## 실험 규칙 (E0~E6 플레이북)
- E0: baseline(whisper-small) → E1: large-v3 무튜닝 → E2: LoRA 파인튜닝
  → E3: +증강 → E4: +LLM 후교정 → E5: +디코딩 튜닝 → E6: 앙상블(여유 시)
- 기준점(E0, Zeroth 457샘플): whisper-small CER 12.25% / WER 37.5% — ERR 계산의 baseline 후보(대회 공식 baseline 확정 전까지는 CER 12.25%를 기준으로 상대 개선율 가늠)
- E1b(사이드 프로브, 메인 시퀀스 아님): 콜센터 긴 통화(eval.py --dataset callcenter)로 small vs large-v3의
  순위가 짧은 발화(E0'/E1)와 다르게 나오는지 확인. 짧은 발화 결과만으로 모델을 확정하지 않기 위함
- 모든 실험은 eval.py로 측정하고 experiments.md에 기록
  (형식: 실험ID / 변경점 / 학습데이터 / 샘플수 / WER(원본) / CER(원본) / WER(정규화) / CER(정규화) / 소요시간 / 커밋해시)
  대회 ERR 산정 시 정규화(구두점 제거·띄어쓰기 통일) 여부가 아직 불확실해서 eval.py가 원본/정규화 값을 항상 같이 출력·기록함
- 중간 점검은 100샘플로 빠르게, **E2 파인튜닝 전/후 등 의사결정용 비교는 500~1000샘플로 측정**
  (짧은 문장 오류 하나가 WER을 크게 흔들어 100샘플은 노이즈가 큼. whisper-small 100샘플 30초 기준 500~1000샘플도 수 분이면 됨)
- 평가셋은 절대 학습에 사용 금지 (leakage 금지, 증강도 평가셋엔 미적용)
- 큰 파일 삭제·이동 전에 반드시 사용자 확인받기

## 플러그인 워크플로우
1. 덩치 있는 작업(파인튜닝 스크립트, 파이프라인 통합 등)은 바로 코드 작성하지 말고
   superpowers로 계획부터 수립 후 진행 (brainstorm → plan → 구현)
2. 스크립트 완성 후 커밋 전에 반드시 /code-review 실행
   — 특히 학습/평가 데이터 누수(leakage), 평가셋 오염 여부 중점 확인
3. claude-mem은 세션 간 기억 보조용. 실험 결과의 공식 기록은 experiments.md
   (claude-mem 기억에 의존하지 말 것)
4. frontend-design은 4주차 데모 UI(FastAPI 원페이지) 작업 때만 사용
5. security-guidance는 이 프로젝트에서 사용 안 함

## 소통 규칙
- 한국어로 답할 것. 코드·명령어·에러 로그는 원문 그대로
- 새 개념은 1~2문장 설명 후 바로 실행으로 연결
- 작업 완료 시 "정리된 것 / 다음 할 일" 3줄 요약
