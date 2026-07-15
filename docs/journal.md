# 작업 일지

## 2026-07-14 (4차)
- Claude Code로 평가 파이프라인 완성 (extract/dataset/metrics/eval + pytest 9개)
- E0' 확보: whisper-small 의료(간호사) 100샘플 — 원본 WER 26.69%/CER 7.68%, 정규화 23.50%/7.21%
- 오류 분석 결과 음소 오인식 위주(표기 차이 아님) → E2 파인튜닝·E4 후교정 타겟 명확
- 방침: 원본/정규화 지표 동시 기록, 의사결정 비교는 500~1000샘플
- 데이터 추출 완료 (eval 42,177 / train 79,905 / kspon 15GB)
- UI 아이디어(P2): SBAR 전문용어 클릭 시 설명 툴팁
- 다음: E1(large-v3) 진행 예정

## 2026-07-15 (1차)
- E1(large-v3, 의료 eval 100샘플) 측정: 원본은 small보다 나쁘지만(WER 36.09%/CER 14.76%) 정규화 후엔 근접/약간 우위
- E1 CER 최악 샘플 분석 → large-v3가 짧은 발화 뒤에 "자막 제공", "OOO 배달의민족" 같은 유튜브 자막 크레딧류 환각을 뚜렷하게 생성함을 확인 (analyze_errors.py에 --metric, 반복/환각 탐지 추가)
- 새 데이터셋 조사: AIHub 186 "복지 분야 콜센터 상담데이터"(210GB, 다운로드 완전) — 대회 데이터 유력 후보로 CLAUDE.md에 기록. 통화 단위로 발화 클립이 나뉘어 있고 파일명순 이어붙이면 실제 긴 통화(중앙값 6분) 재구성 가능
- E1b: 콜센터 긴 통화 25건(60~180s)에서 small vs large-v3 비교 → 짧은 발화와 순위가 뒤집힘(정규화 CER: large-v3 9.97% vs small 14.58%, large-v3 우위). 짧은 클립만으로 모델을 확정하면 안 된다는 근거 확보
- **중요 업데이트**: 강의자료에서 ERR 정의 확인 — `ERR(%) = (Baseline−New)/Baseline×100`, 절대 오류율이 아니라 baseline 대비 상대 개선율. 한국어는 CER이 주 지표. "ERR 10% 이상 = 20점". 수업 파인튜닝은 LoRA 아닌 full FT(Seq2SeqTrainer), 노트북 03→05→07→08 순서
- E2 전략 확정: 리허설은 small full FT(수업 호환), 본선은 large-v3 FT + 환각 억제 디코딩 후보. 최종 모델 선택은 절대 CER이 아니라 baseline 대비 상대 개선 폭 기준
- 다음: E2(파인튜닝) 착수, 콜센터 train 데이터 파인튜닝용 추출 여부 결정

## 2026-07-15 (2차)
- 과제 노트북 3개(03-1-aihub186 텍스트/음성정보/필터링) 로직을 로컬 재현: 210GB가 아직 zip 상태(압축 해제 전)라 zip 중앙 디렉터리 메타데이터(`file_size`)만으로 wav duration을 계산해 전체 압축 해제 없이 처리(16kHz/16bit/mono/44바이트 헤더 가정, 표본 검증으로 확인). `scripts/aihub186_{data,orgtext,wav_info,category_hours,quick_eval}.py` + pytest 26개 추가
- orgtext.tsv(2,048,986행) / wav_info.tsv(2,049,025개, 총 2,650.5시간) 생성 완료, category1>2>3 계층 집계(train/validation 구분)로 "학습100h+/평가10h+" 후보 5개 도출 → AI콜 시나리오 적합성·윤리 이유로 정신건강상담(우울증) 제외하고 차량요청/외래/적용기준 등 3개 추천
- **코드 리뷰(8각도) 실행**: 확정 8건(undersized wav 파일 음수 duration 가드 누락, BadZipFile/OSError 미포착, 표본 검증이 zip 전체가 아닌 앞부분만 확인, 헤더 바이트 길이 미검증, 빈 결과를 통과로 오판, 중복 상수 선언, inner join 미매칭 카운트 누락, 불필요한 로컬 import) 전부 수정 → 테스트 36개 통과 → 커밋(057bdc4, 이미 push됨) 확인
- **카테고리 방향 확정**: 팀 잠정안은 1순위(차량요청) 대신 **`대학병원>진료안내>외래`**. 학습량 차이 7%로 거의 동급 + 기존 의료(간호사) 데이터로 증강 가능 + 최종 시나리오를 "AI 인수인계 콜"(SBAR 자동완성)로 잡으면 병원 도메인이 일관됨
- E0-186a(차량요청)/E0-186b(외래) whisper-small 무튜닝 100샘플 비교: CER 거의 동일(원본 11.94%/11.87%, 정규화 9.87%/10.26%) → 외래로 바꿔도 baseline 성능 손해 없음 확인, 외래로 확정
- 노트북4(prepare_dataset-whisper_tiny) 입수: asr_dataset.tsv→0.1~30초 필터→중복제거→train/dev/validation 3분할→HF Dataset 저장이 공식 파이프라인(모델은 MODEL_ID로 tiny/small/large 전환)
- **E2 방침 변경**: small 리허설 스킵, 확정 카테고리(외래) 186 데이터로 직행 — 노트북3 필터링과 학습 데이터가 같아 일석이조
- 다음: 노트북4 로직 로컬 재현(외래 카테고리 asr_dataset.tsv → HF Dataset) → Seq2SeqTrainer로 whisper-small(또는 large-v3) 파인튜닝 실행
