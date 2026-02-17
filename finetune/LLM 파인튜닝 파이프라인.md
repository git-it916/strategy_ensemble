# KOSPI 퀀트 트레이딩 LLM 파인튜닝 파이프라인

Qwen2.5-7B-Instruct를 한국 주식시장 퀀트 트레이딩 전용 LLM으로 만드는 3-Stage 파인튜닝 파이프라인.

---

## 전체 파이프라인 흐름도

```
[Stage 1] 범용 금융 지식 학습 (SFT)
  HuggingFace 5개 데이터셋 (200K)
  ↓ QLoRA (4-bit NF4, rank=32)
  ↓ 2 epochs, lr=2e-4
  output_unsloth/final/ (LoRA adapter)

[Stage 2] 전략 특화 학습 (SFT)
  Synthetic data (5,000건, 12개 태스크)
  + 실제 시장 스냅샷 (2020-2025 KRX 일봉)
  ↓ Stage 1 LoRA 위에 이어서 학습 (또는 --skip-stage1)
  ↓ 3~5 epochs, lr=5e-5
  output_stage2/final/ (LoRA adapter)

[Stage 3] DPO 정렬 학습 (Preference Optimization)
  DPO preference pairs (2,000건)
  chosen: Stage 2 정답 응답
  rejected: 결함 응답 (스키마위반/전략오류/품질저하/환각)
  ↓ Stage 2 LoRA 위에 DPO 학습
  ↓ 1 epoch, lr=5e-7, beta=0.1
  output_stage3/final/ (LoRA adapter)
  ↓ GGUF Q4_K_M 변환
  output_stage3/gguf/ (4.5GB)
  ↓ Ollama 등록
  ollama run qwen2.5-kospi-ft-s3
```

---

## 디렉토리 구조

```
finetune/
├── train_unsloth.md              # 이 문서
├── prepare_dataset.py            # 실제 reasoning log → SFT 변환 (선택)
├── evaluate.py                   # Base vs Fine-tuned 비교 평가
│
├── config/
│   ├── prompts.py                # QUANT_SYSTEM_PROMPT (전략 시스템 프롬프트)
│   ├── datasets.py               # HuggingFace 데이터셋 5개 설정
│   └── universe.py               # 한국 주식 유니버스 97종목
│
├── data/
│   ├── loaders.py                # HuggingFace 데이터 로더 (Stage 1용)
│   ├── synthetic.py              # Synthetic 데이터 생성기 (Stage 2용)
│   ├── gen_dpo_pairs.py          # DPO preference pair 생성기 (Stage 3용)
│   ├── market_snapshot.py        # 실제 시장 데이터 스냅샷 로더
│   ├── synthetic_data.json       # 생성된 SFT 학습 데이터 (git-ignored)
│   └── dpo_pairs.json            # 생성된 DPO 학습 데이터 (git-ignored)
│
├── training/
│   ├── stage1.py                 # Stage 1 학습 스크립트
│   ├── stage2.py                 # Stage 2 학습 스크립트
│   └── stage3_dpo.py             # Stage 3 DPO 학습 스크립트
│
├── output_unsloth/               # Stage 1 산출물 (git-ignored)
│   ├── checkpoint-*/             # 학습 체크포인트
│   ├── final/                    # LoRA adapter
│   └── gguf/                     # GGUF 모델 + Modelfile
│
├── output_stage2/                # Stage 2 산출물 (git-ignored)
│   ├── final/                    # LoRA adapter
│   └── gguf/                     # GGUF 모델 + Modelfile
│
└── output_stage3/                # Stage 3 산출물 (git-ignored)
    ├── final/                    # LoRA adapter
    └── gguf/                     # GGUF 모델 + Modelfile
```

---

## 하드웨어 요구사항

| 항목 | 최소 | 현재 환경 |
|------|------|-----------|
| GPU VRAM | 16GB | RTX 3090 Ti 24GB |
| RAM | 32GB | 64GB DDR5-5600 |
| 디스크 | 50GB | Samsung 990 PRO 1TB |
| CUDA | 11.8+ | 12.x |

---

## 환경 준비

```bash
conda create -n unsloth_env python=3.10 -y
conda activate unsloth_env
pip install unsloth transformers trl datasets peft accelerate bitsandbytes

# GGUF 변환용 빌드 도구
sudo apt install cmake make build-essential -y
```

---

## Stage 1: 범용 금융 지식 학습

### 1.1 데이터셋 구성

5개의 HuggingFace 데이터셋을 비율에 따라 샘플링 (`total_samples=200,000`):

| 데이터셋 | HuggingFace ID | 비율 | 샘플 | 언어 | 설명 |
|----------|----------------|------|------|------|------|
| Finance-Instruct-500k | `Josephgflowers/Finance-Instruct-500k` | 40% | 80K | EN | 금융 전반 지식 |
| Sujet-Finance-Instruct | `sujet-ai/Sujet-Finance-Instruct-177k` | 25% | 50K | EN | 금융 NLP 18개 태스크 |
| KRX Won-Instruct | `KRX-Data/Won-Instruct` | 20% | 40K | KO | 한국시장 전문 (핵심) |
| Trading Dataset v2 | `darkknight25/trading_dataset_v2` | 10% | 20K | EN | 기술적 지표 → 매매 시그널 |
| FinGPT Sentiment | `FinGPT/fingpt-sentiment-train` | 5% | 10K | EN | 금융 뉴스 감성 분석 |

각 데이터셋의 컬럼 구조가 다르므로 전용 로더가 ChatML 형식으로 통일:

```
Finance-Instruct:   system / user / assistant          → 그대로 사용
Sujet-Finance:      system_prompt / user_prompt / answer → 매핑
KRX Won-Instruct:   prompt / original_response         → user/assistant로 매핑
Trading Dataset:    instruction / input / output        → instruction+input을 user로 합침
FinGPT Sentiment:   instruction / input / output        → instruction+input을 user로 합침
```

### 1.2 ChatML 형식

모든 데이터는 Qwen2.5의 네이티브 채팅 포맷으로 통일:

```
<|im_start|>system
{시스템 프롬프트}<|im_end|>
<|im_start|>user
{사용자 질문}<|im_end|>
<|im_start|>assistant
{모델 응답}<|im_end|>
```

### 1.3 학습 실행

```bash
conda activate unsloth_env
python finetune/training/stage1.py
```

옵션 조정:
```bash
python finetune/training/stage1.py --epochs 3 --total-samples 200000 --lr 2e-4
```

### 1.4 Stage 1 학습 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| 베이스 모델 | `unsloth/Qwen2.5-7B-Instruct` | 4-bit NF4 양자화 로드 |
| LoRA rank | 32 | A: (hidden×32), B: (32×hidden) |
| LoRA alpha | 16 | 스케일링: alpha/r = 0.5 |
| target modules | q,k,v,o_proj + gate,up,down_proj | 28 layers × 7 modules = 196 adapters |
| 학습 가능 파라미터 | ~57M / 7.6B (0.75%) | |
| Epochs | 2 | |
| Batch size | 2 | |
| Gradient accumulation | 8 | Effective batch = 16 |
| Learning rate | 2e-4 | Cosine schedule + 10% warmup |
| Optimizer | AdamW 8-bit | VRAM 절약 |
| Packing | True | 짧은 예제 묶기 → 패딩 최소화 |
| 총 스텝 | ~25,000 | |
| 학습 시간 | ~30시간 | RTX 3090 Ti 기준 |

### 1.5 Stage 1 학습 결과

| 항목 | 값 |
|------|-----|
| Loss 변화 | 0.93 → 0.80 → 0.68 |
| LoRA adapter 크기 | 308MB |
| GGUF Q4_K_M 크기 | 4.5GB |
| VRAM 사용 | ~23GB / 24.6GB |

---

## Stage 2: 전략 특화 학습

### 2.1 Synthetic Data 생성

12개 태스크 유형, 5,000건의 학습 데이터를 자동 생성:

| 태스크 | 비율 | 건수 | 설명 |
|--------|------|------|------|
| full_analysis | 22% | 1,100 | 종합 분석 (regime + 포지션 + valuation) |
| regime_classification | 10% | 500 | 시장 regime 분류 |
| overnight_gap | 10% | 500 | 미국장 연동 오버나이트 갭 전략 |
| sector_analysis | 8% | 400 | US 이벤트 → 한국 섹터 영향 (25개 시나리오) |
| risk_management | 7% | 350 | MDD/레버리지 리스크 관리 |
| regime_transition | 7% | 350 | regime 전환 시 포지션 조정 |
| opening_momentum | 6% | 300 | 시초가 갭업 모멘텀 |
| fx_impact | 6% | 300 | 환율 변동 → 수출/내수주 영향 |
| stop_loss_management | 6% | 300 | 기존 포지션 손절/익절 판단 |
| intraday_rebalancing | 6% | 300 | 장중 포트폴리오 리밸런싱 |
| what_if | 6% | 300 | What-if 시나리오 분석 (7가지 충격) |
| multi_step | 6% | 300 | 멀티스텝 추론 (regime→포지션→리스크) |

**다양성 확보 기법:**
- 질문 템플릿: 태스크별 8~12개 패턴 (총 86종)
- 시장 데이터 표현: 4가지 형식 (bullet 40%, sentence 20%, compact 20%, table 20%)
- 컨텍스트 변주: 날짜, NAV, 투자이력, 리스크 한도를 랜덤 부여
- 5개 regime: STRONG_BULL, MILD_BULL, WEAKENING, SHORT_TERM_STRESS, BEAR

### 2.2 실제 시장 데이터 통합

`data/market_daily/` 디렉토리의 KRX 일봉 데이터(2006~2026, 650종목)에서 실제 시장 스냅샷을 추출하여 학습 데이터의 사실성을 높입니다.

```
data/market_daily/
├── year=2020/
│   ├── month=1/data.parquet
│   ├── month=2/data.parquet
│   └── ...
├── year=2021/
└── ...
```

**MarketSnapshotStore** (`finetune/data/market_snapshot.py`):
- 파티션 parquet에서 날짜 범위 필터링 로드
- 종목별 파생 피처 계산: daily return, SMA-20/60, realized vol 20d
- 시장 전체 지표 계산: 시총 가중 수익률, breadth, 10d momentum, 10d vol
- 일자별 regime 자동 분류 (threshold 기반)
- `sample_snapshot(regime=)`: 해당 regime 날짜에서 랜덤 스냅샷 반환

**Regime 분류 기준:**

| Regime | 조건 |
|--------|------|
| BEAR | 시장수익률 < -2%, breadth < 30%, momentum < -5% |
| SHORT_TERM_STRESS | 시장수익률 < -1.5%, vol > 25% |
| STRONG_BULL | 시장수익률 > +1%, breadth > 65%, momentum > +3% |
| WEAKENING | 시장수익률 < 0%, breadth 35~50% |
| MILD_BULL | 시장수익률 > 0%, breadth > 50% |

**실제 스냅샷 통계 (2020-01~2025-12):**

| Regime | 날짜 수 |
|--------|---------|
| MILD_BULL | 755 |
| WEAKENING | 435 |
| SHORT_TERM_STRESS | 179 |
| STRONG_BULL | 75 |
| BEAR | 24 |
| **합계** | **1,468** |

`--real-data-ratio 0.7` 설정으로 70%는 실제 데이터 기반, 30%는 순수 synthetic으로 생성하여 다양성과 사실성을 모두 확보합니다.

**실제 데이터 적용 범위:**
- 시장 지표 (KOSPI 변동률, momentum, vol, breadth): 실제 값 사용
- 종목 valuation (PER, PBR): 실제 값 사용 (0.5<PER<200, 0.01<PBR<50 범위)
- Telegram 감성, 외국인/기관 수급, USD/KRW, VIX: regime 일관 랜덤 생성
- EV/EBITDA, ROE, F-Score: 항상 랜덤 (원본 데이터에 없음)

### 2.3 Synthetic Data 생성 명령

```bash
# 기본 (5,000건, 실제 시장 데이터 없이)
python finetune/data/synthetic.py

# 실제 시장 데이터 포함 (권장)
python finetune/data/synthetic.py --use-real-data --start-date 2020-01-01

# 전체 옵션
python finetune/data/synthetic.py \
    --count 5000 \
    --use-real-data \
    --start-date 2020-01-01 \
    --end-date 2025-12-31 \
    --real-data-ratio 0.7 \
    --output finetune/data/synthetic_data.json
```

### 2.4 Stage 2 학습 실행

```bash
conda activate unsloth_env
python finetune/training/stage2.py
```

옵션 조정:
```bash
python finetune/training/stage2.py --epochs 5 --lr 3e-5
```

### 2.5 Stage 2 학습 파라미터

Stage 1과의 핵심 차이:

| 파라미터 | Stage 1 | Stage 2 | 이유 |
|---------|---------|---------|------|
| 데이터 | HuggingFace 200K | Synthetic 5K | 범용 → 전략 특화 |
| 베이스 | Qwen2.5-7B 원본 | Stage 1 LoRA 병합 | 누적 학습 |
| Learning rate | 2e-4 | 5e-5 | 이미 학습된 모델이므로 낮게 |
| Epochs | 2 | 3~5 | 데이터가 적으므로 더 많이 반복 |
| Grad accumulation | 8 | 4 | Effective batch 8 |
| 예상 스텝 | ~25,000 | ~1,875 | |
| 예상 시간 | ~30시간 | ~1~2시간 | |

Stage 2 내부 동작:
1. Qwen2.5-7B-Instruct 4-bit 로드
2. Stage 1 LoRA adapter를 직접 로드 (또는 `--skip-stage1`로 베이스부터)
3. Synthetic 데이터로 SFT 학습
4. Stage 2 LoRA 저장 (`output_stage2/final/`)
5. GGUF Q4_K_M 자동 변환 (`output_stage2/gguf/`)

---

## Stage 3: DPO 정렬 학습

### 3.1 DPO (Direct Preference Optimization)란?

SFT(Stage 2)는 "정답만 보여주며 따라하라"고 학습합니다. DPO는 한 단계 더 나아가 **"좋은 응답과 나쁜 응답을 비교"**하여 모델이 좋은 패턴을 선호하도록 정렬합니다.

```
SFT (Stage 2):  prompt → chosen (정답 하나만 학습)
DPO (Stage 3):  prompt → chosen (좋은 응답) vs rejected (나쁜 응답) → 차이를 학습
```

DPO의 장점:
- Reward model이 필요 없음 (RLHF 대비 단순)
- 학습이 안정적 (reference model 대비 KL divergence로 조절)
- SFT만으로는 교정하기 어려운 **나쁜 습관**(스키마 위반, 전략 모순, 환각)을 억제

### 3.2 DPO Preference Pair 생성

`gen_dpo_pairs.py`가 Stage 2 synthetic data에서 preference pair를 자동 생성:

| Rejection 타입 | 비율 | 설명 |
|----------------|------|------|
| strategy_violation | 30% | regime과 맞지 않는 exposure/포지션 (BEAR인데 풀롱 등) |
| quality_degradation | 30% | 근거 없음, 너무 짧음, 모든 종목에 같은 이유 |
| schema_violation | 20% | 필수 필드 누락, 잘못된 타입 (exposure가 "high" 등) |
| hallucination | 20% | 존재하지 않는 종목, PER 2000, leverage 5x |

추가로 `--collect-model-rejects` 옵션으로 Stage 2 모델의 **실제 나쁜 출력**도 수집 가능:
- 모델에 높은 temperature(0.7)로 응답 생성
- 스키마 불량/필드 누락인 출력을 rejected로 채택

### 3.3 DPO 데이터 생성 명령

```bash
# 기본 (2,000 pairs, synthetic rejected만)
python finetune/data/gen_dpo_pairs.py

# Stage 2 모델 출력도 수집 (Ollama 필요)
python finetune/data/gen_dpo_pairs.py \
    --collect-model-rejects \
    --model qwen2.5-kospi-ft-s2 \
    --model-reject-ratio 0.15

# 전체 옵션
python finetune/data/gen_dpo_pairs.py \
    --count 2000 \
    --input finetune/data/synthetic_data.json \
    --output finetune/data/dpo_pairs.json
```

### 3.4 Stage 3 학습 실행

```bash
conda activate unsloth_env
python finetune/training/stage3_dpo.py
```

옵션 조정:
```bash
python finetune/training/stage3_dpo.py --beta 0.05 --epochs 2 --lr 5e-7
```

### 3.5 Stage 3 학습 파라미터

| 파라미터 | Stage 2 (SFT) | Stage 3 (DPO) | 이유 |
|---------|---------------|---------------|------|
| 방법 | SFT | DPO | 정답 학습 → 선호 정렬 |
| 데이터 | Synthetic 5K | DPO pairs 2K | chosen/rejected 쌍 |
| Learning rate | 5e-5 | 5e-7 | DPO는 매우 낮은 lr 필요 |
| Epochs | 3~5 | 1~2 | DPO는 과적합에 민감 |
| Batch size | 2 | 1 | DPO는 2x 메모리 사용 |
| Grad accumulation | 4 | 8 | Effective batch = 8 |
| Beta (KL penalty) | - | 0.1 | 높을수록 보수적 변화 |
| Optimizer | AdamW 8-bit | AdamW 8-bit | |
| 예상 시간 | ~1-2시간 | ~30분-1시간 | |

Beta 튜닝 가이드:
- `beta=0.05`: 공격적 — 많이 변함, 과적합 위험
- `beta=0.1`: 기본값 — 균형
- `beta=0.3`: 보수적 — 안전하지만 변화 적음
- `beta=0.5`: 매우 보수적 — SFT와 거의 동일

---

## GGUF 변환 & Ollama 등록

### 자동 변환 (stage2.py에 내장)

stage2.py 실행 시 학습 완료 후 GGUF 변환 + Modelfile 생성까지 자동 처리됩니다.

### 수동 변환 (자동 변환 실패 시)

Stage 2/3 모두 동일한 방식입니다. `output_stage2`를 `output_stage3`로 바꾸면 됩니다.

```bash
# 1. llama.cpp 클론 (최초 1회)
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake . -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=ON
cmake --build build --config Release -j$(nproc)
cd ..

# 2. Safetensors → GGUF F16 (Stage 3 예시)
python llama.cpp/convert_hf_to_gguf.py \
    finetune/output_stage3/gguf/ \
    --outfile finetune/output_stage3/gguf/model-f16.gguf \
    --outtype f16

# 3. GGUF F16 → Q4_K_M
llama.cpp/build/bin/llama-quantize \
    finetune/output_stage3/gguf/model-f16.gguf \
    finetune/output_stage3/gguf/model-Q4_K_M.gguf \
    Q4_K_M

# 4. 중간 파일 정리
rm finetune/output_stage3/gguf/model-f16.gguf
rm finetune/output_stage3/gguf/model-0000*.safetensors
```

### GGUF 양자화 종류 비교

| 양자화 | 비트 | 크기 | 품질 | 용도 |
|--------|------|------|------|------|
| F16 | 16-bit | 15GB | 100% | 벤치마크, 품질 중시 |
| Q8_0 | 8-bit | 7.5GB | ~99.5% | 품질 최우선 |
| Q5_K_M | 5-bit | 5.2GB | ~98% | 균형잡힌 선택 |
| **Q4_K_M** | **4-bit** | **4.5GB** | **~97%** | **실용적 최적 (현재 사용)** |
| Q3_K_M | 3-bit | 3.5GB | ~93% | 메모리 극한 |

Q4_K_M의 "K_M": K-quant 방식으로 중요한 레이어(첫/마지막 레이어, attention output)는 Q6_K(6-bit), 나머지는 Q4_K(4-bit)로 차등 양자화.

### Ollama 등록 및 실행

```bash
# Stage 2 모델 등록
ollama create qwen2.5-kospi-ft-s2 -f finetune/output_stage2/gguf/Modelfile

# Stage 3 모델 등록 (DPO 적용)
ollama create qwen2.5-kospi-ft-s3 -f finetune/output_stage3/gguf/Modelfile

# 대화형 실행
ollama run qwen2.5-kospi-ft-s3

# API 호출
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-kospi-ft-s3",
  "prompt": "KOSPI -2.3%, breadth 25%, momentum -6%. Classify regime and recommend positions.",
  "stream": false
}'
```

Python에서 사용:
```python
import requests

response = requests.post("http://localhost:11434/api/generate", json={
    "model": "qwen2.5-kospi-ft-s3",
    "prompt": "KOSPI -2.3%, foreign net selling -800B KRW. Classify regime.",
    "stream": False,
})
print(response.json()["response"])
```

---

## 모델 평가

```bash
# Base vs Fine-tuned 비교 (Ollama 실행 필요)
python finetune/evaluate.py

# 옵션
python finetune/evaluate.py \
    --base-model qwen2.5:32b \
    --ft-model qwen2.5-kospi-ft-s2 \
    --n-samples 50 \
    --output finetune/eval_results.json
```

평가 지표:
- JSON validity rate: 유효한 JSON 응답 비율
- Schema compliance: 기대 구조 일치 여부
- Signal quality: score 범위 (-1~1) 준수
- Korean rationale: 한국어 근거 제공 여부
- Latency: 응답 시간
- ROUGE score: reasoning log 대비 오버랩 (선택)

---

## 전체 실행 순서 요약

```bash
# 0. 환경 활성화
conda activate unsloth_env

# 1. Stage 1 학습 (~30시간) — 생략 가능 (--skip-stage1)
python finetune/training/stage1.py

# 2. Synthetic 데이터 생성 (~1분)
python finetune/data/synthetic.py --use-real-data --start-date 2020-01-01

# 3. Stage 2 SFT 학습 (~1~2시간)
python finetune/training/stage2.py              # Stage 1 이어서
python finetune/training/stage2.py --skip-stage1 # 또는 베이스부터

# 4. DPO 데이터 생성 (~10초)
python finetune/data/gen_dpo_pairs.py

# 5. Stage 3 DPO 학습 (~30분~1시간)
python finetune/training/stage3_dpo.py

# 6. GGUF 변환 (자동 실패 시 수동)
# → 아래 "수동 변환" 참조

# 7. Ollama 등록
ollama create qwen2.5-kospi-ft-s3 -f finetune/output_stage3/gguf/Modelfile

# 8. 테스트
ollama run qwen2.5-kospi-ft-s3

# 9. Stage 2 vs Stage 3 비교 평가
python finetune/evaluate.py \
    --base-model qwen2.5-kospi-ft-s2 \
    --ft-model qwen2.5-kospi-ft-s3
```

---

## 기술 상세

### QLoRA = Quantized LoRA

```
[4-bit 양자화 베이스 모델] ← VRAM 3.6GB (동결, 학습 안 함)
        +
[LoRA adapter A×B]         ← VRAM ~0.3GB (이것만 학습)
        +
[Gradient/Optimizer]       ← VRAM ~3-4GB (학습 중 임시)
─────────────────────────────────────────────
합계: ~7-8GB VRAM → RTX 3090 Ti (24GB)에서 여유 있게 동작
```

LoRA 원리:
```
원래: W (전체 가중치 행렬) 업데이트 → 72억 파라미터 전부 학습
LoRA: W 고정 + A×B만 학습 → ~5,700만 파라미터만 학습 (전체의 0.75%)

W' = W + A × B × (alpha / r)
         └── rank=32인 작은 행렬
```

### Packing

짧은 예제 여러 개를 max_seq_length(2048 토큰)에 묶어 GPU 효율 극대화:
```
일반:    [예제A(300토큰) + 패딩(1748)]  ← 85% 낭비
Packing: [예제A(300) + B(500) + C(400) + D(800) + 패딩(48)]  ← 2% 낭비
```

### Learning Rate Schedule (Cosine)

```
lr
0.0002 ┤    ╭──────╮
       │   ╱        ╲
       │  ╱  warmup   ╲  cosine decay
       │ ╱              ╲
       │╱                ╲
0      ┤                  ╲___
       └──────────────────────── steps
       0   10%                 100%
```

### 학습 로그 해석

```
{'loss': 0.8415, 'grad_norm': 0.188, 'learning_rate': 0.000130, 'epoch': 0.93}
```

| 지표 | 의미 | 정상 범위 |
|------|------|-----------|
| loss | 예측-정답 차이 (낮을수록 좋음) | 시작 ~0.95 → 끝 ~0.65 |
| grad_norm | 그래디언트 크기 | 0.1~1.0 (>5.0이면 불안정) |
| learning_rate | 현재 학습률 | cosine에 따라 감소 |
| epoch | 진행도 (1.0 = 전체 1바퀴) | 0~N |

---

## 트러블슈팅

### VRAM 부족 (OOM)
```bash
# batch_size 줄이기
python finetune/training/stage2.py --batch-size 1
# max_seq_length 줄이기
python finetune/training/stage2.py --max-seq-length 1024
# lora_rank 줄이기
python finetune/training/stage2.py --lora-rank 16
```

### 학습 중단 후 재개
```python
# trainer.train() 호출 시:
stats = trainer.train(resume_from_checkpoint=True)
```

### GGUF 변환 실패: "make: not found"
```bash
sudo apt install cmake make build-essential -y
```

### Stage 1 adapter 없이 Stage 2 실행
Stage 1 adapter가 없으면 Stage 2는 원본 Qwen2.5-7B에서 바로 학습합니다 (경고 출력).

### 학습 중 다른 GPU 작업 불가
학습이 VRAM ~23GB를 사용하므로 Ollama 등 다른 GPU 프로그램은 학습 완료 후 실행.
