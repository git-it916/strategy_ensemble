# Unsloth QLoRA Fine-tuning Pipeline

Qwen2.5-7B-Instruct를 한국 주식시장 퀀트 트레이딩 LLM으로 파인튜닝하는 전체 파이프라인 문서.

---

## 전체 파이프라인 흐름도

```
[1] HuggingFace 데이터 다운로드 (5개 데이터셋)
 ↓
[2] 데이터 정규화 → ChatML 형식 통합
 ↓
[3] 베이스 모델 로드 (Qwen2.5-7B, 4-bit 양자화)
 ↓
[4] LoRA 어댑터 부착 (QLoRA)
 ↓
[5] SFTTrainer로 학습
 ↓
[6] LoRA 어댑터 저장 (adapter_model.safetensors)
 ↓
[7] LoRA + 베이스 모델 병합 (merge)
 ↓
[8] Safetensors → GGUF F16 변환
 ↓
[9] GGUF F16 → Q4_K_M 양자화
 ↓
[10] Ollama 등록 (Modelfile)
```

---

## Step 1: 환경 준비

### 1.1 Conda 환경 생성

```bash
conda create -n unsloth_env python=3.10 -y
conda activate unsloth_env
pip install unsloth transformers trl datasets peft accelerate bitsandbytes
```

### 1.2 시스템 빌드 도구 (GGUF 변환용)

```bash
sudo apt install cmake make build-essential -y
```

### 1.3 하드웨어 요구사항

| 항목 | 최소 | 우리 환경 |
|------|------|-----------|
| GPU VRAM | 16GB | RTX 3090 Ti 24GB |
| RAM | 32GB | 64GB DDR5-5600 |
| 디스크 | 50GB | Samsung 990 PRO 1TB |
| CUDA | 11.8+ | 12.x |

---

## Step 2: 데이터 다운로드 및 정규화

### 2.1 데이터셋 구성

5개의 HuggingFace 데이터셋을 비율에 따라 샘플링합니다.
`total_samples=200,000` 기준:

| 데이터셋 | HuggingFace ID | 비율 | 샘플 수 | 언어 | 설명 |
|----------|----------------|------|---------|------|------|
| Finance-Instruct-500k | `Josephgflowers/Finance-Instruct-500k` | 40% | 80,000 | EN | 금융 전반 지식 (정책, 투자, 경제이론) |
| Sujet-Finance-Instruct | `sujet-ai/Sujet-Finance-Instruct-177k` | 25% | 50,000 | EN | 금융 NLP 18개 태스크 (QA, 요약, 감성분석) |
| KRX Won-Instruct | `KRX-Data/Won-Instruct` | 20% | 40,000 | KO | 한국거래소 기반 한국어 금융 Q&A (핵심) |
| Trading Dataset v2 | `darkknight25/trading_dataset_v2` | 10% | 20,000 | EN | 기술적 지표 → 매매 시그널 |
| FinGPT Sentiment | `FinGPT/fingpt-sentiment-train` | 5% | 10,000 | EN | 금융 뉴스 감성 분석 |

### 2.2 각 데이터셋의 컬럼 구조 차이

각 데이터셋은 컬럼 이름이 다르므로, 전용 로더 함수가 이를 ChatML로 통일합니다:

```
Finance-Instruct:   system / user / assistant          → 그대로 사용
Sujet-Finance:      system_prompt / user_prompt / answer → 매핑
KRX Won-Instruct:   prompt / original_response         → user/assistant로 매핑
Trading Dataset:    instruction / input / output        → instruction+input을 user로 합침
FinGPT Sentiment:   instruction / input / output        → instruction+input을 user로 합침
```

### 2.3 ChatML 형식 변환

모든 데이터는 아래 형식으로 통일됩니다. 이 형식은 Qwen2.5의 네이티브 채팅 포맷입니다:

```
<|im_start|>system
{시스템 프롬프트}<|im_end|>
<|im_start|>user
{사용자 질문}<|im_end|>
<|im_start|>assistant
{모델 응답}<|im_end|>
```

**시스템 프롬프트 규칙:**
- 원본 데이터에 system prompt가 있으면 → 그대로 사용
- 없으면 → `QUANT_SYSTEM_PROMPT` (우리 전략 프롬프트)를 기본값으로 삽입
- KRX 데이터셋만 한국어 전용 시스템 프롬프트 사용
- Trading/Sentiment 데이터셋은 각각 전용 시스템 프롬프트 사용

### 2.4 셔플

모든 데이터셋을 합친 후 `random.seed(42)`로 셔플하여 학습 순서를 랜덤화합니다.
이렇게 하면 모델이 특정 데이터셋에 편향되지 않고 골고루 학습합니다.

---

## Step 3: 베이스 모델 로드 (4-bit 양자화)

### 3.1 양자화란?

모델의 가중치(weight)는 원래 FP32(32비트)나 FP16(16비트) 부동소수점으로 저장됩니다.
양자화는 이 가중치를 더 적은 비트로 표현하여 메모리를 절약하는 기술입니다:

```
FP32 (32-bit): 1개 파라미터 = 4 bytes
FP16 (16-bit): 1개 파라미터 = 2 bytes  → 50% 절약
INT8 (8-bit):  1개 파라미터 = 1 byte   → 75% 절약
INT4 (4-bit):  1개 파라미터 = 0.5 byte  → 87.5% 절약
```

Qwen2.5-7B의 파라미터 수: 약 72억개
- FP16 로드: 72억 × 2 bytes = **14.4 GB** VRAM
- 4-bit 로드: 72억 × 0.5 bytes = **3.6 GB** VRAM ← 우리가 사용하는 방식

### 3.2 NF4 (NormalFloat4)

Unsloth의 `load_in_4bit=True`는 bitsandbytes 라이브러리의 **NF4** 양자화를 사용합니다.
NF4는 일반 INT4와 다르게, 정규분포를 따르는 neural network 가중치에 최적화된 4-bit 양자화입니다:

```
일반 INT4: 0~15 사이의 균등 분포 값으로 매핑
NF4:       정규분포의 분위수(quantile)로 매핑 → 가중치 분포에 더 적합
```

이렇게 하면 정보 손실을 최소화하면서 VRAM을 대폭 절약할 수 있습니다.

### 3.3 실제 코드

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",  # Unsloth 최적화 버전
    max_seq_length=2048,                         # 입력 토큰 최대 길이
    dtype=None,                                  # auto-detect (bf16 on Ampere+)
    load_in_4bit=True,                           # NF4 양자화 적용
)
```

`unsloth/Qwen2.5-7B-Instruct`는 HuggingFace 원본 `Qwen/Qwen2.5-7B-Instruct`와 동일한 가중치이지만,
Unsloth가 커스텀 CUDA 커널로 attention, MLP 등을 최적화한 버전입니다.

---

## Step 4: LoRA 어댑터 부착 (QLoRA)

### 4.1 LoRA란?

**LoRA (Low-Rank Adaptation)**는 전체 모델을 학습하는 대신, 작은 어댑터 행렬만 학습하는 기법입니다.

```
원래 학습: W (전체 가중치 행렬) 업데이트     → 72억 파라미터 전부 학습
LoRA 학습: W는 고정(frozen), A×B만 학습      → 수백만 파라미터만 학습

W' = W + A × B
     │       │
     │       └─ LoRA 행렬 (학습 대상): rank=32 → 매우 작은 행렬
     └───────── 원본 가중치 (동결): 4-bit 양자화 상태 유지
```

### 4.2 QLoRA = Quantized + LoRA

QLoRA는 **4-bit 양자화된 베이스 모델** 위에 LoRA를 적용하는 것입니다:

```
[4-bit 양자화 베이스 모델] ← VRAM 3.6GB (동결, 학습 안 함)
        +
[LoRA 어댑터 A×B]         ← VRAM ~0.3GB (이것만 학습)
        +
[Gradient/Optimizer]       ← VRAM ~3-4GB (학습 중 임시 메모리)
─────────────────────────────────────────────
합계: 약 7-8GB VRAM (나머지는 데이터 버퍼 등)
```

RTX 3090 Ti(24GB)에서 여유 있게 돌아가는 이유가 이것입니다.

### 4.3 하이퍼파라미터

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,                    # LoRA 랭크: A는 (hidden_dim × 32), B는 (32 × hidden_dim)
    lora_alpha=16,           # 스케일링 계수: LoRA 출력에 alpha/r = 16/32 = 0.5 곱함
    target_modules=[         # LoRA를 부착할 레이어들
        "q_proj", "k_proj",  # Attention의 Query, Key
        "v_proj", "o_proj",  # Attention의 Value, Output
        "gate_proj",         # MLP의 Gate
        "up_proj",           # MLP의 Up projection
        "down_proj",         # MLP의 Down projection
    ],
    lora_dropout=0,          # Unsloth 최적화: dropout 0 권장
    bias="none",             # bias 학습 안 함
    use_gradient_checkpointing="unsloth",  # Unsloth 전용 체크포인팅
    random_state=42,
)
```

**target_modules 설명:**
Transformer의 각 레이어에는 Attention과 MLP 두 블록이 있습니다.
총 28개 레이어 × 7개 모듈 = 196개 LoRA 어댑터가 부착됩니다.

**학습 가능 파라미터:**
- 전체: 7,615,616,000 (76억)
- 학습 대상: 약 57,000,000 (5,700만) → **전체의 0.75%만 학습**

### 4.4 Gradient Checkpointing

`use_gradient_checkpointing="unsloth"`는 VRAM을 절약하는 기법입니다.
정상적으로는 forward pass의 모든 중간 결과를 메모리에 저장해야 backward pass에서 사용할 수 있습니다.
Gradient checkpointing은 일부 중간 결과만 저장하고, 나머지는 backward 시 다시 계산합니다:

```
일반:       forward 결과 전부 저장 → VRAM 多, 속도 빠름
Checkpointing: 일부만 저장, 나머지 재계산 → VRAM 少, 속도 약간 느림 (~30%)
Unsloth:    Unsloth 자체 최적화 → 일반 checkpointing보다 효율적
```

---

## Step 5: SFTTrainer로 학습

### 5.1 SFTTrainer란?

**SFT (Supervised Fine-Tuning) Trainer**는 HuggingFace의 `trl` 라이브러리에서 제공하는
언어 모델 파인튜닝 전용 트레이너입니다.

일반 Trainer와 차이점:
- 텍스트 데이터를 자동으로 토큰화
- `packing=True` 옵션으로 짧은 예제 여러 개를 하나의 시퀀스로 묶기 가능
- ChatML 등 채팅 형식 데이터 처리에 최적화

### 5.2 학습 파라미터

```python
TrainingArguments(
    num_train_epochs=2,                      # 전체 데이터 2바퀴 반복
    per_device_train_batch_size=2,           # GPU에 한 번에 올리는 샘플 수
    gradient_accumulation_steps=8,           # 8번 미니배치 후 가중치 업데이트
    learning_rate=2e-4,                      # 학습률 (0.0002)
    warmup_ratio=0.1,                        # 전체 스텝의 10%는 warmup
    lr_scheduler_type="cosine",              # 코사인 스케줄러
    logging_steps=25,                        # 25 스텝마다 로그 출력
    save_steps=500,                          # 500 스텝마다 체크포인트 저장
    save_total_limit=3,                      # 최근 3개 체크포인트만 유지
    bf16=True,                               # BFloat16 연산 (Ampere GPU)
    optim="adamw_8bit",                      # 8-bit Adam 옵티마이저 (VRAM 절약)
    seed=42,
    report_to="none",                        # W&B 등 외부 로깅 비활성화
)
```

### 5.3 실효 배치 크기 (Effective Batch Size)

```
실효 배치 = per_device_train_batch_size × gradient_accumulation_steps
          = 2 × 8
          = 16
```

GPU 메모리 때문에 한 번에 2개만 올리지만, 그래디언트를 8번 누적한 후 가중치를 업데이트하므로
사실상 16개씩 학습하는 것과 동일한 효과입니다.

### 5.4 학습 스텝 수 계산

```
total_steps = (dataset_size × epochs) / effective_batch_size
            = (200,000 × 2) / 16
            = 25,000 steps
```

packing이 켜져 있으면 실제 스텝 수는 데이터 길이에 따라 달라질 수 있습니다.

### 5.5 Packing

`packing=True`는 짧은 예제 여러 개를 `max_seq_length` (2048 토큰)에 맞춰 하나로 묶습니다:

```
일반 (packing=False):
  [예제A(300토큰) + 패딩(1748토큰)]  ← 패딩 85% → GPU 낭비
  [예제B(500토큰) + 패딩(1548토큰)]  ← 패딩 75% → GPU 낭비

Packing (packing=True):
  [예제A(300) + 예제B(500) + 예제C(400) + 예제D(800) + 패딩(48)]  ← 패딩 2% → 효율적!
```

주의: packing 시 step 수가 줄어들어 학습이 더 빨리 끝납니다.

### 5.6 Learning Rate 스케줄 (Cosine)

```
lr
0.0002 ┤    ╭──────╮
       │   ╱        ╲
       │  ╱  warmup   ╲  cosine decay
       │ ╱              ╲
       │╱                ╲
0      ┤                  ╲___
       └──────────────────────── steps
       0   2,500        22,500  25,000
          (10%)                (100%)
```

1. **Warmup (0~10%):** lr을 0에서 0.0002까지 선형 증가 → 학습 초기 불안정 방지
2. **Cosine decay (10~100%):** lr을 코사인 곡선으로 부드럽게 감소 → 수렴

### 5.7 학습 중 출력되는 로그 해석

```
{'loss': 0.8415, 'grad_norm': 0.188, 'learning_rate': 0.000130, 'epoch': 0.93}
```

| 지표 | 의미 | 정상 범위 |
|------|------|-----------|
| `loss` | 모델 예측과 정답의 차이 (낮을수록 좋음) | 시작 ~0.95 → 끝 ~0.65 |
| `grad_norm` | 그래디언트 크기 (학습 안정성 지표) | 0.1~1.0 (>5.0이면 불안정) |
| `learning_rate` | 현재 학습률 | cosine 스케줄에 따라 감소 |
| `epoch` | 학습 진행도 (1.0 = 전체 데이터 1바퀴) | 0~2.0 |

### 5.8 체크포인트

500 스텝마다 `checkpoint-{step}/` 폴더에 저장됩니다.
`save_total_limit=3`이므로 최신 3개만 유지하고 이전 것은 자동 삭제됩니다.

학습이 중단되면 다음과 같이 재개할 수 있습니다:
```python
stats = trainer.train(resume_from_checkpoint=True)  # 가장 최근 체크포인트에서 재개
```

---

## Step 6: LoRA 어댑터 저장

### 6.1 저장되는 파일

학습이 끝나면 `output_unsloth/final/` 에 LoRA 어댑터가 저장됩니다:

```
output_unsloth/final/
├── adapter_config.json          # LoRA 설정 (rank, alpha, target_modules 등)
├── adapter_model.safetensors    # 학습된 LoRA 가중치 (308MB)
├── tokenizer.json               # 토크나이저
├── tokenizer_config.json        # 토크나이저 설정
├── special_tokens_map.json      # 특수 토큰 매핑
├── added_tokens.json            # 추가 토큰
├── chat_template.jinja          # ChatML 템플릿
├── merges.txt                   # BPE merge 규칙
├── vocab.json                   # 어휘 사전
├── training_config.json         # 학습 설정 기록
└── README.md                    # 자동 생성 모델 카드
```

**핵심 파일:** `adapter_model.safetensors` (308MB)
- 전체 모델(15GB)이 아니라 LoRA 어댑터만 저장하므로 매우 작음
- 이 파일 + 원본 베이스 모델이 있으면 파인튜닝된 모델을 복원 가능

### 6.2 Safetensors 형식

`.safetensors`는 HuggingFace가 개발한 텐서 저장 형식입니다:
- `.bin` (PyTorch pickle)보다 안전함 (임의 코드 실행 불가)
- 메모리 매핑(mmap) 지원 → 대용량 파일 빠르게 로드
- 업계 표준으로 자리잡은 형식

---

## Step 7: LoRA + 베이스 모델 병합 (Merge)

### 7.1 왜 병합이 필요한가?

추론(inference) 시 LoRA를 사용하는 방법은 두 가지입니다:

```
방법 1: 베이스 모델 + LoRA 어댑터 따로 로드 (Python/HuggingFace)
  - 장점: 여러 LoRA를 교체 가능
  - 단점: HuggingFace 프레임워크 필요

방법 2: 베이스 모델에 LoRA를 병합하여 하나의 모델로 만듦 (Ollama/llama.cpp)
  - 장점: 단일 파일로 배포 가능, llama.cpp/Ollama에서 바로 사용
  - 단점: 병합 후 LoRA 분리 불가
```

Ollama에서 사용하려면 **방법 2**가 필요합니다.

### 7.2 병합 과정

```
원본 가중치 W (4-bit) → FP16으로 역양자화(dequantize)
                            ↓
             W_merged = W_fp16 + (A × B × alpha/r)
                            ↓
              병합된 전체 모델 (FP16, ~15GB safetensors)
```

Unsloth의 `model.save_pretrained_gguf()`가 이 과정을 자동으로 수행합니다.
수동으로 하려면 `model.save_pretrained_merged()`를 사용할 수도 있습니다.

### 7.3 병합 후 파일

```
output_unsloth/gguf/
├── model-00001-of-00004.safetensors   # 병합된 전체 모델 (분할)
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── model.safetensors.index.json       # 분할 파일 인덱스
├── config.json                        # 모델 설정
├── tokenizer.json
└── ...기타 토크나이저 파일들
```

총 약 15GB. 이 파일들은 GGUF 변환의 입력으로 사용된 후 삭제 가능합니다.

---

## Step 8: Safetensors → GGUF 변환

### 8.1 GGUF란?

**GGUF (GGML Universal Format)**는 llama.cpp에서 사용하는 모델 파일 형식입니다.
기존 GGML 형식의 후속으로, 모델 가중치 + 메타데이터 + 토크나이저를 하나의 파일에 담습니다.

```
Safetensors (HuggingFace 생태계)  →  GGUF (llama.cpp/Ollama 생태계)
  여러 파일 (model-0001.safetensors...)     하나의 파일 (model.gguf)
  Python 필수                               C++ 네이티브, Python 불필요
  GPU 추론 최적화                           CPU/GPU 모두 지원
```

### 8.2 변환 도구: convert_hf_to_gguf.py

llama.cpp 레포지토리의 변환 스크립트를 사용합니다:

```bash
# llama.cpp 클론 (최초 1회)
git clone https://github.com/ggml-org/llama.cpp.git

# 변환: safetensors → GGUF (FP16)
python llama.cpp/convert_hf_to_gguf.py \
    finetune/output_unsloth/gguf/ \
    --outfile finetune/output_unsloth/gguf/qwen2.5-7b-kospi-f16.gguf \
    --outtype f16
```

이 단계에서는 양자화 없이 FP16 그대로 GGUF 형식으로만 변환합니다.
출력: `qwen2.5-7b-kospi-f16.gguf` (~15GB)

---

## Step 9: GGUF 양자화 (F16 → Q4_K_M)

### 9.1 왜 양자화하는가?

GGUF F16 파일은 15GB로 여전히 크기 때문에, 추론 시 VRAM/RAM을 많이 차지합니다.
Q4_K_M 양자화로 크기를 1/3로 줄이면서 품질 손실을 최소화합니다.

### 9.2 양자화 종류 비교

| 양자화 | 비트 | 크기 | 품질 | 용도 |
|--------|------|------|------|------|
| F16 | 16-bit | 15GB | 100% (원본) | 벤치마크, 품질 중시 |
| Q8_0 | 8-bit | 7.5GB | ~99.5% | 품질 최우선 |
| Q5_K_M | 5-bit | 5.2GB | ~98% | 균형잡힌 선택 |
| **Q4_K_M** | **4-bit** | **4.5GB** | **~97%** | **실용적 최적 (우리 선택)** |
| Q4_K_S | 4-bit | 4.2GB | ~96% | Q4_K_M보다 약간 작고 약간 낮은 품질 |
| Q3_K_M | 3-bit | 3.5GB | ~93% | 메모리 극한 상황 |
| Q2_K | 2-bit | 2.8GB | ~85% | 실험용 (품질 저하 큼) |

### 9.3 Q4_K_M의 "K_M" 의미

```
Q4  = 4-bit 양자화
K   = K-quant 방식 (block-wise 양자화, 더 정밀)
M   = Medium (중요한 레이어는 더 높은 비트로 양자화)
```

K-quant는 레이어 중요도에 따라 양자화 비트를 다르게 적용합니다:
- 중요한 레이어 (첫 번째/마지막 레이어, attention output, ffn_down): **Q6_K (6-bit)**
- 나머지 레이어: **Q4_K (4-bit)**

이렇게 하면 전체 크기는 Q4에 가까우면서 품질은 더 좋습니다.

### 9.4 llama-quantize 빌드 및 실행

```bash
# llama.cpp 빌드 (최초 1회)
cd llama.cpp
cmake . -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=ON
cmake --build build --config Release -j$(nproc)

# 양자화 실행
./build/bin/llama-quantize \
    finetune/output_unsloth/gguf/qwen2.5-7b-kospi-f16.gguf \
    finetune/output_unsloth/gguf/qwen2.5-7b-kospi-Q4_K_M.gguf \
    Q4_K_M
```

출력 로그 예시:
```
llama_model_quantize_impl: model size  = 14526.27 MiB
llama_model_quantize_impl: quant size  =  4460.45 MiB
```

14.5GB → 4.5GB로 약 **69% 크기 감소**.

### 9.5 양자화 후 정리

F16 GGUF와 merge된 safetensors는 더 이상 필요 없으므로 삭제하여 디스크를 절약합니다:

```bash
rm finetune/output_unsloth/gguf/qwen2.5-7b-kospi-f16.gguf       # 15GB 절약
rm finetune/output_unsloth/gguf/model-0000*.safetensors          # 15GB 절약
rm finetune/output_unsloth/gguf/model.safetensors.index.json
```

---

## Step 10: Ollama 등록

### 10.1 Modelfile 생성

Ollama는 `Modelfile`이라는 설정 파일을 사용하여 모델을 등록합니다.
Docker의 `Dockerfile`과 유사한 개념입니다:

```Dockerfile
# GGUF 파일 경로
FROM /path/to/qwen2.5-7b-kospi-Q4_K_M.gguf

# 추론 파라미터
PARAMETER temperature 0.3        # 낮을수록 결정적(deterministic) 응답
PARAMETER num_predict 4096       # 최대 생성 토큰 수
PARAMETER stop "<|im_end|>"      # 생성 중지 토큰
PARAMETER stop "<|im_start|>"    # 생성 중지 토큰

# 시스템 프롬프트 (모든 대화에 자동 삽입)
SYSTEM """
You are a Korean equity market quantitative analyst...
"""
```

### 10.2 Ollama 등록 및 실행

```bash
# 모델 등록
ollama create qwen2.5-kospi-ft -f finetune/output_unsloth/gguf/Modelfile

# 대화형 실행
ollama run qwen2.5-kospi-ft

# API로 호출 (프로그래밍용)
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-kospi-ft",
  "prompt": "Current regime: WEAKENING. What positions should we take?",
  "stream": false
}'
```

### 10.3 Python에서 사용

```python
import requests

response = requests.post("http://localhost:11434/api/generate", json={
    "model": "qwen2.5-kospi-ft",
    "prompt": "KOSPI -2.3%, foreign net selling -800B KRW. Classify regime.",
    "stream": False,
})
print(response.json()["response"])
```

---

## 최종 디렉토리 구조

```
finetune/
├── train_unsloth.py              # 학습 스크립트
├── train_unsloth.md              # 이 문서
└── output_unsloth/
    ├── checkpoint-24000/         # 학습 체크포인트 (최근 3개)
    ├── checkpoint-24500/
    ├── checkpoint-25000/
    ├── final/                    # LoRA 어댑터 (308MB)
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   ├── training_config.json
    │   └── ...토크나이저 파일들
    └── gguf/                     # GGUF 모델 (Ollama용)
        ├── qwen2.5-7b-kospi-Q4_K_M.gguf  # 최종 모델 (4.5GB)
        └── Modelfile                      # Ollama 설정
```

---

## 실제 학습 결과 요약

| 항목 | 값 |
|------|-----|
| 베이스 모델 | Qwen2.5-7B-Instruct |
| 학습 방법 | QLoRA (4-bit NF4 + LoRA rank=32) |
| 학습 데이터 | 200,000 samples (5개 데이터셋) |
| Epochs | 2 |
| Total steps | ~25,000 |
| 학습 시간 | ~30시간 (RTX 3090 Ti) |
| Loss 변화 | 0.93 → 0.80 → 0.68 |
| LoRA 어댑터 크기 | 308MB |
| GGUF Q4_K_M 크기 | 4.5GB |
| GPU | NVIDIA RTX 3090 Ti (24GB VRAM) |
| VRAM 사용 | ~23GB / 24.6GB |

---

## 트러블슈팅

### GGUF 변환 실패: "make: not found"
```bash
sudo apt install cmake make build-essential -y
```

### 학습 중단 후 재개
```python
# train() 함수에서:
stats = trainer.train(resume_from_checkpoint=True)
```

### VRAM 부족 (OOM)
- `batch_size` 줄이기: 2 → 1
- `max_seq_length` 줄이기: 2048 → 1024
- `lora_rank` 줄이기: 32 → 16

### 학습 중 다른 GPU 작업 불가
학습이 VRAM을 23GB 사용하므로 Ollama 등 다른 GPU 프로그램 실행 불가.
학습 완료 후 사용하거나, CPU 모드로 실행해야 합니다.
