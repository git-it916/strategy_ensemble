"""
Finetune 패키지: Qwen2.5-7B → KOSPI 퀀트 트레이딩 LLM

패키지 구조:
  config/     - 설정 (시스템 프롬프트, 데이터셋 설정, 종목 유니버스)
  data/       - 데이터 로딩 및 synthetic 데이터 생성
  training/   - Stage 1 (범용 금융) + Stage 2 (전략 특화) 학습 파이프라인

실행:
  python finetune/data/synthetic.py          # Synthetic 데이터 생성
  python finetune/training/stage1.py         # Stage 1 학습
  python finetune/training/stage2.py         # Stage 2 학습
"""
