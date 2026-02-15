# Quant Ensemble - Regime-Detecting Strategy Selection System

한국 주식시장을 위한 레짐 기반 알파 전략 선별 및 앙상블 시스템

#### 가상환경 코드
conda activate strategy_ensemble

## Overview

이 시스템은 시장 레짐(국면)을 감지하고, 각 레짐에 최적화된 알파 전략을 선택/가중하여 포트폴리오를 구성합니다.

### Key Features

- **Regime Detection**: HMM/GMM 기반 시장 국면 분류
- **Multiple Alpha Strategies**: 모멘텀, 평균회귀, 퀄리티 등 다양한 알파 전략
- **Ensemble Methods**: MoE, Stacking, Meta-Labeling 앙상블
- **Anti-Leakage**: 미래 정보 유출 방지를 위한 철저한 검증
- **Bloomberg Integration**: Bloomberg API를 통한 실시간 데이터 수집
- **Walk-Forward Optimization**: 시계열 교차검증

## Project Structure

```
quant_ensemble/
├── config/                 # 설정 파일
│   ├── backtest.yaml      # 백테스트 설정
│   └── live.yaml          # 라이브 트레이딩 설정
├── scripts/               # 실행 스크립트
│   ├── run_etl.py        # ETL 파이프라인
│   ├── train_models.py   # 모델 학습
│   ├── run_backtest.py   # 백테스트 실행
│   └── generate_dummy_data.py  # 테스트 데이터 생성
├── src/
│   ├── common/           # 공통 타입 및 유틸리티
│   ├── ingestion/        # 데이터 수집 (Bloomberg)
│   ├── features/         # 피처 엔지니어링
│   ├── labels/           # 라벨 생성 및 누수 방지
│   ├── signals/          # 시그널 모델
│   │   ├── alpha/        # 규칙 기반 알파
│   │   └── models/       # ML/DL 모델
│   ├── ensemble/         # 앙상블 방법론
│   ├── portfolio/        # 포트폴리오 관리
│   ├── backtest/         # 백테스트 엔진
│   └── live/             # 라이브 트레이딩
└── tests/                # 테스트 코드
```

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/strategy_ensemble.git
cd strategy_ensemble

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

### 가상환경 설정 
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- pandas >= 2.0
- numpy >= 1.24
- scikit-learn >= 1.3
- torch >= 2.0 (선택)
- blpapi (Bloomberg API, 선택)

## Quick Start

### 1. 더미 데이터 생성 (Bloomberg 없이 테스트)

```bash
cd quant_ensemble
python scripts/generate_dummy_data.py --n-tickers 100 --n-days 1000
```

### 2. ETL 파이프라인 실행

```bash
# Bloomberg 연결 시
python scripts/run_etl.py --config config/backtest.yaml

# 더미 데이터 사용 시
python scripts/run_etl.py --config config/backtest.yaml --synthetic
```

### 3. 모델 학습

```bash
python scripts/train_models.py --ensemble moe --data-dir data/processed
```

### 4. 백테스트 실행

```bash
python scripts/run_backtest.py --model-dir models --start-date 2023-07-01 --end-date 2024-12-31
```

## Architecture

### Signal Models

모든 시그널 모델은 `SignalModel` 인터페이스를 구현합니다:

```python
class SignalModel(ABC):
    @abstractmethod
    def fit(self, features_df, labels_df, config) -> dict

    @abstractmethod
    def predict(self, date, features_df) -> pd.DataFrame
```

### Alpha Strategies

| Strategy | Description | Typical Horizon |
|----------|-------------|-----------------|
| MomentumAlpha | 가격 모멘텀 기반 | 1-3 months |
| MeanReversionAlpha | 평균 회귀 | 1-2 weeks |
| QualityAlpha | 재무 품질 | 6-12 months |

### Ensemble Methods

| Method | Description |
|--------|-------------|
| **MoE (Mixture of Experts)** | 레짐별 전문가 모델 가중 |
| **Stacking** | OOF 예측 기반 메타 모델 |
| **Meta-Labeling** | 이진 신호 필터링 |

### Anti-Leakage Mechanisms

1. **Point-in-Time Features**: 해당 시점에 알 수 있는 정보만 사용
2. **Purged K-Fold**: 학습/검증 사이 시간 간격 (purge) 적용
3. **Embargo**: 레이블 계산 기간만큼 추가 간격

## Configuration

### backtest.yaml

```yaml
universe:
  index_ticker: "KOSPI200 Index"
  n_stocks: 100

labels:
  forward_return_days: 21
  label_type: "return"

training:
  train_end_date: "2023-06-30"
  n_folds: 5
  purge_days: 21

backtest:
  start_date: "2023-07-01"
  end_date: "2024-12-31"
  initial_capital: 100000000
  rebalance_frequency: "daily"
  commission_bps: 1.5
  tax_bps: 23.0

portfolio:
  allocation_method: "topk"
  top_k: 20
  max_weight: 0.1
  max_leverage: 1.0

ensemble:
  type: "moe"
  n_regimes: 2
  smoothing_alpha: 0.2
```

## API Reference

### BacktestEngine

```python
from backtest import BacktestEngine, BacktestConfig

config = BacktestConfig(
    start_date=pd.Timestamp("2023-07-01"),
    end_date=pd.Timestamp("2024-12-31"),
    initial_capital=100_000_000,
)

engine = BacktestEngine(config)
result = engine.run(model, features_df, prices_df)

# Results
print(result.metrics)
print(result.returns_series)
```

### MoE Ensemble

```python
from ensemble import MoEEnsemble
from signals.models.regime_nn import RegimeClassifier

# Create base models
base_models = [momentum_alpha, reversion_alpha, quality_alpha]

# Create regime model
regime_model = RegimeClassifier({"n_regimes": 2})

# Create MoE ensemble
ensemble = MoEEnsemble(
    base_models=base_models,
    regime_model=regime_model,
    config={"smoothing_alpha": 0.2},
)

# Fit and predict
ensemble.fit(features_df, labels_df)
predictions = ensemble.predict(date, features_df)
```

## Performance Metrics

백테스트 결과에는 다음 지표가 포함됩니다:

- **Returns**: Total Return, Annual Return, Monthly Returns
- **Risk**: Volatility, Max Drawdown, VaR, CVaR
- **Risk-Adjusted**: Sharpe Ratio, Sortino Ratio, Calmar Ratio
- **Trading**: Win Rate, Profit Factor, Average Win/Loss

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_leakage.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue.
