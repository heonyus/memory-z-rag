# v59: Summary + Contrastive (SGD)

## 목적
v57(요약, contrastive 없음)에 contrastive loss를 추가. NLL만 사용한 v57 대비 contrastive loss가 retrieval 성능에 미치는 영향을 검증.

## 설정
- **데이터**: `doc_sum` (요약), 50개 문서, segment_len=512
- **옵티마이저**: SGD, lr=5.0
- **에폭**: 3000
- **Contrastive loss**: InfoNCE, lambda=0.1 (v57 대비 추가)
- **Early stopping**: patience=100

## 핵심 변수
v57과의 유일한 차이: `contrastive_lambda: 0.0 → 0.1`.
이 실험으로 contrastive loss의 효과를 분리해서 측정할 수 있음.

## 실행
```bash
uv run python -m train.run --config experiments/v59_summary_contrastive/config.py
```
