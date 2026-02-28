# v61: Warmup Summary (요약, lr warmup)

## 목적
v60(원문+warmup)과 동일 설정에서 요약 텍스트로 변경. Warmup 기법이 원문/요약 각각에 어떤 영향을 주는지 비교.

## 설정
- **데이터**: `doc_sum` (요약), 50개 문서, segment_len=512
- **옵티마이저**: SGD, lr=1.0
- **에폭**: 3000
- **Contrastive loss**: InfoNCE, lambda=0.1
- **Warmup**: 100 iterations, lr: 0.2 → 1.0
- **Perfect match stop**: 활성화

## 핵심 변수
v60과의 차이: `text_column: text → doc_sum`, `segment_len: 128 → 512`.
요약은 세그먼트가 더 길고(512) 적으므로, 같은 warmup 설정에서 수렴 속도와 최종 성능이 어떻게 다른지 확인.

## 실행
```bash
uv run python -m train.run --config experiments/v61_warmup_summary/config.py
```
