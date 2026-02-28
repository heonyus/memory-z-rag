# v60: Warmup SGD (원문, lr warmup)

## 목적
SGD에 learning rate warmup을 도입. 초기 학습의 불안정성(lr=5.0으로 바로 시작)을 완화하여 수렴 품질 개선 시도.

## 설정
- **데이터**: `text` (원문), 50개 문서, segment_len=128
- **옵티마이저**: SGD, lr=1.0 (v58의 lr=5.0에서 낮춤)
- **에폭**: 3000
- **Contrastive loss**: InfoNCE, lambda=0.1
- **Warmup**: 100 iterations, lr: 0.2 → 1.0 (LinearLR)
- **Perfect match stop**: 활성화 (모든 세그먼트 완벽 복원 시 중단)

## 핵심 변수
- v58 대비 lr을 5.0 → 1.0으로 낮추고 warmup 추가
- `perfect_match_stop`: reconstruction이 완벽해지면 더 이상 NLL을 줄일 필요가 없으므로 조기 종료
- warmup으로 초반 학습 안정화 → 더 좋은 local minimum 도달 기대

## 실행
```bash
uv run python -m train.run --config experiments/v60_warmup_sgd/config.py
```
