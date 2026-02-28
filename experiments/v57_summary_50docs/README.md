# v57: Summary 50 docs (SGD, contrastive 없음)

## 목적
요약 텍스트(doc_sum) 50개 문서로 스케일업. SGD + 높은 lr로 장기 학습. Contrastive loss 없이 순수 NLL만으로 retrieval이 가능한지 확인.

## 설정
- **데이터**: `doc_sum` (요약), 50개 문서, segment_len=512
- **옵티마이저**: SGD, lr=5.0
- **에폭**: 3000
- **Contrastive loss**: 비활성화 (lambda=0.0)
- **Early stopping**: patience=100

## 결과
| 메트릭 | 값 |
|--------|-----|
| MRR | 0.739 |
| Top1 / Top5 / Top10 / Top20 | 0.56 / 0.84 / 0.90 / 0.96 |

## 핵심 관찰
- NLL만으로도 retrieval이 어느 정도 가능 (MRR=0.739)
- 하지만 contrastive loss를 추가한 v59, v62에 비해 성능이 낮음
- 요약 텍스트는 segment_len=512로 더 긴 세그먼트 사용 (원문 128 대비)
- 이 실험을 contrastive loss로 확장한 것이 v62

## 실행
```bash
uv run python -m train.run --config experiments/v57_summary_50docs/config.py
```
