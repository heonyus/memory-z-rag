# v63: Text 10 docs (소규모 AdamW)

## 목적
v56(50 docs, AdamW)에서 문서 수를 10개로 줄인 ablation. 데이터 규모가 retrieval 성능에 미치는 영향을 확인.

## 설정
- **데이터**: `text` (원문), 10개 문서 (디렉토리명은 5docs이지만 실제 num_docs=10), segment_len=128
- **옵티마이저**: AdamW, lr=1e-2
- **에폭**: 300
- **Contrastive loss**: InfoNCE, lambda=0.1

## 결과
| 메트릭 | 값 |
|--------|-----|
| MRR | 0.695 |
| Top1 / Top5 / Top10 / Top20 | 0.50 / 0.74 / 0.82 / 0.94 |

## 핵심 관찰
- v56(50 docs, MRR=0.837) 대비 MRR이 0.695로 하락
- 문서 수가 줄어도 세그먼트 수 자체는 줄어서 검색 공간이 작지만, 학습 데이터도 줄어서 z_embedding의 분별력이 저하
- contrastive loss에서 negative 수가 줄어든 영향도 있을 수 있음

## 실행
```bash
uv run python -m train.run --config experiments/v63_text_5docs/config.py
```
