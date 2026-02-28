# v58: Text 50 docs (SGD + contrastive)

## 목적
원문 텍스트(text) 50개 문서를 SGD로 학습. v57(요약+SGD)과 동일 조건에서 원문 텍스트로 바꿔 비교.

## 설정
- **데이터**: `text` (원문), 50개 문서, segment_len=128
- **옵티마이저**: SGD, lr=5.0
- **에폭**: 3000
- **Contrastive loss**: InfoNCE, lambda=0.1
- **Early stopping**: patience=100

## 핵심 변수
v57과의 차이점은 `text_column`만 다름 (doc_sum → text). SGD + 높은 lr 조합에서 원문 텍스트가 요약보다 유리한지 확인.

## 실행
```bash
uv run python -m train.run --config experiments/v58_text_50docs/config.py
```
