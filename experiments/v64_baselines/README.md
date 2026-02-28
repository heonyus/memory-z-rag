# v64: Baseline Retrievers

## 목적
z-embedding retrieval과 비교할 baseline 검색 모델들의 성능 측정. BM25, E5, Contriever, DPR 4개 방법 비교.

## 평가 방식
동일한 50개 문서(193 세그먼트)에서 query → 세그먼트 검색. Gold document의 세그먼트가 상위에 있으면 hit.

**주의**: 이 평가는 "gold doc의 아무 세그먼트 hit" 기준으로, answer 텍스트가 포함된 세그먼트인지는 확인하지 않음. 따라서 baseline 성능이 과대 평가될 수 있음.

## 결과

| Retriever | Top1 | Top5 | Top10 | Top20 | MRR |
|-----------|------|------|-------|-------|-----|
| **Contriever** | 0.98 | 0.98 | 1.00 | 1.00 | 0.983 |
| **E5** (base) | 0.98 | 0.98 | 0.98 | 0.98 | 0.981 |
| **BM25** | 0.96 | 1.00 | 1.00 | 1.00 | 0.975 |
| **DPR** | 0.20 | 0.62 | 0.78 | 0.86 | 0.392 |
| z-embedding (v56) | 0.78 | 0.88 | 0.98 | 0.98 | 0.837 |

## 핵심 관찰
- BM25, E5, Contriever 모두 MRR 0.97+ → doc-segment 기준 평가가 너무 관대
- DPR만 유일하게 낮은 성능(MRR=0.392) → domain mismatch 가능성
- z-embedding(MRR=0.837)과 baseline(0.97+) 사이의 gap은 평가 기준의 차이도 포함

## 실행
```bash
python -m eval.retrieval_baseline --retriever bm25
python -m eval.retrieval_baseline --retriever e5
python -m eval.retrieval_baseline --retriever contriever
python -m eval.retrieval_baseline --retriever dpr
```
