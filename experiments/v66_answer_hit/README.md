# v66: Answer-based Hit Criterion 재평가

## 목적
v48(mz-RAG)과 동일한 answer-based hit 기준으로 v56 z-retrieval과 v64 baselines을 재평가.

## 배경
v56/v64의 기존 평가는 "gold doc의 아무 세그먼트 hit" (doc-based) 기준 → 과대평가.
v48은 "정답 텍스트가 포함된 세그먼트 hit" (answer-based) 기준 → 더 엄격하고 정확.

## Hit Criterion 차이
- **Doc-based (v56/v64)**: `seg_idx in doc_seg_set` → 문서의 아무 세그먼트면 OK
- **Answer-based (v48/v66)**: `_segment_contains_answer(seg_text, answers)` → 정답 포함 세그먼트만

## 결과 비교

### v64 (doc-based) vs v66 (answer-based)
| Retriever | v64 Top1 | v66 Top1 | v64 MRR | v66 MRR |
|-----------|----------|----------|---------|---------|
| z-embedding | 0.78 | TBD | 0.837 | TBD |
| BM25 | 0.96 | TBD | 0.975 | TBD |
| E5 | 0.98 | TBD | 0.981 | TBD |
| Contriever | 0.98 | TBD | 0.983 | TBD |
| DPR | 0.20 | TBD | 0.392 | TBD |

### v48 참고 (answer-based, mz-RAG)
| Retriever | Top1 | MRR |
|-----------|------|-----|
| z-embedding (v41) | 0.68 | 0.733 |
| BM25 | 0.68 | 0.787 |
| Contriever | 0.70 | 0.821 |

## 실행
```bash
cd /home/lhe339/data/memory-z-rag
python -m eval.retrieval_v66 --checkpoint experiments/v56_triviaqa_recon/runs/20260226_073923/best.pt
python -m eval.retrieval_baseline_v66 --retriever bm25
python -m eval.retrieval_baseline_v66 --retriever e5
python -m eval.retrieval_baseline_v66 --retriever contriever
python -m eval.retrieval_baseline_v66 --retriever dpr
```
