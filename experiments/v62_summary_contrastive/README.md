# v62: Summary + Contrastive (메인 요약 모델)

## 목적
v57(요약, NLL only)을 contrastive loss로 확장. 요약 텍스트 기반의 메인 실험으로, v56(원문)과 결합하여 dual-pool retrieval 가능성 검증.

## 설정
- **데이터**: `doc_sum` (요약), 50개 문서, segment_len=512
- **옵티마이저**: SGD, lr=5.0
- **에폭**: 3000
- **Contrastive loss**: InfoNCE, lambda=0.1

## 결과

### Reconstruction
| 메트릭 | 값 |
|--------|-----|
| NLL (mean/median) | 0.012 / 0.006 |
| ROUGE-L | 0.798 |
| Token F1 | 0.802 |

### Retrieval (v62 단독)
| 메트릭 | 값 |
|--------|-----|
| MRR | 0.686 |
| Top1 / Top5 / Top10 / Top20 | 0.60 / 0.80 / 0.86 / 0.94 |

### v62 + v56 결합 Retrieval
| 방법 | MRR |
|------|-----|
| by_score (점수 기반) | 0.853 |
| by_rank (순위 기반) | 0.869 |

Score 기반: 원문 풀이 82% 승리. Rank 기반: 요약 풀이 66% 승리.

## 핵심 관찰
- NLL이 0.012로 매우 낮음 → 요약 텍스트를 거의 완벽하게 복원
- 하지만 retrieval(MRR=0.686)은 v56(0.837) 대비 낮음 → 요약이 query와의 의미적 매칭에 불리
- **v56과 결합하면 MRR=0.869** → 원문/요약 풀이 상호 보완적
- 이 관찰이 dual-pool (원문+요약) retrieval 전략의 근거

## 추가 스크립트
- `retrieval_combined.py`: v62(요약) + v56(원문) 두 풀을 합쳐서 retrieval 평가
- `qa_combined.py`: 결합된 풀로 QA 평가

## 실행
```bash
# 학습
uv run python -m train.run --config experiments/v62_summary_contrastive/config.py

# v62 + v56 결합 retrieval
python experiments/v62_summary_contrastive/retrieval_combined.py
python experiments/v62_summary_contrastive/qa_combined.py
```
