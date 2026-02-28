# v56: TriviaQA Text Reconstruction (50 docs, AdamW)

## 목적
원문 텍스트(text) 50개 문서로 스케일업. AdamW + contrastive loss를 도입하여 retrieval과 reconstruction을 동시에 학습.

## 설정
- **데이터**: `text` (원문), 50개 문서, segment_len=128 → 193개 세그먼트
- **옵티마이저**: AdamW, lr=1e-2
- **에폭**: 300
- **Contrastive loss**: InfoNCE, lambda=0.1, temperature=0.07

## 결과
| 메트릭 | 값 |
|--------|-----|
| NLL (mean/median) | 0.137 / 0.012 |
| ROUGE-L | 0.853 |
| Token F1 | 0.861 |
| MRR | 0.837 |
| Top1 / Top5 / Top10 / Top20 | 0.78 / 0.88 / 0.98 / 0.98 |

### QA (Gemini)
| top_k | EM | F1 |
|-------|------|------|
| 1 | 0.42 | 0.459 |
| 5 | 0.58 | 0.654 |
| 10 | 0.62 | 0.676 |
| 20 | 0.76 | 0.799 |

## 핵심 관찰
- segment_len=128로 짧게 자른 원문을 사용하여 세그먼트 수가 많음 (193개)
- AdamW + lr=1e-2 조합이 SGD보다 안정적으로 수렴
- contrastive loss 도입으로 retrieval 성능(MRR=0.837)이 의미 있는 수준에 도달
- v62와 결합하면 MRR=0.869까지 향상 → 원문/요약 풀이 상호 보완적

## 실행
```bash
uv run python -m train.run --config experiments/v56_triviaqa_recon/config.py
```
