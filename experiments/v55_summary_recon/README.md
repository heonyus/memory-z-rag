# v55: Summary Reconstruction (2 docs, SGD)

## 목적
z-embedding 학습의 첫 번째 실험. 소규모 데이터(2개 문서)에서 요약 텍스트를 복원할 수 있는지 검증.

## 설정
- **데이터**: `doc_sum` (요약 텍스트), 2개 문서, segment_len=512
- **옵티마이저**: SGD, lr=5.0
- **에폭**: 3000
- **Contrastive loss**: 비활성화 (lambda=0.0)

## 핵심 변수
NLL loss만으로 z_embedding이 요약 텍스트를 복원할 수 있는지 확인하는 순수 reconstruction 실험.
문서 수가 2개로 매우 작아서 overfitting 가능성이 높지만, 컨셉 검증이 목적.

## 실행
```bash
uv run python -m train.run --config experiments/v55_summary_recon/config.py
```
