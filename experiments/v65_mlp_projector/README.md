# v65: MLP Projector + Sigmoid Contrastive Loss

## 목적
교수님 피드백 반영. z_embedding → LLM input embedding 공간의 직접 매핑 한계를 보완하기 위해 MLP projector 도입. Sigmoid (SigLIP) loss로 continual learning 대비.

## 아키텍처

```
                    ┌→ LLM → 텍스트 복원 (NLL loss, 기존과 동일)
z_embedding ────────┤
                    └→ MLP projector (3072→1024→GELU→3072) → projected z
                                                                 ↕ sigmoid loss
                                                          content_matrix (고정)
```

- **NLL 경로**: z → LLM 직접 (기존과 동일, projector 안 거침)
- **Contrastive 경로**: z → MLP → projected_z ↔ content_matrix (BCE)
- Sigmoid loss는 InfoNCE와 달리 softmax가 없어서, 새 문서 추가 시 기존 z 분포에 영향 없음

## 설정
- **Projector**: hidden=1024, layers=2, dropout=0.0
- **Contrastive loss**: Sigmoid (SigLIP), lambda=0.1
- **use_projected_z**: True (retrieval 시 projected z 사용)
- 나머지: 기본 config 상속 (AdamW, lr=1e-2, 300 epochs)

## Continual Learning (Phase 2, 미구현)
새 문서 추가 시:
1. 기존 z + MLP를 freeze (`freeze_base_z=True`, `freeze_projector=True`)
2. 새 문서의 z만 학습
3. MLP가 이미 z→retrieval 공간 매핑을 학습했으므로, 새 z도 같은 공간에 매핑됨

## 실행
```bash
python -m train.run --config experiments/v65_mlp_projector/config.py
```

## 상태
config만 작성됨. 학습 결과 미확인.
