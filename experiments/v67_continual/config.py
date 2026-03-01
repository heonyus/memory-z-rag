"""v67: Continual learning — v65 projector freeze + doc 0~49 요약 추가학습.

v65 best.pt에서 resume.
- 기존 z (50 docs text, 193 seg) freeze
- projector (MLP) freeze
- doc 0~49의 doc_sum으로 새 z 학습
"""

CONFIG = {
    # 모델
    "llm_name": "meta-llama/Llama-3.1-8B-Instruct",

    # 데이터: text 50개 로드 후, 같은 50개를 doc_sum으로 다시 append
    "text_column": "text",
    "num_docs": 50,
    "segment_len": 128,
    "append_column": "doc_sum",     # doc 0~49 요약을 뒤에 추가
    "append_num_docs": 50,
    "new_segment_len": 512,         # 요약 세그먼트 길이
    "base_num_docs": 50,            # text 세그먼트 수 기준점

    # projector (v65에서 가져옴)
    "projector_hidden": 1024,
    "projector_layers": 2,
    "projector_dropout": 0.0,
    "use_projected_z": True,

    # continual learning
    "freeze_projector": True,
    "freeze_base_z": True,

    # 학습
    "optimizer": "sgd",
    "lr": 5.0,
    "epochs": 3000,

    # contrastive
    "contrastive_loss_type": "sigmoid",
    "contrastive_lambda": 0.1,

    # 저장
    "save_dir": "experiments/v67_continual/runs",
}
