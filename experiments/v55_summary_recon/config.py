"""v55: summary 텍스트 복원, 2 docs, 3000 epochs."""

CONFIG = {
    "text_column": "doc_sum",
    "num_docs": 2,
    "segment_len": 512,

    "optimizer": "sgd",
    "lr": 5.0,
    "epochs": 3000,
    "contrastive_lambda": 0.0,
    "checkpoint_every": 500,

    "save_dir": "experiments/v55_summary_recon/runs",
}
