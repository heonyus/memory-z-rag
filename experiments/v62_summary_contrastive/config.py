"""v62: v57 + contrastive loss. summary 텍스트, SGD lr=5.0, 3000 epochs."""

CONFIG = {
    "csv_path": "data/docs_triviaqa_50.csv",
    "text_column": "doc_sum",
    "num_docs": 50,
    "segment_len": 512,

    "optimizer": "sgd",
    "lr": 5.0,
    "epochs": 3000,
    "contrastive_lambda": 0.1,
    "early_stop_patience": 100,
    "checkpoint_every": 500,

    "save_dir": "experiments/v62_summary_contrastive/runs",
}
