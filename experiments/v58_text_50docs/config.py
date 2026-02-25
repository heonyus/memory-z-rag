"""v58: text 원문 복원, 50 docs, SGD lr=5.0, 3000 epochs."""

CONFIG = {
    "csv_path": "data/docs_triviaqa_50.csv",
    "text_column": "text",
    "num_docs": 50,
    "segment_len": 128,

    "optimizer": "sgd",
    "lr": 5.0,
    "epochs": 3000,
    "contrastive_lambda": 0.0,
    "early_stop_patience": 100,
    "checkpoint_every": 500,

    "save_dir": "experiments/v58_text_50docs/runs",
}
