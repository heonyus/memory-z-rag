"""v63: v56 세팅 + num_docs=5. TriviaQA 원문, AdamW lr=0.01, 300 epochs."""

CONFIG = {
    "csv_path": "data/docs_triviaqa_50.csv",
    "text_column": "text",
    "num_docs": 10,
    "segment_len": 128,

    "optimizer": "adamw",
    "lr": 1e-2,
    "weight_decay": 0.0,
    "epochs": 300,
    "contrastive_lambda": 0.1,
    "checkpoint_every": 50,

    "save_dir": "experiments/v63_text_5docs/runs",
}
