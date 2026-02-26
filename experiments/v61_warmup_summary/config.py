"""v61: v59 대비 lr 낮추고 warmup 추가. summary 텍스트, SGD lr=1.0, warmup 0.2→1.0."""

CONFIG = {
    "csv_path": "data/docs_triviaqa_50.csv",
    "text_column": "doc_sum",
    "num_docs": 50,
    "segment_len": 512,

    "optimizer": "sgd",
    "lr": 1.0,
    "weight_decay": 0.0,
    "epochs": 3000,
    "contrastive_lambda": 0.1,
    "early_stop_patience": 0,
    "checkpoint_every": 500,

    # warmup: lr starts at 0.2*1.0=0.2, linearly ramps to 1.0 over 100 epochs
    "warmup_iters": 100,
    "warmup_start_factor": 0.2,

    # stop when all segments perfectly reconstructed
    "perfect_match_stop": True,

    "save_dir": "experiments/v61_warmup_summary/runs",
}
