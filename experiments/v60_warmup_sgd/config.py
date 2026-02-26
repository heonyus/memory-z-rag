"""v60: SGD lr=1.0, warmup 0.2→1.0 over 100 epochs, perfect-match stop, 3000 epochs max."""

CONFIG = {
    "csv_path": "data/docs_triviaqa_50.csv",
    "text_column": "text",
    "num_docs": 50,
    "segment_len": 128,

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

    "save_dir": "experiments/v60_warmup_sgd/runs",
}
