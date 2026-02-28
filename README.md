# memory-z-rag

```bash
# trainig
python -m train.run --config experiments/v65_mlp_projector/config.py

# evaluation
python -m eval.retrieval --checkpoint runs/.../best.pt
python -m eval.qa --checkpoint runs/.../best.pt
python -m eval.recon --checkpoint runs/.../best.pt
```
