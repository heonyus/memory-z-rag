"""Dense 검색 인코더 (Contriever, DPR, E5 통합)."""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class DenseEncoder:
    def __init__(self, model_name, device="cuda", normalize=True,
                 max_length=256, prefix_query="", prefix_doc="",
                 pooling="mean"):
        self.device = device
        self.normalize = normalize
        self.max_length = max_length
        self.prefix_q = prefix_query
        self.prefix_d = prefix_doc
        self.pooling = pooling

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _pool(self, out, mask):
        if self.pooling == "cls":
            p = getattr(out, "pooler_output", None)
            return p if p is not None else out.last_hidden_state[:, 0]
        # mean pooling
        h = out.last_hidden_state
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    def encode(self, texts, batch_size=16, prefix=""):
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[i:i + batch_size]]
            inputs = self.tok(batch, return_tensors="pt", padding=True,
                              truncation=True, max_length=self.max_length)
            ids = inputs["input_ids"].to(self.device)
            mask = inputs["attention_mask"].to(self.device)
            with torch.no_grad():
                pooled = self._pool(self.model(ids, mask), mask)
            if self.normalize:
                pooled = F.normalize(pooled, dim=1)
            embs.append(pooled.cpu())
        return torch.cat(embs)

    def encode_queries(self, texts, batch_size=16):
        return self.encode(texts, batch_size, self.prefix_q)

    def encode_docs(self, texts, batch_size=16):
        return self.encode(texts, batch_size, self.prefix_d)
