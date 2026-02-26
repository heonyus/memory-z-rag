"""Retriever 통합 모듈: BM25 / E5 / Contriever / DPR.

사용법:
    from eval.retrievers import build_retriever
    retriever = build_retriever("bm25", seg_texts)
    ranked = retriever.rank("Who wrote Romeo and Juliet?")  # list[int]
"""

import math
import re
import torch
import torch.nn.functional as F
from collections import Counter
from transformers import AutoModel, AutoTokenizer


# ── BM25 ──────────────────────────────────────────────────────────────

def _bm25_tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.doc_freqs = []
        self.doc_lens = []
        self.df = Counter()
        self.n = 0

        for doc in corpus:
            toks = _bm25_tokenize(doc)
            freq = Counter(toks)
            self.doc_freqs.append(freq)
            self.doc_lens.append(len(toks))
            self.df.update(freq.keys())
            self.n += 1

        self.avgdl = sum(self.doc_lens) / self.n if self.n else 0
        self.idf = {
            t: math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for t, df in self.df.items()
        }

    def get_scores(self, query):
        q_toks = _bm25_tokenize(query)
        scores = [0.0] * self.n
        for i, freq in enumerate(self.doc_freqs):
            dl = self.doc_lens[i]
            if dl == 0:
                continue
            s = 0.0
            for t in q_toks:
                if t not in freq:
                    continue
                tf = freq[t]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf.get(t, 0) * tf * (self.k1 + 1) / denom
            scores[i] = s
        return scores


# ── Dense Encoder ─────────────────────────────────────────────────────

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


# ── Retriever 인터페이스 + 구현 ───────────────────────────────────────

PRESETS = {
    "bm25": {"type": "bm25"},
    "e5": {
        "type": "dense",
        "model_name": "intfloat/e5-base-v2",
        "prefix_query": "query: ",
        "prefix_doc": "passage: ",
        "pooling": "mean",
    },
    "contriever": {
        "type": "dense",
        "model_name": "facebook/contriever-msmarco",
        "pooling": "mean",
    },
    "dpr": {
        "type": "dense",
        "model_name": "NAACL2022/spider-trivia-question-encoder",
        "pooling": "cls",
    },
}


class Retriever:
    """통합 인터페이스."""

    def rank(self, query):
        """query → segment indices sorted by relevance (best first)."""
        raise NotImplementedError


class BM25Retriever(Retriever):
    def __init__(self, seg_texts):
        self.index = BM25Index(seg_texts)

    def rank(self, query):
        scores = self.index.get_scores(query)
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


class DenseRetriever(Retriever):
    def __init__(self, seg_texts, device="cuda", **kwargs):
        self.encoder = DenseEncoder(device=device, **kwargs)
        self.doc_embs = self.encoder.encode_docs(seg_texts).to(device)
        self.device = device

    def rank(self, query):
        q = self.encoder.encode_queries([query]).to(self.device)
        scores = (q @ self.doc_embs.T).squeeze(0)
        return torch.argsort(scores, descending=True).tolist()


class DualDenseRetriever(Retriever):
    """DPR: query encoder와 doc encoder가 별도."""

    def __init__(self, seg_texts, query_model, doc_model, device="cuda", **kwargs):
        self.q_enc = DenseEncoder(model_name=query_model, device=device, **kwargs)
        self.d_enc = DenseEncoder(model_name=doc_model, device=device, **kwargs)
        self.doc_embs = self.d_enc.encode_docs(seg_texts).to(device)
        self.device = device

    def rank(self, query):
        q = self.q_enc.encode_queries([query]).to(self.device)
        scores = (q @ self.doc_embs.T).squeeze(0)
        return torch.argsort(scores, descending=True).tolist()


def build_retriever(name, seg_texts, device="cuda"):
    """이름으로 retriever 생성.

    Args:
        name: "bm25" | "e5" | "contriever" | "dpr"
        seg_texts: list[str]
        device: CUDA device
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown retriever: {name}. Available: {list(PRESETS.keys())}")

    preset = PRESETS[name]
    t = preset["type"]

    if t == "bm25":
        return BM25Retriever(seg_texts)
    elif t == "dense":
        return DenseRetriever(
            seg_texts, device=device,
            model_name=preset["model_name"],
            prefix_query=preset.get("prefix_query", ""),
            prefix_doc=preset.get("prefix_doc", ""),
            pooling=preset.get("pooling", "mean"),
        )
    elif t == "dense_dual":
        return DualDenseRetriever(
            seg_texts, device=device,
            query_model=preset["query_model"],
            doc_model=preset["doc_model"],
            pooling=preset.get("pooling", "cls"),
        )
