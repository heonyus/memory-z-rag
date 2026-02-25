"""BM25 검색."""

import math
import re
from collections import Counter


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.doc_freqs = []
        self.doc_lens = []
        self.df = Counter()
        self.n = 0

        for doc in corpus:
            toks = tokenize(doc)
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
        q_toks = tokenize(query)
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
