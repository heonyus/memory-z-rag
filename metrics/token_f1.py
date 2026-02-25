"""Token F1 (unigram overlap 기반)."""

from collections import Counter


def token_f1(gen_ids, ref_ids):
    if not gen_ids or not ref_ids:
        return 0.0
    gc = Counter(gen_ids)
    rc = Counter(ref_ids)
    overlap = sum((gc & rc).values())
    p = overlap / len(gen_ids)
    r = overlap / len(ref_ids)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)
