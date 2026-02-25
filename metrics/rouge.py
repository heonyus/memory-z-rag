"""ROUGE-L F1 (token-level LCS 기반)."""


def lcs_length(a, b):
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    if n > m:
        a, b = b, a
        m, n = n, m
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l_f1(gen_ids, ref_ids):
    if not gen_ids or not ref_ids:
        return 0.0
    lcs = lcs_length(gen_ids, ref_ids)
    p = lcs / len(gen_ids)
    r = lcs / len(ref_ids)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)
