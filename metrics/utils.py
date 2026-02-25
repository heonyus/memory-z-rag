"""집계 유틸."""


def summarize(values):
    """mean, median 계산."""
    if not values:
        return {"mean": 0.0, "median": 0.0}
    s = sorted(values)
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"mean": sum(s) / n, "median": med}
