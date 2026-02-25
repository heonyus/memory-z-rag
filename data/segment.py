"""토큰 시퀀스를 고정 길이 세그먼트로 분할."""


def build_segments(length, seg_len):
    """(start, end) 튜플 리스트 반환."""
    if length <= 0:
        return [(0, 1)]
    if length <= seg_len:
        return [(0, length)]
    ranges = []
    start = 0
    while start < length:
        end = min(start + seg_len, length)
        ranges.append((start, end))
        if end >= length:
            break
        start += seg_len
    return ranges
