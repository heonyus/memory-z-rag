"""QA 평가. python -m eval.qa --checkpoint runs/.../best.pt [--config ...]"""

import argparse, json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_config


def normalize_answer(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred, gold):
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = set(p_toks) & set(g_toks)
    if not common:
        return 0.0
    p = len(common) / len(p_toks)
    r = len(common) / len(g_toks)
    return 2 * p * r / (p + r)


def call_gemini(prompt, model_name, max_tokens):
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={"max_output_tokens": max_tokens},
    )
    return resp.text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="실험별 config.py 경로")
    args = parser.parse_args()
    config = load_config(args.config)

    # TODO: 검색 결과를 context로 Gemini에 QA
    # eval/retrieval.py의 결과를 먼저 생성해야 함
    print("qa eval: retrieval 결과 필요. eval/retrieval.py를 먼저 실행.")


if __name__ == "__main__":
    main()
