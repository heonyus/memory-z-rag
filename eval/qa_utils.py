"""QA 평가 공통 유틸: 메트릭(EM/F1), Gemini 호출, 프롬프트 빌더."""

import os
import re
import time


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


def best_metrics(pred, gold_list):
    """gold_list 중 가장 높은 EM/F1 반환."""
    best_em = max(exact_match(pred, g) for g in gold_list)
    best_f1 = max(f1_score(pred, g) for g in gold_list)
    return best_em, best_f1


def call_gemini(prompt, model_name="gemini-2.5-flash-lite", max_tokens=64):
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={"max_output_tokens": max_tokens},
    )
    return resp.text.strip()


def call_gemini_with_retry(prompt, model_name="gemini-2.5-flash-lite", k=None):
    """call_gemini + 1회 재시도."""
    try:
        return call_gemini(prompt, model_name=model_name)
    except Exception as e:
        label = f" (k={k})" if k is not None else ""
        print(f"  Gemini error{label}: {e}, retrying...")
        time.sleep(2)
        try:
            return call_gemini(prompt, model_name=model_name)
        except Exception as e2:
            print(f"  Gemini failed: {e2}")
            return ""


def build_qa_prompt(context, question):
    return (
        f"Answer the question based ONLY on the given context. "
        f"Give a short, concise answer (a few words).\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
