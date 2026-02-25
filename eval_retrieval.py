"""검색 평가. z-token, BM25, Contriever, DPR, E5를 비교한다.

사용법:
    python eval_retrieval.py --checkpoint runs/.../best.pt
"""

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import CONFIG, get_run_dir
from model import ZModel


# ── 유틸 함수 ──

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_segments(length, segment_len):
    if length <= 0:
        return [(0, 1)]
    if length <= segment_len:
        return [(0, length)]
    ranges = []
    start = 0
    while start < length:
        end = min(start + segment_len, length)
        ranges.append((start, end))
        if end >= length:
            break
        start += segment_len
    return ranges


# ── 데이터 로딩 ──

def load_docs_and_queries(csv_path, text_column, num_docs, tokenizer):
    """문서와 (query, answers) 쌍을 로드한다."""
    docs = []
    queries = []  # [(doc_idx, query_text, answer_list), ...]

    with open(csv_path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= num_docs:
                break
            text = row[text_column]
            if not text or not text.strip():
                continue
            encoded = tokenizer(text, add_special_tokens=False, truncation=False)
            doc_idx = len(docs)
            docs.append({"text": text, "input_ids": encoded["input_ids"]})

            # 쿼리/답변 파싱
            raw_queries = json.loads(row.get("queries", "[]"))
            raw_answers = json.loads(row.get("answers", "[]"))
            if raw_queries:
                queries.append((doc_idx, raw_queries[0], raw_answers))

    return docs, queries


def split_into_segments(docs, tokenizer, segment_len):
    seg_ids_list, seg_to_doc, doc_to_segs, seg_texts = [], [], [], []
    for doc_idx, doc in enumerate(docs):
        ids = doc["input_ids"]
        segs = []
        for start, end in build_segments(len(ids), segment_len):
            seg_idx = len(seg_ids_list)
            seg_ids_list.append(torch.tensor(ids[start:end], dtype=torch.long))
            seg_to_doc.append(doc_idx)
            segs.append(seg_idx)
            seg_texts.append(tokenizer.decode(ids[start:end], skip_special_tokens=True))
        doc_to_segs.append(segs)
    return seg_ids_list, seg_to_doc, doc_to_segs, seg_texts


# ── 검색 평가 함수 ──

def compute_metrics(rankings, queries, seg_to_doc, seg_texts, top_k_list):
    """검색 결과에서 MRR과 Top-K Accuracy를 계산한다."""
    mrr_sum = 0.0
    top_k_hits = {k: 0 for k in top_k_list}
    num_queries = len(queries)

    for q_idx, (doc_idx, query, answers) in enumerate(queries):
        ranking = rankings[q_idx]  # 세그먼트 인덱스 정렬 리스트
        answers_lower = [a.lower() for a in answers]

        # 정답이 포함된 세그먼트의 순위 찾기
        found_rank = None
        for rank, seg_idx in enumerate(ranking, 1):
            seg_text_lower = seg_texts[seg_idx].lower()
            if any(ans in seg_text_lower for ans in answers_lower):
                found_rank = rank
                break

        if found_rank is not None:
            mrr_sum += 1.0 / found_rank
            for k in top_k_list:
                if found_rank <= k:
                    top_k_hits[k] += 1

    results = {"mrr": mrr_sum / num_queries if num_queries > 0 else 0.0}
    for k in top_k_list:
        results[f"top{k}"] = top_k_hits[k] / num_queries if num_queries > 0 else 0.0
    results["num_queries"] = num_queries
    return results


# ── z-token 검색 ──

def ztoken_retrieval(model, queries, seg_ids_list, seg_texts, tokenizer, device):
    """z-token embedding과 query embedding 간 cosine similarity로 검색."""
    rankings = []
    num_segs = len(seg_ids_list)

    # 모든 z-embedding 가져오기
    with torch.no_grad():
        z_matrix = model.z_embeddings.weight.data.float()  # (N, H)
        z_matrix = F.normalize(z_matrix, dim=1)

    for doc_idx, query, answers in tqdm(queries, desc="z-token 검색"):
        # 쿼리 embedding
        tokens = tokenizer(query, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens["input_ids"].to(device)
        with torch.no_grad():
            q_embed = model.llm.get_input_embeddings()(input_ids)
            q_embed = q_embed.mean(dim=1).squeeze(0).float()  # (H,)
            q_embed = F.normalize(q_embed, dim=0)

        scores = z_matrix @ q_embed  # (N,)
        ranking = scores.argsort(descending=True).tolist()
        rankings.append(ranking)

    return rankings


# ── BM25 검색 ──

def bm25_retrieval(queries, seg_texts):
    """BM25로 검색 (rank-bm25 라이브러리 사용)."""
    from rank_bm25 import BM25Okapi

    tokenized = [text.lower().split() for text in seg_texts]
    bm25 = BM25Okapi(tokenized)

    rankings = []
    for doc_idx, query, answers in tqdm(queries, desc="BM25 검색"):
        scores = bm25.get_scores(query.lower().split())
        ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        rankings.append(ranking)

    return rankings


# ── Dense 검색 (Contriever, DPR, E5) ──

def dense_retrieval(queries, seg_texts, doc_model_name, query_model_name=None,
                    prefix_query="", prefix_doc="", pooling="mean",
                    max_length=512, device="cuda"):
    """Dense encoder로 검색. DPR처럼 query/doc 모델이 다를 수 있다."""
    if query_model_name is None:
        query_model_name = doc_model_name

    def load_encoder(name):
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModel.from_pretrained(name).to(device)
        mdl.eval()
        return tok, mdl

    def encode(texts, tok, mdl, prefix="", batch_size=16):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[i:i+batch_size]]
            tokens = tok(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_length).to(device)
            with torch.no_grad():
                out = mdl(**tokens)
                if pooling == "cls":
                    emb = out.last_hidden_state[:, 0, :]
                else:  # mean
                    mask = tokens["attention_mask"].unsqueeze(-1).float()
                    emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            embeddings.append(F.normalize(emb.float(), dim=1).cpu())
        return torch.cat(embeddings, dim=0)

    # 문서 인코딩
    print(f"  Doc 모델 로드: {doc_model_name}")
    doc_tok, doc_mdl = load_encoder(doc_model_name)
    doc_embeds = encode(seg_texts, doc_tok, doc_mdl, prefix=prefix_doc)
    del doc_mdl, doc_tok

    # 쿼리 인코딩
    if query_model_name != doc_model_name:
        print(f"  Query 모델 로드: {query_model_name}")
    q_tok, q_mdl = load_encoder(query_model_name)

    rankings = []
    label = query_model_name.split("/")[-1]
    for doc_idx, query, answers in tqdm(queries, desc=f"{label} 검색"):
        q_embed = encode([query], q_tok, q_mdl, prefix=prefix_query)
        scores = (q_embed @ doc_embeds.T).squeeze(0)
        ranking = scores.argsort(descending=True).tolist()
        rankings.append(ranking)

    del q_mdl, q_tok
    torch.cuda.empty_cache()

    return rankings


# ── 메인 ──

def main():
    parser = argparse.ArgumentParser(description="검색 평가")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text_column", type=str, default=CONFIG["text_column"])
    parser.add_argument("--num_docs", type=int, default=CONFIG["num_docs"])
    parser.add_argument("--csv_path", type=str, default=CONFIG["csv_path"])
    args = parser.parse_args()

    set_seed(CONFIG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    top_k = CONFIG["retrieval_top_k"]

    # 데이터 로딩
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    docs, queries = load_docs_and_queries(
        args.csv_path, args.text_column, args.num_docs, tokenizer
    )
    seg_ids_list, seg_to_doc, doc_to_segs, seg_texts = split_into_segments(
        docs, tokenizer, CONFIG["segment_len"]
    )
    print(f"문서: {len(docs)}개, 세그먼트: {len(seg_ids_list)}개, 쿼리: {len(queries)}개")

    if not queries:
        print("쿼리가 없어서 검색 평가를 건너뜁니다.")
        return

    # 모델 로드
    model = ZModel(CONFIG["llm_name"], len(seg_ids_list), CONFIG["num_z_tokens"],
                    CONFIG["quantization"], device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.z_embeddings.load_state_dict(ckpt["z_embeddings"])

    results = {}
    run_dir = get_run_dir(prefix="retrieval")

    # 1. z-token 검색
    print("\n=== z-token 검색 ===")
    z_rankings = ztoken_retrieval(model, queries, seg_ids_list, seg_texts, tokenizer, device)
    results["z_token"] = compute_metrics(z_rankings, queries, seg_to_doc, seg_texts, top_k)
    print(f"  {results['z_token']}")

    # 2. BM25 검색
    print("\n=== BM25 검색 ===")
    bm25_rankings = bm25_retrieval(queries, seg_texts)
    results["bm25"] = compute_metrics(bm25_rankings, queries, seg_to_doc, seg_texts, top_k)
    print(f"  {results['bm25']}")

    # 3. Contriever
    print("\n=== Contriever 검색 ===")
    contriever_rankings = dense_retrieval(
        queries, seg_texts, "facebook/contriever", device=device
    )
    results["contriever"] = compute_metrics(
        contriever_rankings, queries, seg_to_doc, seg_texts, top_k
    )
    print(f"  {results['contriever']}")

    # 4. DPR (query/doc 모델이 다름)
    print("\n=== DPR 검색 ===")
    dpr_rankings = dense_retrieval(
        queries, seg_texts,
        doc_model_name="facebook/dpr-ctx_encoder-single-nq-base",
        query_model_name="facebook/dpr-question_encoder-single-nq-base",
        pooling="cls", max_length=256, device=device,
    )
    results["dpr"] = compute_metrics(dpr_rankings, queries, seg_to_doc, seg_texts, top_k)
    print(f"  {results['dpr']}")

    # 5. E5
    print("\n=== E5 검색 ===")
    e5_rankings = dense_retrieval(
        queries, seg_texts, "intfloat/e5-large-v2",
        prefix_query="query: ", prefix_doc="passage: ",
        device=device,
    )
    results["e5"] = compute_metrics(e5_rankings, queries, seg_to_doc, seg_texts, top_k)
    print(f"  {results['e5']}")

    # 결과 저장
    with open(run_dir / "retrieval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 요약 출력
    print(f"\n=== 검색 평가 요약 ===")
    print(f"{'Method':<15} {'MRR':>8} {'Top1':>8} {'Top5':>8} {'Top10':>8}")
    print("-" * 47)
    for method, m in results.items():
        print(f"{method:<15} {m['mrr']:>8.3f} {m.get('top1',0):>8.3f} "
              f"{m.get('top5',0):>8.3f} {m.get('top10',0):>8.3f}")

    # rankings도 저장 (eval_qa.py에서 사용)
    all_rankings = {
        "z_token": z_rankings,
        "bm25": bm25_rankings,
    }
    with open(run_dir / "rankings.json", "w") as f:
        json.dump(all_rankings, f)

    print(f"\n  결과 저장: {run_dir}")


if __name__ == "__main__":
    main()
