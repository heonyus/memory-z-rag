"""InfoNCE contrastive loss + content matrix 생성."""

import torch
import torch.nn.functional as F


def contrastive_loss(query_embeds, doc_embeds, temperature=0.07, label_idx=0):
    """query vs 전체 doc embeddings 간의 InfoNCE loss."""
    query_norm = F.normalize(query_embeds.float(), dim=1)
    doc_norm = F.normalize(doc_embeds.float(), dim=1)
    logits = (query_norm @ doc_norm.T) / temperature
    labels = torch.tensor([label_idx], device=query_norm.device)
    return F.cross_entropy(logits, labels)


def build_content_matrix(llm, seg_ids):
    """각 세그먼트의 평균 content embedding을 계산.

    Returns: [num_segments, hidden_size] normalized tensor.
    """
    embeddings = []
    with torch.no_grad():
        for seg_tensor in seg_ids:
            token_embeds = llm.get_input_embeddings()(seg_tensor.unsqueeze(0).to("cuda"))
            mean_embed = token_embeds.mean(dim=1).squeeze(0).float()
            embeddings.append(mean_embed)
    return F.normalize(torch.stack(embeddings), dim=1)
