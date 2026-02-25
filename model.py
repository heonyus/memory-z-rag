"""Z-token 모델. Frozen LLM + 학습 가능한 z_embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ZModel(nn.Module):
    """세그먼트마다 z-token 하나를 학습해서 텍스트를 복원하는 모델.

    LLM은 완전히 freeze하고, z_embeddings만 학습한다.
    """

    def __init__(self, llm_name, num_segments, num_z_tokens=1,
                 quantization="4bit", device="cuda"):
        super().__init__()
        self.device = device
        self.num_z_tokens = num_z_tokens
        self.num_segments = num_segments

        # ── LLM 로드 (4bit 양자화) ──
        quant_config = None
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        if quant_config is not None:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_name, quantization_config=quant_config,
                device_map="auto", torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device)

        # LLM 파라미터 전부 freeze
        for p in self.llm.parameters():
            p.requires_grad = False

        self.hidden_size = self.llm.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── z_embeddings: 각 세그먼트마다 학습 가능한 벡터 ──
        self.z_embeddings = nn.Embedding(num_segments, self.hidden_size)
        nn.init.normal_(self.z_embeddings.weight, mean=0.0, std=0.02)
        self.z_embeddings = self.z_embeddings.to(device)

        print(f"[ZModel] LLM={llm_name}, hidden={self.hidden_size}, "
              f"segments={num_segments}, z_tokens={num_z_tokens}")

    def forward(self, seg_idx, input_ids, attention_mask):
        """Teacher forcing으로 NLL loss 계산.

        Args:
            seg_idx: 세그먼트 인덱스 (batch,)
            input_ids: 세그먼트 토큰 (batch, seq_len)
            attention_mask: 어텐션 마스크 (batch, seq_len)

        Returns:
            dict with "loss" (NLL) and "logits"
        """
        B = input_ids.size(0)

        # z-token embedding 가져오기
        z_embed = self.z_embeddings(seg_idx).unsqueeze(1)  # (B, 1, hidden)
        z_embed = z_embed.to(dtype=torch.bfloat16)

        # 문서 토큰 embedding
        doc_embed = self.llm.get_input_embeddings()(input_ids)

        # [z, tok_0, tok_1, ..., tok_{n-2}] → 예측 대상: [tok_0, tok_1, ..., tok_{n-1}]
        combined = torch.cat([z_embed, doc_embed[:, :-1, :]], dim=1)

        # attention mask도 맞춰줌
        z_mask = torch.ones(B, self.num_z_tokens, dtype=attention_mask.dtype,
                            device=attention_mask.device)
        combined_mask = torch.cat([z_mask, attention_mask[:, :-1]], dim=1)

        # LLM forward
        outputs = self.llm(inputs_embeds=combined, attention_mask=combined_mask)

        # ★ logit slicing: z-token 위치의 logit이 첫 번째 토큰을 예측
        logits = outputs.logits[:, self.num_z_tokens - 1:, :]

        # labels 만들기 (padding 위치는 -100으로 무시)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # NLL loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, seg_idx, max_new_tokens=128):
        """z-token에서 텍스트 생성 (autoregressive)."""
        self.eval()
        idx = torch.tensor([seg_idx], device=self.device)
        z_embed = self.z_embeddings(idx).unsqueeze(1).to(dtype=torch.bfloat16)

        out = self.llm.generate(
            inputs_embeds=z_embed,
            attention_mask=torch.ones(1, self.num_z_tokens,
                                     device=self.device, dtype=torch.long),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def get_z_embedding(self, seg_idx):
        """검색용: 세그먼트의 z-embedding 벡터 반환."""
        idx = torch.tensor([seg_idx], device=self.device)
        return self.z_embeddings(idx).squeeze(0).detach()
