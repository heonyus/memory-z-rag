"""ZModel: frozen LLM + learnable z_embeddings.

LLM은 4bit quantization으로 로드하고, 파라미터를 전부 freeze.
학습 가능한 파라미터는 z_embeddings (nn.Embedding) 뿐이다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ZModel(nn.Module):

    def __init__(self, llm_name, num_segments, num_z=1, device="cuda"):
        super().__init__()
        self.device = device
        self.num_z = num_z

        # LLM 로드 (4bit quantization)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # LLM freeze
        for param in self.llm.parameters():
            param.requires_grad = False

        self.hidden_size = self.llm.config.hidden_size

        # 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # z_embeddings: 세그먼트당 하나의 학습 가능한 벡터
        self.z_embeddings = nn.Embedding(num_segments, self.hidden_size).to(device)
        nn.init.normal_(self.z_embeddings.weight, std=0.02)

    def forward(self, seg_idx, input_ids, attention_mask):
        """Teacher forcing. z + doc[:-1] → doc 예측.

        Returns: (nll_loss, logits)
        """
        batch_size = input_ids.size(0)

        # z embedding + document embedding 결합
        z_embed = self.z_embeddings(seg_idx).unsqueeze(1).to(torch.bfloat16)
        doc_embed = self.llm.get_input_embeddings()(input_ids)
        combined = torch.cat([z_embed, doc_embed[:, :-1, :]], dim=1)

        # attention mask 결합
        z_mask = torch.ones(batch_size, self.num_z,
                            dtype=attention_mask.dtype, device=attention_mask.device)
        combined_mask = torch.cat([z_mask, attention_mask[:, :-1]], dim=1)

        # LLM forward
        logits = self.llm(inputs_embeds=combined, attention_mask=combined_mask).logits
        logits = logits[:, self.num_z - 1:, :]  # z 위치 이후만 사용

        # loss 계산 (pad 위치는 ignore)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        return loss, logits

    @torch.no_grad()
    def generate(self, seg_idx, max_new_tokens=256):
        """z embedding만으로 텍스트 생성 (autoregressive)."""
        self.eval()
        idx_tensor = torch.tensor([seg_idx], device=self.device)
        z_embed = self.z_embeddings(idx_tensor).unsqueeze(1).to(torch.bfloat16)

        output_ids = self.llm.generate(
            inputs_embeds=z_embed,
            attention_mask=torch.ones(1, self.num_z,
                                     device=self.device, dtype=torch.long),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return output_ids[0].tolist()
