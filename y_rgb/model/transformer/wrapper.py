import torch
from torch import nn

from .module import DecoderForCausalLM, DecoderForConditionalLLM, Encoder
from ..base_model import BaseModel
from ...common.registry import registry


@registry.register_model("TransformerForConditionalLLM")
class TransformerForConditionalLLM(BaseModel):
    def __init__(
        self,
        vocab_n,
        num_layers: int,
        d: int = 512,
        n: int = 4,
        max_len: int = 30,
        d_ff: int = 1024,
        dropout: float = 0.1,
        use_rope: bool = True,
        *args,
        **kwargs
    ):
        """
        @param n: attention head
        @param max_len: for sinusoidal pos encode
        """
        super().__init__()
        self.encoder = Encoder(
            vocab_n = vocab_n,
            num_layers = num_layers,
            d = d,
            n = n,
            max_len = max_len,
            d_ff = d_ff,
            dropout = dropout,
            use_rope = use_rope,
        )
        self.decoder = DecoderForConditionalLLM(
            vocab_n = vocab_n,
            num_layers = num_layers,
            d = d,
            n = n,
            max_len = max_len,
            d_ff = d_ff,
            dropout = dropout,
            use_rope = use_rope,
        )
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, y, x_mask, y_mask):
        m = self.encoder(x, x_mask)
        return self.decoder(y, m, y_mask, x_mask)

    def beam_search(self):
        raise NotImplementedError

    def greedy_decode(self, generator, x, x_mask, max_len, bos, eos):
        memory = self.encoder(x, x_mask)
        y_hat = torch.full((x.shape[0], 1), fill_value = bos).cuda()
        for i in range(max_len - 1):
            out = self.decoder(y_hat, memory, None, x_mask)
            next_word = generator(out)
            y_hat = torch.cat([y_hat, next_word.unsqueeze(-1)], dim = 1)

            if torch.all(next_word == eos):
                break
        return y_hat

@registry.register_model("TransformerForClassification")
class TransformerForClassification(BaseModel):
    def __init__(
        self,
        vocab_n,
        num_layers: int,
        d: int = 512,
        n: int = 4,
        max_len: int = 30,
        d_ff: int = 1024,
        dropout: float = 0.1,
        use_rope: bool = True,
        *args,
        **kwargs
    ):
        """
        @param n: attention head
        @param max_len: for sinusoidal pos encode
        """
        super().__init__()
        self.encoder = Encoder(
            vocab_n = vocab_n,
            num_layers = num_layers,
            d = d,
            n = n,
            max_len = max_len,
            d_ff = d_ff,
            dropout = dropout,
            use_rope = use_rope,
        )
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, x_mask):
        repr = self.encoder(x, x_mask)
        logit = self.mean_pooling(repr, x_mask)
        return logit

    def mean_pooling(self, token_embeddings, attention_mask):
        """
        global-meaning-pooling
        this method just mean the sequence in the sequence dim based on mask

        @param token_embeddings:
        @param attention_mask: (B, l)
        @return:
        """
        attention_mask = ~attention_mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype) # [B, l, d]
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim = 1)  # [B, d]
        sum_mask = torch.clamp(input_mask_expanded.sum(dim = 1), min = 1e-4)  # [B, d]
        return sum_embeddings / sum_mask

@registry.register_model("TransformerForCausalLLM")
class TransformerForCausalLLM(BaseModel):
    def __init__(
        self,
        vocab_n,
        num_layers: int,
        d: int = 512,
        n: int = 4,
        max_len: int = 30,
        d_ff: int = 1024,
        dropout: float = 0.1,
        use_rope: bool = True,
        *args,
        **kwargs
    ):
        """
        @param n: attention head
        @param max_len: for sinusoidal pos encode
        """
        super().__init__()
        self.decoder = DecoderForCausalLM(
            vocab_n = vocab_n,
            num_layers = num_layers,
            d = d,
            n = n,
            max_len = max_len,
            d_ff = d_ff,
            dropout = dropout,
            use_rope = use_rope,
        )
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, x_mask):
        return self.decoder(x, x_mask)
