import math
import typing

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import flash_attn
    import flash_attn.layers.rotary
    has_flash_attn = True
except ImportError:
    torch.backends.cuda.enable_flash_sdp(enabled=True)
    has_flash_attn = False

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float,
        training: bool) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(
            x, bias, scale, residual, prob, training)

    return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def bias_dropout_add_scale_fused_train(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, True)


def bias_dropout_add_scale_fused_inference(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, False)


def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.precompute()

    def precompute(self):
        inv_freq = 1.0 / \
            (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(self.max_seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq.clone())
        emb = torch.cat((freqs, freqs), dim=-1)
        # dims are: batch, seq_len, qkv, head, dim
        cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        # This makes the transformation on v an identity.
        cos_cached[:, :, 2, :, :].fill_(1.)
        sin_cached[:, :, 2, :, :].fill_(0.)

        self.register_buffer('cos_cached', cos_cached)
        self.register_buffer('sin_cached', sin_cached)

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
    if has_flash_attn:
        cos = cos[0, :, 0, 0, :cos.shape[-1]//2]
        sin = sin[0, :, 0, 0, :sin.shape[-1]//2]
        return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)
    else:
        return (qkv * cos) + (rotate_half(qkv) * sin)


# function overload
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            - math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding,
                 torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes


    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.cond_dim = cond_dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def flops(self, seq_len=128):
        per_token_flops = 0
        per_token_flops += 2 * self.dim * 3 * self.dim  # attn_qkv
        per_token_flops += 2 * seq_len * self.dim  # softmax attention
        per_token_flops += 2 * self.dim * self.dim  # attn_out
        per_token_flops += 2 * 2 * self.dim * 4 * self.dim  # mlp
        flops = per_token_flops * seq_len
        flops += 2 * self.cond_dim * 6 * self.dim  # adaLN_modulation
        return flops

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa, scale_msa, gate_msa, shift_mlp,
         scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv,
                        'b s (three h d) -> b s three h d',
                        three=3,
                        h=self.n_heads)
        cos, sin = rotary_cos_sin
        qkv = apply_rotary_pos_emb(
            qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        if has_flash_attn:
            qkv = rearrange(qkv, 'b s ... -> (b s) ...')
            if seqlens is None:
                cu_seqlens = torch.arange(
                    0, (batch_size + 1) * seq_len, step=seq_len,
                    dtype=torch.int32, device=qkv.device)
            else:
                cu_seqlens = seqlens.cumsum(-1)
            x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, seq_len, 0., causal=False)
            x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
        else:
            q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(
                1, 2), qkv[:, :, 2].transpose(1, 2)
            x = F.scaled_dot_product_attention(q, k, v)

            x = rearrange(x, 'b h s d -> b s (h d)', b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(x),
                                  None,
                                  gate_msa,
                                  x_skip,
                                  self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(
                self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout)
        return x


class DDiTBlockWithMask(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.cond_dim = cond_dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()
        self.adaLN_modulation.bias.data[2::6] = 1.0  # gate_msa
        self.adaLN_modulation.bias.data[5::6] = 1.0  # gate_mlp

    def flops(self, seq_len=128):
        per_token_flops = 0
        per_token_flops += 2 * self.dim * 3 * self.dim  # attn_qkv
        per_token_flops += 2 * seq_len * self.dim  # softmax attention
        per_token_flops += 2 * self.dim * self.dim  # attn_out
        per_token_flops += 2 * 2 * self.dim * 4 * self.dim  # mlp
        flops = per_token_flops * seq_len
        flops += 2 * self.cond_dim * 6 * self.dim  # adaLN_modulation
        return flops

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c, attention_mask=None, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa, scale_msa, gate_msa, shift_mlp,
         scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv,
                        'b s (three h d) -> b s three h d',
                        three=3,
                        h=self.n_heads)
        cos, sin = rotary_cos_sin
        qkv = apply_rotary_pos_emb(
            qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        if has_flash_attn and attention_mask is None:
            # Use flash attention only when no attention mask is needed
            qkv = rearrange(qkv, 'b s ... -> (b s) ...')
            if seqlens is None:
                cu_seqlens = torch.arange(
                    0, (batch_size + 1) * seq_len, step=seq_len,
                    dtype=torch.int32, device=qkv.device)
            else:
                cu_seqlens = seqlens.cumsum(-1)
            x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, seq_len, 0., causal=False)
            x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
        else:
            # Use standard attention with optional masking
            q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(
                1, 2), qkv[:, :, 2].transpose(1, 2)

            if attention_mask is not None:
                # Convert attention_mask to the format expected by scaled_dot_product_attention
                # attention_mask should be (batch_size, seq_len) with 1 for valid tokens, 0 for padded
                attn_mask = attention_mask.bool()  # (B, S)
                attn_mask = attn_mask.unsqueeze(
                    1) & attn_mask.unsqueeze(2)  # (B, S, S)
                attn_mask = attn_mask.unsqueeze(1)  # (B, 1, S, S)
                float_mask = torch.zeros(
                    attn_mask.shape, device=q.device, dtype=q.dtype)
                float_mask = float_mask.masked_fill(~attn_mask, -1e9)

                x = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=float_mask)
            else:
                x = F.scaled_dot_product_attention(q, k, v)

            x = rearrange(x, 'b h s d -> b s (h d)', b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(x),
                                  None,
                                  gate_msa,
                                  x_skip,
                                  self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(
                self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim,
                                          2 * hidden_size,
                                          bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)

        self.config = config
        self.vocab_size = vocab_size
        self.rounded_vocab_size = vocab_size + (128 - vocab_size % 128) % 128

        self.vocab_embed = EmbeddingLayer(
            config.model.hidden_size, self.rounded_vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.rotary_emb = Rotary(
            config.model.hidden_size // config.model.n_heads,
            max_seq_len=config.model.max_seq_len,
        )

        blocks = []
        for _ in range(config.model.n_blocks):
            blocks.append(DDiTBlock(config.model.hidden_size,
                                    config.model.n_heads,
                                    config.model.cond_dim,
                                    dropout=config.model.dropout))
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDitFinalLayer(
            config.model.hidden_size,
            self.rounded_vocab_size,
            config.model.cond_dim)

        self.register_buffer("logit_bias", torch.full((1, 1, 1), 0.0))

    def flops(self, seq_len=128):
        return sum(b.flops(seq_len) for b in self.blocks)

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, indices, sigma):
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
        x = self.output_layer(x, c)

        x = x.scatter_add(-1, indices.unsqueeze(-1),
                          self.logit_bias.to(x.dtype).expand_as(x))

        return x




class MMDDiTBlockWithMask(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1, 
                 num_experts=4, num_experts_per_tok=2, norm_topk_prob=True, share_adaln=False):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.cond_dim = cond_dim
        self.mlp_ratio = mlp_ratio

        # Attention layers (unified for both modalities)
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        # MoE parameters
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

        # Keep original MLP as the first expert (expert 0) for backward compatibility
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))

        # Add K-1 additional experts
        self.additional_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, mlp_ratio * dim, bias=True),
                nn.GELU(approximate='tanh'),
                nn.Linear(mlp_ratio * dim, dim, bias=True))
            for _ in range(num_experts - 1)
        ])

        # MoE gate with special initialization to favor expert 0
        self.gate = nn.Linear(dim, num_experts, bias=False)

        self.share_adaln = share_adaln
        # AdaLN modulation (unified for continuous modalities)
        self.adaLN_modulation_x = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation_x.weight.data.zero_()
        self.adaLN_modulation_x.bias.data.zero_()
        self.adaLN_modulation_x.bias.data[2::6] = 1.0  # gate_msa
        self.adaLN_modulation_x.bias.data[5::6] = 1.0  # gate_mlp
        if self.share_adaln:
            self.adaLN_modulation_z = None
        else:    
            # AdaLN modulation (unified for discrete modalities)
            self.adaLN_modulation_z = nn.Linear(cond_dim, 6 * dim, bias=True)
            self.adaLN_modulation_z.weight.data.zero_()
            self.adaLN_modulation_z.bias.data.zero_()
            self.adaLN_modulation_z.bias.data[2::6] = 1.0  # gate_msa
            self.adaLN_modulation_z.bias.data[5::6] = 1.0  # gate_mlp
        

    def _init_gate_weights(self):
        """Initialize gate weights to favor expert 0 at the beginning of training"""
        with torch.no_grad():
            nn.init.normal_(self.gate.weight, mean=0.0, std=0.1)
            # Set expert 0 weights to positive values to make it more likely to be selected
            self.gate.weight[:, 0].fill_(1.0)

    def get_all_experts(self):
        """Return all experts as a list, with original MLP as expert 0"""
        experts = [self.mlp]
        experts.extend(list(self.additional_experts))
        return experts

    def moe_forward(self, hidden_states):
        """
        MoE forward pass based on MoeSparseBlock implementation
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states_flat)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        # Cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )

        # Get all experts (original MLP + additional experts)
        experts = self.get_all_experts()

        # One hot encode the selected experts to create an expert mask
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if len(top_x) > 0:  # Only process if there are tokens assigned to this expert
                # Index the correct hidden states and compute the expert hidden state
                current_state = hidden_states_flat[top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                # Add to final hidden states
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    def flops(self, seq_len=128):
        per_token_flops = 0
        per_token_flops += 2 * self.dim * 3 * self.dim  # attn_qkv
        per_token_flops += 2 * seq_len * self.dim  # softmax attention
        per_token_flops += 2 * self.dim * self.dim  # attn_out
        # MoE flops (approximate - depends on routing)
        per_token_flops += 2 * self.top_k * 2 * self.dim * 4 * self.dim  # approximate MoE mlp
        per_token_flops += 2 * self.dim * self.num_experts  # gate
        flops = per_token_flops * seq_len
        flops += 2 * self.cond_dim * 6 * self.dim  # adaLN_modulation
        return flops

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, z, rotary_cos_sin, c, attention_mask=None, seqlens=None, c_cont=None):
        """
        Forward pass for dual-modality MoE DiT block.
        
        Args:
            x: First modality tensor (batch_size, seq_len, dim)
            z: Second modality tensor (batch_size, seq_len, dim)
            rotary_cos_sin: Rotary position embeddings (shared for both modalities)
            c: Condition tensor for z modality (batch_size, cond_dim)
            attention_mask: Optional attention mask (batch_size, 2*seq_len) for concatenated sequence
            seqlens: Optional sequence lengths
            c_cont: Condition tensor for x modality (batch_size, cond_dim)
            
        Returns:
            Tuple of (x_out, z_out) after processing
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x,
         scale_mlp_x, gate_mlp_x) = self.adaLN_modulation_x(c)[:, None].chunk(6, dim=2)
        (shift_msa_z, scale_msa_z, gate_msa_z, shift_mlp_z,
         scale_mlp_z, gate_mlp_z) = self.adaLN_modulation_z(c_cont)[:, None].chunk(6, dim=2) if not self.share_adaln else \
        (shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x)

        # Attention operation on concatenated sequence
        x_skip, z_skip = x, z
        x_modulated = modulate_fused(self.norm1(x), shift_msa_x, scale_msa_x)
        z_modulated = modulate_fused(self.norm1(z), shift_msa_z, scale_msa_z)
        combined = torch.cat([x_modulated, z_modulated], dim=1)
        combined_seq_len = combined.shape[1]

        qkv = self.attn_qkv(combined)
        qkv = rearrange(qkv,
                        'b s (three h d) -> b s three h d',
                        three=3,
                        h=self.n_heads)
        
        # Extend rotary embeddings for concatenated sequence
        cos, sin = rotary_cos_sin
        # Repeat rotary embeddings for both modalities
        cos_extended = torch.cat([cos, cos], dim=1)  # (1, 2*seq_len, 1, 1, dim)
        sin_extended = torch.cat([sin, sin], dim=1)  # (1, 2*seq_len, 1, 1, dim)
        
        qkv = apply_rotary_pos_emb(
            qkv, cos_extended.to(qkv.dtype), sin_extended.to(qkv.dtype))

        if has_flash_attn and attention_mask is None:
            # Use flash attention only when no attention mask is needed
            qkv = rearrange(qkv, 'b s ... -> (b s) ...')
            if seqlens is None:
                cu_seqlens = torch.arange(
                    0, (batch_size + 1) * combined_seq_len, step=combined_seq_len,
                    dtype=torch.int32, device=qkv.device)
            else:
                cu_seqlens = seqlens.cumsum(-1)
            combined = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, combined_seq_len, 0., causal=False)
            combined = rearrange(combined, '(b s) h d -> b s (h d)', b=batch_size)
        else:
            # Use standard attention with optional masking
            q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(
                1, 2), qkv[:, :, 2].transpose(1, 2)

            if attention_mask is not None:
                # extend attention mask to both modalities
                attention_mask = torch.cat([attention_mask, attention_mask], dim=1)
                # Convert attention_mask to the format expected by scaled_dot_product_attention
                # attention_mask should be (batch_size, 2*seq_len) with 1 for valid tokens, 0 for padded
                attn_mask = attention_mask.bool()  # (B, 2*S)
                attn_mask = attn_mask.unsqueeze(
                    1) & attn_mask.unsqueeze(2)  # (B, 2*S, 2*S)
                attn_mask = attn_mask.unsqueeze(1)  # (B, 1, 2*S, 2*S)
                float_mask = torch.zeros(
                    attn_mask.shape, device=q.device, dtype=q.dtype)
                float_mask = float_mask.masked_fill(~attn_mask, -1e9)

                combined = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=float_mask)
            else:
                combined = F.scaled_dot_product_attention(q, k, v)

            combined = rearrange(combined, 'b h s d -> b s (h d)', b=batch_size)

        x, z = combined[:, :seq_len, :], combined[:, seq_len:, :]
        x = bias_dropout_scale_fn(self.attn_out(x),
                                       None,
                                       gate_msa_x,
                                       x_skip,
                                       self.dropout)
        z = bias_dropout_scale_fn(self.attn_out(z),
                                       None,
                                       gate_msa_z,
                                       z_skip,
                                       self.dropout)

        # MoE operation on concatenated sequence
        mlp_input_x = modulate_fused(self.norm2(x), shift_mlp_x, scale_mlp_x)
        mlp_input_z = modulate_fused(self.norm2(z), shift_mlp_z, scale_mlp_z)
        mlp_output_x, router_logits_x = self.moe_forward(mlp_input_x)
        mlp_output_z, router_logits_z = self.moe_forward(mlp_input_z)
        
        mlp_output_x = bias_dropout_scale_fn(
            mlp_output_x, None, gate_mlp_x, x, self.dropout)
        mlp_output_z = bias_dropout_scale_fn(
            mlp_output_z, None, gate_mlp_z, z, self.dropout)
        
        return mlp_output_x, mlp_output_z


class MixtureDDiTBlockWithMask(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, n_single_modal_blocks=2, mlp_ratio=4, dropout=0.1, 
                 num_experts=4, num_experts_per_tok=2, norm_topk_prob=True, share_adaln=False):
        """
        Mixture DiT Block that combines multi-modal interaction with single-modal processing.

        Args:
            dim: Hidden dimension
            n_heads: Number of attention heads  
            cond_dim: Condition dimension
            n_single_modal_blocks: Number of single-modal blocks for each modality (n >= 0)
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
        """
        super().__init__()
        self.n_single_modal_blocks = n_single_modal_blocks

        # Multi-modal interaction block
        self.mm_block = MMDDiTBlockWithMask(
            dim=dim,
            n_heads=n_heads,
            cond_dim=cond_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
            share_adaln=share_adaln
        )

        # Single-modal blocks for x modality
        self.single_modal_blocks_x = nn.ModuleList([
            DDiTBlockWithMask(
                dim=dim,
                n_heads=n_heads,
                cond_dim=cond_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(n_single_modal_blocks)
        ])

        # Single-modal blocks for z modality (separate from x modality)
        self.single_modal_blocks_z = nn.ModuleList([
            DDiTBlockWithMask(
                dim=dim,
                n_heads=n_heads,
                cond_dim=cond_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(n_single_modal_blocks)
        ])

    def forward(self, x, z, rotary_cos_sin, c, attention_mask=None, seqlens=None, c_cont=None):
        """
        Forward pass through the mixture block.

        Args:
            x: First modality tensor (batch_size, seq_len, dim)
            z: Second modality tensor (batch_size, seq_len, dim)  
            rotary_cos_sin: Rotary position embeddings
            c: Condition tensor (batch_size, cond_dim)
            attention_mask: Optional attention mask (batch_size, seq_len)
            seqlens: Optional sequence lengths

        Returns:
            Tuple of (x_out, z_out) after processing
        """
        # Step 1: Multi-modal interaction
        x, z = self.mm_block(x, z, rotary_cos_sin, c, attention_mask, seqlens, c_cont)

        # Step 2: Single-modal processing for each modality separately
        # Process x modality through its dedicated blocks
        for block_x in self.single_modal_blocks_x:
            x = block_x(x, rotary_cos_sin, c, attention_mask, seqlens)

        # Process z modality through its dedicated blocks
        c_cont = c_cont if c_cont is not None else c
        for block_z in self.single_modal_blocks_z:
            z = block_z(z, rotary_cos_sin, c_cont, attention_mask, seqlens)

        return x, z

    def flops(self, seq_len=128):
        """Calculate FLOPs for the entire mixture block"""
        # FLOPs from multi-modal block (approximation - would need to implement in MMDDiTBlockWithMask)
        mm_flops = 0  # Would need to be implemented in MMDDiTBlockWithMask

        # FLOPs from single-modal blocks
        single_modal_flops = 0
        if self.n_single_modal_blocks > 0:
            block_flops = self.single_modal_blocks_x[0].flops(seq_len)
            single_modal_flops = 2 * self.n_single_modal_blocks * block_flops  # 2 modalities

        return mm_flops + single_modal_flops

class MoeDiT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)

        self.config = config
        self.vocab_size = vocab_size
        self.rounded_vocab_size = vocab_size + (128 - vocab_size % 128) % 128

        # Embedding layers for both modalities
        self.vocab_embed = EmbeddingLayer(
            config.model.hidden_size, self.rounded_vocab_size)
        self.latent_dim = config.model.get("latent_dim", config.model.hidden_size)
        self.latent_embed = DDitFinalLayer(self.latent_dim, config.model.hidden_size, config.model.cond_dim)

        # Shared components
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.rotary_emb = Rotary(
            config.model.hidden_size // config.model.n_heads,
            max_seq_len=config.model.max_seq_len,
        )

        # Mixture blocks with varying n_single_modal_blocks
        # but for convenience, we do not specify n_blocks; instead, we use n_single_list and makes sure #params is similar
        # assert len(config.model.n_single_list) == config.model.n_blocks, \
        #     f"n_single_list length ({len(config.model.n_single_list)}) must equal n_blocks ({config.model.n_blocks})"

        blocks = []
        for n_single_blocks in config.model.n_single_list:
            blocks.append(MixtureDDiTBlockWithMask(
                dim=config.model.hidden_size,
                n_heads=config.model.n_heads,
                cond_dim=config.model.cond_dim,
                n_single_modal_blocks=n_single_blocks,
                dropout=config.model.dropout,
                num_experts=config.model.get("num_experts", 4),
                num_experts_per_tok=config.model.get("num_experts_per_tok", 2),
                norm_topk_prob=config.model.get("norm_topk_prob", True),
                share_adaln=config.model.get("share_adaln", False)
            ))
        self.blocks = nn.ModuleList(blocks)

        # Separate output layers for each modality
        self.output_layer_x = DDitFinalLayer(
            config.model.hidden_size,
            self.rounded_vocab_size,
            config.model.cond_dim)

        self.output_layer_z = DDitFinalLayer(
            config.model.hidden_size,
            self.latent_dim,
            config.model.cond_dim)
        
        # Optional LayerNorm for continuous output normalization
        self.continuous_output_norm = LayerNorm(self.latent_dim) if config.model.get("use_continuous_output_norm", False) else nn.Identity()

        # Bias buffers for each modality
        self.register_buffer("logit_bias_x", torch.full((1, 1, 1), 0.0))

    def flops(self, seq_len=128):
        """Calculate total FLOPs for the model"""
        return sum(b.flops(seq_len) for b in self.blocks)

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, indices, z, sigma, attention_mask=None, c_cont=None):
        """
        Forward pass for multi-modal DiT.

        Args:
            indices: Token indices for discrete modality (batch_size, seq_len)
            z: Latent embeddings for continuous modality (batch_size, seq_len, hidden_size)  
            sigma: Diffusion timestep (batch_size,)
            attention_mask: Optional attention mask (batch_size, seq_len)

        Returns:
            Tuple of (logits_x, logits_z) for each modality
        """
        # Shared condition encoding
        c = F.silu(self.sigma_map(sigma))
        c_cont = F.silu(self.sigma_map(c_cont)) if c_cont is not None else c

        # Embed both modalities
        x = self.vocab_embed(indices)
        z = self.latent_embed(z, c_cont)

        # Shared rotary embeddings
        # Can use either x or z since same seq_len
        rotary_cos_sin = self.rotary_emb(x)

        # Process through mixture blocks
        for i in range(len(self.blocks)):
            x, z = self.blocks[i](
                x, z, rotary_cos_sin, c, attention_mask, seqlens=None, c_cont=c_cont)

        # Separate final layers for each modality
        logits_x = self.output_layer_x(x, c)
        z = self.output_layer_z(z, c_cont)
        z = self.continuous_output_norm(z)

        # Add bias (scatter_add for numerical stability)
        logits_x = logits_x.scatter_add(
            -1, indices.unsqueeze(-1), self.logit_bias_x.to(logits_x.dtype).expand_as(logits_x))

        return logits_x, z

    def forward_single_modality(self, indices, sigma, modality='x', attention_mask=None):
        """
        Forward pass for single modality (useful for inference or ablation studies).

        Args:
            indices: Token indices (batch_size, seq_len)
            sigma: Diffusion timestep (batch_size,)
            modality: 'x' or 'z' to specify which modality to process
            attention_mask: Optional attention mask (batch_size, seq_len)

        Returns:
            Logits for the specified modality
        """
        if modality == 'x':
            # Process only x modality, use zeros for z
            x = self.vocab_embed(indices)
            z = torch.zeros_like(x)

            c = F.silu(self.sigma_map(sigma))
            rotary_cos_sin = self.rotary_emb(x)

            for i in range(len(self.blocks)):
                x, z = self.blocks[i](
                    x, z, rotary_cos_sin, c, attention_mask, seqlens=None)
            z = self.output_layer_z(z, c)
            z = self.continuous_output_norm(z)

            logits = self.output_layer_x(x, c)
            logits = logits.scatter_add(-1, indices.unsqueeze(-1),
                                        self.logit_bias_x.to(logits.dtype).expand_as(logits))
            return logits, z

        elif modality == 'z':
            # Process only z modality, use zeros for x
            x = torch.zeros_like(z)

            c = F.silu(self.sigma_map(sigma))
            rotary_cos_sin = self.rotary_emb(z)

            for i in range(len(self.blocks)):
                x, z = self.blocks[i](
                    x, z, rotary_cos_sin, c, attention_mask, seqlens=None)

            logits = self.output_layer_x(x, c)
            z = self.output_layer_z(z, c)
            z = self.continuous_output_norm(z)
            logits = logits.scatter_add(-1, indices.unsqueeze(-1),
                                        self.logit_bias_x.to(logits.dtype).expand_as(logits))
            return logits, z
        else:
            raise ValueError(
                f"Invalid modality: {modality}. Must be 'x' or 'z'.")