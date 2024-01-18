import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional


class SpikingSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SpikingSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, f"dim {embed_dim} should be divided by num_heads {num_heads}."
        self.embed_dim = embed_dim
        self.heads = num_heads

        self.proj_plif = neuron.ParametricLIFNode(backend='cupy')
        self.q_conv = layer.Conv1d(self.embed_dim, self.embed_dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = layer.BatchNorm1d(self.embed_dim)
        self.q_plif = neuron.ParametricLIFNode(backend='cupy')

        self.k_conv = layer.Conv1d(self.embed_dim, self.embed_dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = layer.BatchNorm1d(self.embed_dim)
        self.k_plif = neuron.ParametricLIFNode(backend='cupy')

        self.v_conv = layer.Conv1d(self.embed_dim, self.embed_dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = layer.BatchNorm1d(self.embed_dim)
        self.v_plif = neuron.ParametricLIFNode(backend='cupy')

        self.attn_plif = neuron.ParametricLIFNode(backend='cupy')
        self.proj_conv = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(self.embed_dim)

    def forward(self, q, k, v, key_padding_mask=None):
        T, B, C, H_q, W_q = q.shape
        H_k, W_k = k.shape[-2:]
        H_v, W_v = v.shape[-2:]

        q_out, k_out, v_out = self.proj_plif(q), self.proj_plif(k), self.proj_plif(v)
        q_out, k_out, v_out = q_out.flatten(3), k_out.flatten(3), v_out.flatten(3)

        q_out = self.q_plif(self.q_bn(self.q_conv(q_out))).transpose(-1, -2).reshape(
            T, B, H_q * W_q, self.heads, C // self.heads)
        k_out = self.k_plif(self.k_bn(self.k_conv(k_out))).transpose(-1, -2).reshape(
            T, B, H_k * W_k, self.heads, C // self.heads)
        v_out = self.v_plif(self.v_bn(self.v_conv(v_out))).transpose(-1, -2).reshape(
            T, B, H_v * W_v, self.heads, C // self.heads)

        # attn: [T, B, HEAD, q_len, k_len]
        attn = torch.einsum("tbqhd, tbkhd -> tbhqk", [q_out, k_out]) * 0.125

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.reshape(T, B, 1, 1, H_k * W_k).repeat(1, 1, self.heads, 1, 1)
            attn = attn.masked_fill(key_padding_mask, float('-inf'))

        attn_out = torch.einsum("tbhqk, tbkhd -> tbqhd", [attn, v_out])

        attn_out = attn_out.reshape(T, B, H_q * W_q, C).transpose(-1, -2)
        attn_out = self.attn_plif(self.proj_bn(self.proj_conv(attn_out))).reshape(T, B, C, H_q, W_q)
        return attn_out


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion):
        super(EncoderLayer, self).__init__()
        self.attention = SpikingSelfAttention(embed_dim, num_heads)
        self.mlp1 = nn.Sequential(
            neuron.ParametricLIFNode(backend='cupy'),
            layer.Conv1d(embed_dim, embed_dim * forward_expansion, kernel_size=1, stride=1),
            layer.BatchNorm1d(embed_dim * forward_expansion)
        )
        self.mlp2 = nn.Sequential(
            neuron.ParametricLIFNode(backend='cupy'),
            layer.Conv1d(embed_dim * forward_expansion, embed_dim, kernel_size=1, stride=1),
            layer.BatchNorm1d(embed_dim)
        )
        self.mlp = nn.Sequential(
            self.mlp1,
            self.mlp2
        )

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, src_key_padding_mask):
        q = k = self.with_pos_embed(src, pos)
        out1 = self.attention(q, k, v=src, mask=src_key_padding_mask)
        out1 = out1 + q
        out2 = self.mlp(out1)
        out2 = out2 + out1
        return out2


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, num_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(embed_dim, num_heads, forward_expansion) for _ in range(num_layers)
            ]
        )

    def forward(self, src, pos, src_key_padding_mask):
        out = src
        for layer in self.layers:
            out = layer(out, pos, src_key_padding_mask)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion):
        super(DecoderLayer, self).__init__()
        self.self_attn = SpikingSelfAttention(embed_dim, num_heads)
        self.multihead_attn = SpikingSelfAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            neuron.ParametricLIFNode(backend='cupy'),
            layer.Conv1d(embed_dim, embed_dim * forward_expansion, kernel_size=1, stride=1),
            layer.BatchNorm1d(embed_dim * forward_expansion),
            neuron.ParametricLIFNode(backend='cupy'),
            layer.Conv1d(embed_dim * forward_expansion, embed_dim, kernel_size=1, stride=1),
            layer.BatchNorm1d(embed_dim)
        )

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos, query_embed, memory_key_padding_mask):
        q = k = self.with_pos_embed(tgt, query_embed)
        tgt1 = self.self_attn(q=q, k=k, v=tgt, key_padding_mask=None)
        tgt = tgt + tgt1
        tgt2 = self.multihead_attn(
            q=self.with_pos_embed(tgt, query_embed),
            k=self.with_pos_embed(memory, pos),
            v=memory,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + tgt2
        tgt2 = self.mlp(tgt)
        tgt = tgt + tgt2
        return tgt


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, num_layers, return_imtermediate):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(embed_dim, num_heads, forward_expansion) for _ in range(num_layers)
            ]
        )
        self.return_intermediate = return_imtermediate

    def forward(self, tgt, memory, pos, query_embed, memory_key_padding_mask):
        out = tgt
        intermediate = []
        for layer in self.layers:
            out = layer(out, memory, pos, query_embed, memory_key_padding_mask)
            if self.return_intermediate:
                intermediate.append(out)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return out.unsqueeze(0)


class SpikingTransformer(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            num_heads=8,
            forward_expansion=4,
            num_encoder_layers=6,
            num_decoder_layers=6,
            return_intermediate=False
    ):
        super(SpikingTransformer, self).__init__()
        self.encoder = Encoder(embed_dim, num_heads, forward_expansion, num_encoder_layers)
        self.decoder = Decoder(embed_dim, num_heads, forward_expansion, num_decoder_layers, return_intermediate)
        self._reset_parameters()
        self.embed_dim = embed_dim
        self.heads = num_heads

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src=src, pos=pos_embed, src_key_padding_mask=mask)
        hs = self.decoder(tgt=tgt, memory=memory, pos=pos_embed, query_embed=query_embed, memory_key_padding_mask=mask)
        return hs, memory


def build_transformer(args):
    return SpikingTransformer(
        embed_dim=args.hidden_dim,
        num_heads=args.nheads,
        forward_expansion=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        # 返回decoder的中间层结果
        return_intermediate=True,
    )
