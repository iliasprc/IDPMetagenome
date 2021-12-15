import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops import repeat
from torch import nn

from models.utils import weights_init


class TextTokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 embedding_dim=300,
                 n_output_channels=128,
                 activation=None,
                 *args, **kwargs):
        super(TextTokenizer, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, n_output_channels,
                      kernel_size=(kernel_size, embedding_dim),
                      stride=(stride, 1),
                      padding=(padding, 0), bias=False), nn.ReLU())

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        # print(x.shape)
        x = self.conv_layers(x)
        x = x.transpose(1, 3).squeeze(1)

        return x


def compute_mhsa(q, k, v, scale_factor=1, mask=None):
    # resulted shape will be: [batch, heads, tokens, tokens]
    scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * scale_factor

    if mask is not None:
        assert mask.shape == scaled_dot_prod.shape[2:]
        scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

    attention = torch.softmax(scaled_dot_prod, dim=-1)
    # calc result per head
    return torch.einsum('... i j , ... j d -> ... i d', attention, v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        """
        Implementation of multi-head attention layer of the original dnn model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]

        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be: [3, batch, heads, tokens, dim_head]
        q, k, v = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))

        out = compute_mhsa(q, k, v, mask=mask, scale_factor=self.scale_factor)

        # re-compose: merge heads with dim_head
        out = rearrange(out, "b h t d -> b t (h d)")
        # Apply final linear transformation layer
        return self.W_0(out)


class SelfAttention(nn.Module):
    """
    Implementation of plain self attention mechanism with einsum operations
    Paper: https://arxiv.org/abs/1706.03762
    Blog: https://theaisummer.com/transformer/
    """

    def __init__(self, dim):
        """
        Args:
            dim: for NLP it is the dimension of the embedding vector
            the last dimension size that will be provided in forward(x)
            where x is a 3D tensor
        """
        super().__init__()
        self.to_qvk = nn.Linear(dim, dim * 3, bias=False)
        self.scale_factor = dim ** -0.5  # 1/np.sqrt(dim)

    def forward(self, x, mask=None):
        assert x.dim() == 3, '3D tensor must be provided'
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3 ]

        # decomposition to q,v,k
        # rearrange tensor to [3, batch, tokens, dim] and cast to tuple
        q, k, v = tuple(rearrange(qkv, 'b t (d k) -> k b t d ', k=3))

        # Resulting shape: [batch, tokens, tokens]
        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)
        return torch.einsum('b i j , b j d -> b i d', attention, v)


class TransformerBlock(nn.Module):
    """
    Vanilla dnn block from the original paper "Attention is all you need"
    Detailed analysis: https://theaisummer.com/transformer/
    """

    def __init__(self, dim, heads=8, dim_head=None,
                 dim_linear_block=1024, dropout=0.1, activation=nn.GELU,
                 mhsa=None, prenorm=False):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
            mhsa: if provided you can change the vanilla self-attention block
            prenorm: if the layer norm will be applied before the mhsa or after
        """
        super().__init__()
        self.mhsa = mhsa if mhsa is not None else MultiHeadSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.prenorm = prenorm
        self.drop = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),  # nn.ReLU or nn.GELU
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        if self.prenorm:
            y = self.drop(self.mhsa(self.norm_1(x), mask)) + x
            out = self.linear(self.norm_2(y)) + y
        else:
            y = self.norm_1(self.drop(self.mhsa(x, mask)) + x)
            out = self.norm_2(self.linear(y) + y)
        return out


def expand_to_batch(tensor, desired_size):
    tile = desired_size // tensor.shape[0]
    return repeat(tensor, 'b ... -> (b tile) ...', tile=tile)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


class PositionalEncodingSin(nn.Module):

    def __init__(self, dim, dropout=0.1, max_tokens=5000):
        super(PositionalEncodingSin, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(1, max_tokens, dim)
        position = torch.arange(0, max_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.Tensor([10000.0])) / dim))
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = nn.Parameter(pe)
        self.pe.requires_grad = False

    def forward(self, x):
        batch, seq_tokens, _ = x.size()
        x = x + expand_to_batch(self.pe[:, :seq_tokens, :], desired_size=batch)
        return self.dropout(x)


class IDP_cct(nn.Module):

    def __init__(self, dim, blocks=6, heads=8, dim_head=None, dim_linear_block=1024, dropout=0, prenorm=False,
                 classes=1):
        super().__init__()
        self.embed = nn.Embedding(classes, dim)
        self.conv = TextTokenizer(word_embedding_dim=dim, embedding_dim=dim, n_output_channels=dim, kernel_size=5,
                                  stride=1,
                                  padding=2)
        self.pos_embed = PositionalEncodingSin(dim, dropout=0.1, max_tokens=2500)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=dim, nhead=heads,
                                    dim_feedforward=dim_linear_block, dropout=dropout,
                                    attention_dropout=dropout, drop_path_rate=0.1)
            for i in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, classes))

        self.apply(weights_init)

    def forward(self, x, mask=None):
        # print(x.shape)
        # assert len(x.shape) == 3
        x = self.conv(self.embed(x))
        # print(x.shape)

        x = self.pos_embed(x)  # self.embed(x))
        for layer in self.layers:
            x = layer(x)
        x = self.head(x).squeeze(-1)
        return x


class IDPseqvec(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 256
        self.embed = nn.Sequential(nn.Linear(1024, dim), nn.ReLU(), nn.Linear(dim, 2))

    def forward(self, x, mask=None):
        x = self.embed(x)

        return x


class IDPTransformer(nn.Module):
    def __init__(self, config, dim, blocks=6, heads=8, dim_head=None, dim_linear_block=1024, dropout=0, prenorm=False,
                 embed_dim=30,
                 classes=1):
        super().__init__()
        self.embed = nn.Embedding(embed_dim, dim)
        self.use_elmo = True

        self.pos_embed = PositionalEncodingSin(dim, dropout=0.1, max_tokens=2000)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, dim_feedforward=dim_linear_block, nhead=heads,
                                                   activation='gelu', dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=blocks)

        self.head = nn.Linear(dim, classes)
        self.apply(weights_init)

    def forward(self, x, mask=None):
        # print(x.shape)
        # assert len(x.shape) == 3
        # print(x.shape)
        x = self.embed(x)
        # print(x.shape)

        # x = rearrange(x,'b t c -> b c t')
        # x = self.tcn(x)
        # x = rearrange(x, ' b c t -> b t c')
        x = self.pos_embed(x)  # self.embed(x))
        x = self.transformer_encoder(x)
        x = self.head(x).squeeze(-1)
        return x

#
# m = IDPTransformer(768)
# print(m)
# i = torch.randn(1,1000,768)
# o =  m(i)
# print(o.shape)
# m = IDP_cct(128)
# dim = 128
# # m = TextTokenizer(word_embedding_dim=dim,embedding_dim=dim,kernel_size=1,stride=1,
# #                                                                         padding=1)
# print(m )
# a = torch.randint(0,19,(8,1000))
# o = m(a)
