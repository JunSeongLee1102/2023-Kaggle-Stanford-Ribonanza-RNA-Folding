import admin_torch as admin
import torch
import torch.nn.functional as F
import xformers.ops as xops
from rotary_embedding_torch import RotaryEmbedding as Rope
from torch import Tensor, nn
from x_transformers.x_transformers import RelativePositionBias as RelPB
from apex.normalization import FusedLayerNorm, FusedRMSNorm


def compile(layers):
    return torch.compile(
        layers,
        dynamic=True,
        options={"shape_padding": True},
        disable=torch.cuda.device_count() > 1,
    )


class ProjectOut(nn.Module):
    def __init__(
        self,
        p_dropout: float,
        d_model: int,
        aux_loop: bool,
        aux_struct: bool,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p_dropout)
        d = {"react": nn.Linear(d_model, 2)}
        d = d | ({"loop": nn.Linear(d_model, 7)} if aux_loop else {})
        d = d | ({"struct": nn.Linear(d_model, 3)} if aux_struct else {})
        self.linears = nn.ModuleDict(d)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        out = {k: v(x) for k, v in self.linears.items()}
        return out


class RNA_Model(nn.Module):
    def __init__(
        self,
        pos_rope: bool,
        pos_bias_heads: int,
        pos_bias_params: tuple[int, int],
        norm_rms: str,
        norm_lax: bool,
        emb_grad_frac: float,
        n_mem: int,
        aux_loop: str | None,
        aux_struct: str | None,
        **kwargs,
    ):
        super().__init__()
        global Norm
        Norm = get_layer_norm(FusedRMSNorm if norm_rms else FusedLayerNorm, norm_lax)
        d_model = (d_heads := kwargs["d_heads"]) * (n_heads := kwargs["n_heads"])
        p_dropout, n_layers = kwargs["p_dropout"], kwargs["n_layers"]
        layers = [EncoderLayer(**kwargs) for i in range(n_layers)]
        self.layers = nn.ModuleList(layers)
        self.mem = nn.Parameter(torch.randn(1, n_mem, d_model)) if n_mem else None
        self.emb = nn.Embedding(5, d_model, 0)
        self.rope = Rope(d_heads, seq_before_head_dim=True) if pos_rope else None
        self.pos_bias = None
        if pos_bias_heads:
            assert (heads := pos_bias_heads) <= n_heads
            self.pos_bias = compile(
                RelPB(d_heads**0.5, False, *pos_bias_params, heads)
            )
        self.out = compile(
            ProjectOut(p_dropout, d_model, aux_loop != None, aux_struct != None)
        )
        self.res = kwargs["norm_layout"] == "dual"
        self.emb_grad = emb_grad_frac

    def forward(self, x0: Tensor) -> Tensor:
        seq = x0["seq"]
        mask = x0["mask"]
        bpps = x0["As"]
        x = self.emb(seq)
        b = bpps[:, :, :, 0]
        if self.mem != None:
            mask = F.pad(mask, (self.mem.size(1), 0))
            x = torch.cat([self.mem.expand(x.size(0), -1, -1), x], 1)
        if 1 > self.emb_grad > 0:
            x = x * self.emb_grad + x.detach() * (1 - self.emb_grad)
        res = x * self.layers[0].res_scale if self.res else None
        for f in self.layers:
            if self.res:
                x, res = f(x, b, res, mask, self.rope, self.pos_bias)
            else:
                x = f(x, mask, self.rope, self.pos_bias)
        x = x[:, self.mem.size(1) :] if self.mem != None else x
        return self.out(x)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_heads: int,
        n_heads: int,
        n_layers: int,
        p_dropout: float,
        ffn_multi: int,
        ffn_bias: bool,
        qkv_bias: bool,
        att_fn: str,
        norm_layout: str,
        **kwargs,
    ):
        super().__init__()
        d_ffn = (d_model := d_heads * n_heads) * ffn_multi
        self.att = Multi_head_Attention(d_model, n_heads, p_dropout, qkv_bias, att_fn)
        self.ffn = compile(
            nn.Sequential(
                nn.Linear(d_model, d_ffn, ffn_bias),
                nn.GELU(),
                nn.Dropout(p_dropout),
                nn.Linear(d_ffn, d_model, ffn_bias),
                nn.Dropout(p_dropout),
            )
        )
        if norm_layout == "dual":
            self.forward = self.forward_dual
            self.norm = nn.ModuleList([Norm(d_model) for _ in range(4)])
            self.res_scale = 0.1
        elif norm_layout == "sandwich":
            self.forward = self.forward_sandwich
            self.norm = nn.ModuleList([Norm(d_model) for _ in range(4)])
        else:
            self.forward = self.forward_post
            self.norm = nn.ModuleList([Norm(d_model) for _ in range(2)])
            self.res = nn.ModuleList([admin.as_module(n_layers * 2) for _ in range(2)])
        self.bpp_att = ResidualBPPAttention(
            d_model, kernel_size=kwargs["kernel_size_gc"], dropout=p_dropout
        )
        # self.mbconv = FusedMBConv(d_model, p_dropout)
        self.rnns = nn.ModuleList()
        for i in range(0, kwargs["n_heads_rnn"]):
            if i % 2 == 1:
                self.rnns.append(
                    compile(
                        nn.LSTM(
                            d_model,
                            d_model // 2 // kwargs["n_heads_rnn"],
                            kwargs["n_layers_rnn"],
                            batch_first=True,
                            bidirectional=True,
                            dropout=p_dropout,
                        )
                    )
                )
            else:
                self.rnns.append(
                    compile(
                        nn.GRU(
                            d_model,
                            d_model // 2 // kwargs["n_heads_rnn"],
                            kwargs["n_layers_rnn"],
                            batch_first=True,
                            bidirectional=True,
                            dropout=p_dropout,
                        )
                    )
                )
        # self.block1 = SEResidual(d_model, kernel_size= kwargs["kernel_size_gc"], dropout=p_dropout)
        self.atts = compile(self.forward_attentions)
        self.forward = self.forward_dual

    def forward_attentions(self, x: Tensor, b: Tensor):
        # x = self.mbconv(x)
        x = self.bpp_att(x, b)
        return x

    def forward_post(self, x: Tensor, *args, **kwargs):
        x = self.norm[0](self.res[0](x, self.att(x, *args, **kwargs)))
        x = self.norm[1](self.res[1](x, self.ffn(x)))
        return x

    def forward_sandwich(self, x: Tensor, *args, **kwargs):
        x = x + self.norm[1](self.att(self.norm[0](x), *args, **kwargs))
        x = x + self.norm[3](self.ffn(self.norm[2](x)))
        return x

    def forward_dual(self, x: Tensor, b: Tensor, res: Tensor, *args, **kwargs):
        x_att = self.att(x, *args, **kwargs)
        res = res + x_att * self.res_scale
        x = self.norm[0](x + x_att) + self.norm[1](res)
        x_ffn = self.ffn(x)
        res = res + x_ffn * self.res_scale
        x = self.norm[2](x + x_ffn) + self.norm[3](res)
        x = self.atts(x, b)
        xs = []
        for i in range(0, len(self.rnns)):
            x1, _ = self.rnns[i](x)
            xs.append(x1)
        x = torch.cat(xs, dim=-1)
        return x, res


class ResidualBPPAttention(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.conv1 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.conv2 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.relu = nn.ReLU()

    def forward(self, src, attn):
        src = src.permute([0, 2, 1])
        h = self.conv2(self.conv1(torch.bmm(src, attn)))
        return self.relu(src + h).permute([0, 2, 1])


class Conv(nn.Module):
    def __init__(self, d_in: int, d_out: int, kernel_size: int, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            d_in, d_out, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(d_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        return self.dropout(self.relu(self.bn(self.conv(src))))


class FusedMBConv(nn.Module):
    class SqueezeExcitation(nn.Module):
        def __init__(self, n_channels: int, reduction: int = 16):
            super().__init__()
            self.layers = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(n_channels, n_channels // reduction, 1),
                nn.GELU(),
                nn.Conv1d(n_channels // reduction, n_channels, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.layers(x) * x

    def __init__(self, d_model: int, p_dropout: float, n_groups: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 3, 1, 1, groups=n_groups),
            nn.BatchNorm1d(d_model * 2),
            nn.GELU(),
            self.SqueezeExcitation(d_model * 2),
            nn.Conv1d(d_model * 2, d_model, 1, groups=n_groups),
            nn.BatchNorm1d(d_model),
            nn.Dropout(p_dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(x.transpose(1, 2)).transpose(1, 2)


class Multi_head_Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        p_dropout: float,
        qkv_bias: bool,
        att_fn: str,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_heads = d_model // n_heads
        self.p_dropout = p_dropout
        self.qkv = nn.Linear(d_model, d_model * 3, qkv_bias)
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(p_dropout),
        )
        self.att = self.xmea if att_fn == "xmea" else self.sdpa

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None,
        rope: nn.Module | None,
        pos_bias: nn.Module | None,
    ) -> Tensor:
        B, L, N, D = (*x.shape[:2], self.n_heads, self.d_heads)
        qkv = [_.view(B, L, N, D) for _ in self.qkv(x).chunk(3, -1)]
        if rope != None:
            qkv[0] = rope.rotate_queries_or_keys(qkv[0])
            qkv[1] = rope.rotate_queries_or_keys(qkv[1])
        bias = None
        if pos_bias != None:
            bias = pos_bias(L, L).to(qkv[0].dtype)
            bias = bias.unsqueeze(0).expand(B, -1, -1, -1)
            if N > bias.size(1):
                bias = F.pad(bias, (*[0] * 5, N - bias.size(1)))
        if mask != None:
            mask = self.mask_to_bias(mask, qkv[0].dtype)
            bias = mask if bias == None else bias + mask
        x = self.att(qkv, bias.contiguous()).reshape(B, L, N * D)
        return self.out(x)

    def sdpa(self, qkv: tuple[Tensor, Tensor, Tensor], bias: Tensor | None) -> Tensor:
        p_drop = self.p_dropout if self.training else 0
        qkv = [_.transpose(1, 2) for _ in qkv]
        x = F.scaled_dot_product_attention(*qkv, bias, p_drop)
        return x.transpose(1, 2)

    def xmea(self, qkv: tuple[Tensor, Tensor, Tensor], bias: Tensor | None) -> Tensor:
        p_drop = self.p_dropout if self.training else 0
        if bias != None and (L := qkv[0].size(1)) % 8:
            pad = -(L // -8) * 8 - L
            bias = F.pad(bias, (0, pad, 0, pad)).contiguous()[..., :L, :L]
        x = xops.memory_efficient_attention(*qkv, bias, p_drop)
        return x

    def mask_to_bias(self, mask: Tensor, float_dtype: torch.dtype) -> Tensor:
        mask = mask.to(float_dtype) - 1
        mask[mask < 0] = float("-inf")
        mask = mask.view(mask.size(0), 1, 1, mask.size(-1))
        mask = mask.expand(-1, self.n_heads, mask.size(-1), -1)
        return mask


def get_layer_norm(cls: nn.Module, norm_lax: bool) -> nn.Module:
    class LayerNorm(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.forward = super().forward
            if norm_lax:
                self.lax = torch.compile(lambda x: x / x.max())
                self.forward = self.forward_lax

        def forward_lax(self, x):
            return super().forward(self.lax(x))

    return LayerNorm


def get_bpp_feature(bpp):
    bpp_nb_mean = 0.10559  # mean of bpps_nb across all training data
    bpp_nb_std = 0.096033  # std of bpps_nb across all training data
    bpp_max = bpp.max(-1)[0]
    bpp_sum = bpp.sum(-1)
    bpp_nb = torch.true_divide((bpp > 0).sum(dim=1), bpp.shape[1])
    bpp_nb = torch.true_divide(bpp_nb - bpp_nb_mean, bpp_nb_std)
    return [bpp_max.unsqueeze(2), bpp_sum.unsqueeze(2), bpp_nb.unsqueeze(2)]


class SEResidual(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.conv1 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.conv2 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.relu = nn.ReLU()
        self.se = SELayer(d_model)

    def forward(self, src):
        h = self.conv2(self.conv1(src))
        return self.se(self.relu(src + h))


class SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
