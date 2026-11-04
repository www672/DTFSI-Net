import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange

from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal
from layers.SelfAttention_Family import FullAttention, AttentionLayer

from transformers import BertConfig, BertModel, BertTokenizer
import transformers


class TaTSMLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


class AdaptiveDynamicHeadsCMA(nn.Module):
    def __init__(self, num_heads, num_nodes, channel, device):
        super().__init__()
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        self.channel = channel
        self.device = device
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_heads * channel, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_heads)
        ).to(device)

    def forward(self, cma_outputs):
        B, C, N = cma_outputs[0].shape
        H = self.num_heads
        combined = torch.cat(cma_outputs, dim=1)
        combined_permute = combined.permute(0, 2, 1)
        gates = self.gate_mlp(combined_permute)
        gates = F.softmax(gates, dim=-1)
        combined_heads = combined.view(B, H, C, N).permute(0, 1, 3, 2)
        gates = gates.permute(0, 2, 1).unsqueeze(-1)
        weighted_heads = combined_heads * gates
        weighted_heads = weighted_heads.permute(0, 1, 3, 2)
        fused = weighted_heads.sum(dim=1)
        return fused


class FrequencyAttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.freq_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, freq_enc_out):
        attn_logits = self.freq_attention(freq_enc_out)
        attn_weights = F.softmax(attn_logits, dim=1)
        pooled_freq = (freq_enc_out * attn_weights).sum(dim=1)
        return pooled_freq


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        index = index.to(self.data.device)
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=768, attention_dropout=0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        scale = 1. / math.sqrt(target_embedding.shape[-1])
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        attn = torch.softmax(scale * scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhls,she->blhe", attn, value_embedding)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_block, d_ff=None, dropout=0.1, gate=1):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv4 = nn.Conv1d(d_model, d_model, kernel_size=1)
        d_ff_out = d_model * 2 if attention else d_model
        self.conv5 = nn.Conv1d(d_ff_out, d_block, kernel_size=1)
        self.conv6 = nn.Conv1d(d_ff_out, d_block, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.gelu if attention else F.relu
        self.gate = gate

    def forward(self, x, attn_mask=None):
        if self.attention:
            x_att, _ = self.attention(x, x, x, attn_mask=attn_mask)
            x = x - self.dropout(x_att)

        x_ln = x = self.norm1(x)
        x_ln = self.dropout(self.act(self.conv1(x_ln.transpose(-1, 1))))
        x_ln = self.dropout(self.conv2(x_ln).transpose(-1, 1))
        
        x = (x - x_ln).transpose(-1, 1)
        h_gate = F.sigmoid(self.conv3(x)) if self.gate else 1
        h = h_gate * self.conv4(x)

        out = torch.cat((x_att, x_ln), -1) if self.attention else x_ln
        out = out.transpose(-1, 1)
        gate = F.sigmoid(self.conv5(out)) if self.gate else 1
        out = gate * self.conv6(out)

        return self.norm2(h.transpose(-1, 1)), out.transpose(-1, 1)


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None

    def forward(self, x, attn_mask=None):
        output = 0
        for attn_layer in self.attn_layers:
            x, out = attn_layer(x, attn_mask=attn_mask)
            output = out - output
        return output


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


def _resolve_dataset_config(configs):
    """
    [自适应] 根据数据集名称 / configs 自动推断周期长度和开关。
    支持覆盖：configs.cycle_len / configs.use_freq / configs.ma_kernel 优先级最高。

    返回 dict:
        cycle_len : int      — 周期长度（用于 RecurrentCycle）
        ma_kernel : int      — moving_avg 卷积核（奇数，≈ 周期长度）
        use_freq  : bool     — 是否启用频率分支（Exchange 等无周期数据设为 False）
        use_cycle : bool     — 是否启用周期分解
    """
    # 默认值
    cycle_len = getattr(configs, 'cycle_len', None)
    use_freq = getattr(configs, 'use_freq', None)
    ma_kernel = getattr(configs, 'ma_kernel', None)
    use_cycle = getattr(configs, 'use_cycle', None)

    data_name = getattr(configs, 'data', '').lower()

    if cycle_len is None:
        if 'etth' in data_name or 'electricity' in data_name or 'traffic' in data_name:
            cycle_len = 24          # 1 小时采样，周期 = 一天
        elif 'ettm' in data_name:
            cycle_len = 96          # 15 分钟采样，周期 = 一天
        elif 'weather' in data_name:
            cycle_len = 144         # 10 分钟采样，周期 = 一天
        elif 'exchange' in data_name:
            cycle_len = 1           # 无周期
        else:
            cycle_len = 24          # 兜底

    if use_freq is None:
        # Exchange 无明显周期，频率分支易过拟合
        use_freq = ('exchange' not in data_name)

    if use_cycle is None:
        use_cycle = (cycle_len > 1)

    if ma_kernel is None:
        # 卷积核取周期长度
        k = max(3, cycle_len)
        if k % 2 == 0:
            k += 1
        ma_kernel = k

    return {
        'cycle_len': cycle_len,
        'ma_kernel': ma_kernel,
        'use_freq': use_freq,
        'use_cycle': use_cycle,
    }


class TriModal(nn.Module):

    def __init__(
            self,
            configs,
            device="cuda:0",
            channel=32,
            num_nodes=7,
            seq_len=96,
            pred_len=96,
            dropout_n=0.1,
            d_llm=768,
            text_embedding_dim=64,
            e_layer=1,
            d_layer=1,
            d_ff=32,
            head=8,

            use_cycle=True,
            use_trend=True,
            use_residual=True,
            use_ct_gating=True,
            use_ct_cma=True
    ):

        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.description = "multivariate time series forecasting"

        # ── 解析数据集配置 ──
        ds_cfg = _resolve_dataset_config(configs)
        self.cycle_len = ds_cfg['cycle_len']
        self.use_freq_branch = ds_cfg['use_freq']
        self.use_cycle_ds = ds_cfg['use_cycle']
        ma_kernel = ds_cfg['ma_kernel']

        # 用户传入的开关
        self.use_trend = use_trend
        self.use_residual = use_residual
        self.use_ct_gating = use_ct_gating
        self.use_ct_cma = use_ct_cma
        # 周期分解
        self.use_cycle = use_cycle and self.use_cycle_ds

        # ── Normalize ──
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(device)

        # ── Projection ──
        self.length_to_feature = nn.Linear(seq_len, channel).to(device)

        # ── 位置编码 ──
        self.position_encoding = nn.Parameter(
            torch.randn(1, num_nodes, channel) * 0.02
        ).to(device)

        # ── TaTS Text Projector ──
        self.text_mlp = TaTSMLP(
            [d_llm, d_llm // 8, text_embedding_dim],
            dropout_rate=0.3
        ).to(device)
        self.text_ts_proj = nn.Sequential(
            nn.Linear(text_embedding_dim, channel),
            nn.GELU(),
            nn.Linear(channel, channel)
        ).to(device)

        # ── TaTS Early Fusion ──
        self.tats_fusion = nn.Conv1d(channel * 2, channel, kernel_size=1).to(device)

        # Decomposition
        self.cycleQueue = RecurrentCycle(self.cycle_len, num_nodes).to(device)
        self.moving_avg = moving_avg(kernel_size=ma_kernel, stride=1).to(device)

        self.trend_proj = nn.Linear(seq_len, channel).to(device)
        if self.use_cycle:
            self.cycle_proj = nn.Linear(seq_len, channel).to(device)
        self.residual_proj = nn.Sequential(
            nn.Linear(seq_len, channel),
            nn.GELU(),
            nn.Linear(channel, channel)
        ).to(device)

        # ── 可学习的分解缩放系数 ──
        self.trend_scale = nn.Parameter(torch.tensor(0.1, device=device))
        self.residual_scale = nn.Parameter(torch.tensor(0.1, device=device))
        self.cycle_scale = nn.Parameter(torch.tensor(0.1, device=device))
        self.text_gate_scale = nn.Parameter(torch.tensor(0.0, device=device))

        # ── Encoder (双流残差) ──
        encoder_layers = []
        for i in range(e_layer):
            attn_layer = AttentionLayer(
                FullAttention(
                    mask_flag=False, factor=5, attention_dropout=dropout_n, output_attention=False
                ),
                d_model=channel, n_heads=head
            )
            encoder_layers.append(
                EncoderLayer(
                    attention=attn_layer,
                    d_model=channel,
                    d_block=channel,
                    d_ff=channel * 4,
                    dropout=dropout_n,
                    gate=1
                ).to(device)
            )
        self.ts_encoder = Encoder(encoder_layers).to(device)

        # 频率分支
        if self.use_freq_branch:
            self.freq_token_proj = nn.Linear(3, channel).to(device)
            freq_encoder_layers = []
            for i in range(2):
                freq_attn_layer = AttentionLayer(
                    FullAttention(
                        mask_flag=False, factor=5, attention_dropout=dropout_n, output_attention=False
                    ),
                    d_model=channel, n_heads=head
                )
                freq_encoder_layers.append(
                    EncoderLayer(
                        attention=freq_attn_layer,
                        d_model=channel,
                        d_block=channel,
                        d_ff=channel * 4,
                        dropout=dropout_n,
                        gate=1
                    ).to(device)
                )
            self.freq_encoder = Encoder(freq_encoder_layers).to(device)
            self.freq_pool = FrequencyAttentionPooling(channel).to(device)

            # 频率-文本对齐门
            self.freq_llm_gate = nn.Sequential(
                nn.Linear(channel, channel),
                nn.GELU(),
                nn.Linear(channel, channel),
                nn.Sigmoid()
            ).to(device)
            self.freq_gate_scale = nn.Parameter(torch.tensor(0.1, device=device))

            self.freq_global_gate = nn.Parameter(torch.zeros(1, device=device))

            # Gating Fusion
            self.gate_conv = nn.Conv1d(channel * 2, channel, kernel_size=1).to(device)
            self.gate_norm = nn.LayerNorm(channel * 2).to(device)

        # ── CMA ──
        self.num_cma_heads = 2
        self.cma_heads = nn.ModuleList([
            CrossModal(
                d_model=num_nodes, n_heads=1, d_ff=d_ff,
                norm='LayerNorm', attn_dropout=dropout_n,
                dropout=dropout_n, pre_norm=True
            ).to(device)
            for _ in range(self.num_cma_heads)
        ])
        self.adaptive_dynamic_heads_cma = AdaptiveDynamicHeadsCMA(
            self.num_cma_heads, num_nodes, channel, device
        )

        # ── Residual Fusion ──
        self.residual_alpha = nn.Parameter(torch.ones(channel, device=device) * 0.1)

        # ── Decoder (双流残差) ──
        decoder_layers = []
        for i in range(d_layer):
            self_attn_layer = AttentionLayer(
                FullAttention(
                    mask_flag=False, factor=5, attention_dropout=dropout_n, output_attention=False
                ),
                d_model=channel, n_heads=head
            )
            cross_attn_layer = AttentionLayer(
                FullAttention(
                    mask_flag=False, factor=5, attention_dropout=dropout_n, output_attention=False
                ),
                d_model=channel, n_heads=head
            )
            decoder_layers.append(
                DecoderLayer(
                    self_attention=self_attn_layer,
                    cross_attention=cross_attn_layer,
                    d_model=channel,
                    d_ff=channel * 4,
                    dropout=dropout_n,
                    activation="gelu"
                ).to(device)
            )
        self.decoder = Decoder(decoder_layers).to(device)

        # ── 输出头 ──
        self.c_to_length = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, channel * 2),
            nn.GELU(),
            nn.Dropout(dropout_n),
            nn.Linear(channel * 2, pred_len)
        ).to(device)

        self.input_skip_proj = nn.Linear(seq_len, pred_len).to(device)
        self.skip_gate = nn.Parameter(torch.tensor(-1.0, device=device))

        self.skip_early_proj = nn.Linear(seq_len, channel).to(device)
        self.skip_early_gate = nn.Parameter(torch.tensor(-2.0, device=device))

        self.skip_enc_proj = nn.Linear(seq_len, channel).to(device)
        self.skip_enc_gate = nn.Parameter(torch.tensor(-2.0, device=device))

        self.skip_cma_proj = nn.Linear(seq_len, channel).to(device)
        self.skip_cma_gate = nn.Parameter(torch.tensor(-2.0, device=device))

        # ── LLM 分支 ──
        if configs.llm_model == 'BERT':
            self.bert_config = BertConfig(
                hidden_size=768,
                num_hidden_layers=configs.llm_layers,
                num_attention_heads=12,
                intermediate_size=3072,
                output_attentions=True,
                output_hidden_states=True,
            )

            self.tokenizer = BertTokenizer.from_pretrained(
                "./bert-base-uncased", local_files_only=True
            )
            self.llm_model = BertModel.from_pretrained(
                "./bert-base-uncased", local_files_only=True
            )

            for param in self.llm_model.parameters():
                param.requires_grad = False

            self.reprogramming_layer = ReprogrammingLayer(
                d_model=self.channel,
                n_heads=8,
                d_keys=self.channel // 8,
                d_llm=768
            ).to(device)

            self.llm_proj = nn.Linear(768, self.channel).to(device)

            # 时间 patching
            self.patch_len = 16
            self.stride = 8
            self.num_patches = (seq_len - self.patch_len) // self.stride + 1
            self.patch_proj = nn.Linear(self.patch_len * num_nodes, self.channel).to(device)

            # LLM 残差门控
            self.llm_gate = nn.Parameter(torch.tensor(-2.0, device=device))

            self.llm_to_nodes = nn.Linear(self.num_patches, num_nodes).to(device)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── 频率处理：对时间维做 FFT ──
    def frequency_domain_processing(self, x):
        freq = torch.fft.rfft(x, dim=1)  # [B, seq_len//2+1, N]
        real = freq.real
        imag = freq.imag
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-6)

        feat = torch.stack([real, imag, mag], dim=-1)  # [B, L//2+1, N, 3]
        B, L, N, C = feat.shape
        tokens = feat.permute(0, 2, 1, 3).reshape(B * N, L, C)  # [B*N, L, 3]

        tokens = self.freq_token_proj(tokens)
        out = self.freq_encoder(tokens)
        out = self.freq_pool(out)

        return out.reshape(B, N, self.channel)   # [B, N, channel]

    def forward(
            self,
            input_data,
            input_data_mark,
            embeddings
    ):

        # ── Normalize ──
        input_data = input_data.float()
        input_data = self.normalize_layers(input_data, 'norm')

        B = input_data.size(0)

        # ── TaTS Text Processing ──
        text_emb = self.text_mlp(embeddings)
        text_ts = self.text_ts_proj(text_emb)
        text_ts = text_ts.permute(0, 2, 1)  # [B, C, L_text]

        # ── Decomposition ──
        cycle_index = input_data_mark[:, 0, 0].long()
        trend = self.moving_avg(input_data)       # [B, seq_len, N]

        if self.use_cycle:
            cycle = self.cycleQueue(cycle_index, self.seq_len)  # [B, seq_len, N]
            residual = input_data - trend - cycle
        else:
            cycle = torch.zeros_like(trend)
            residual = input_data - trend

        trend_f = self.trend_proj(trend.permute(0, 2, 1))      # [B, N, C]
        if self.use_cycle:
            cycle_f = self.cycle_proj(cycle.permute(0, 2, 1))    # [B, N, C]
        else:
            cycle_f = torch.zeros_like(trend_f)
        residual_f = self.residual_proj(residual.permute(0, 2, 1))  # [B, N, C]

        # ── Text-guided decomposition ──
        text_gate = text_ts.mean(dim=2)              # [B, C]
        text_gate = torch.sigmoid(text_gate)
        text_gate = text_gate.unsqueeze(1).expand(-1, self.num_nodes, -1)  # [B, N, C]

        trend_f = trend_f * (1 + self.text_gate_scale * text_gate)
        if self.use_cycle:
            cycle_f = cycle_f * (1 + self.text_gate_scale * text_gate)

        # ── Base Projection ──
        base = self.length_to_feature(input_data.permute(0, 2, 1))  # [B, N, C]

        # ── TaTS Early Fusion ──
        base_f = base.permute(0, 2, 1)  # [B, C, N]
        text_f = text_ts
        if text_f.shape[-1] != base_f.shape[-1]:
            text_f = F.adaptive_avg_pool1d(text_f, base_f.shape[-1])

        fusion = torch.cat([base_f, text_f], dim=1)   # [B, 2C, N]
        ts_input = self.tats_fusion(fusion).permute(0, 2, 1)  # [B, N, C]

        # ── Trend / Cycle / Residual Injection ──
        if self.use_trend:
            ts_input = ts_input + self.trend_scale * trend_f
            if self.use_cycle:
                ts_input = ts_input + self.cycle_scale * cycle_f
        if self.use_residual:
            ts_input = ts_input + self.residual_scale * residual_f

        # ── 位置编码 ──
        ts_input = ts_input + self.position_encoding

        # ── 原始输入加到早期融合特征 ──
        skip_early = self.skip_early_proj(input_data.permute(0, 2, 1))  # [B, N, C]
        ts_input = ts_input + torch.sigmoid(self.skip_early_gate) * skip_early

        # ── Encoder ──
        enc_out = self.ts_encoder(ts_input)  # [B, N, C]
        enc_out = enc_out.permute(0, 2, 1)   # [B, C, N]

        # ── LLM 分支 ──
        data_var = input_data.var(dim=[1, 2]).clamp(min=1e-6)
        trend_strength = trend.var(dim=[1, 2]) / data_var
        cycle_strength = cycle.var(dim=[1, 2]) / data_var
        residual_strength = residual.var(dim=[1, 2]) / data_var
        ct_ratio = cycle_strength / (trend_strength + 1e-6)

        freq_all = torch.fft.rfft(input_data, dim=1)
        mag_all = torch.abs(freq_all).mean(dim=2)
        mag_no_dc = mag_all[:, 1:]
        peak_idx = mag_no_dc.argmax(dim=1) + 1
        dominant_period = self.seq_len / peak_idx.float()

        periods_list = dominant_period.tolist()
        ct_list = ct_ratio.tolist()
        res_list = residual_strength.tolist()

        prompts = [
            f"Dominant period: {periods_list[b]:.1f}. "
            f"Cycle trend ratio: {ct_list[b]:.3f}. "
            f"Residual strength: {res_list[b]:.3f}. "
            f"Forecast next {self.pred_len} steps."
            for b in range(B)
        ]

        prompt_ids = self.tokenizer(
            prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        ).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(
            prompt_ids.to(self.device)
        )

        patches = input_data.unfold(
            dimension=1, size=self.patch_len, step=self.stride
        )
        patches = patches.permute(0, 1, 3, 2).reshape(B, self.num_patches, -1)
        patch_emb = self.patch_proj(patches)

        source_embeddings = self.llm_model.get_input_embeddings().weight
        patch_aligned = self.reprogramming_layer(
            patch_emb, source_embeddings, source_embeddings
        )

        llm_input = torch.cat([prompt_embeddings, patch_aligned], dim=1)
        llm_out = self.llm_model(inputs_embeds=llm_input).last_hidden_state
        llm_patch_out = llm_out[:, prompt_embeddings.size(1):, :]

        llm_feat = self.llm_proj(llm_patch_out)
        llm_feat = llm_feat.permute(0, 2, 1)
        llm_feat = self.llm_to_nodes(llm_feat)   # [B, channel, N]

        # 残差门控融合
        gate_val = torch.sigmoid(self.llm_gate)
        enc_out = (1 - gate_val) * enc_out + gate_val * llm_feat

        # ── 原始输入加到Encoder输出 ──
        skip_enc = self.skip_enc_proj(input_data.permute(0, 2, 1)).permute(0, 2, 1)  # [B, C, N]
        enc_out = enc_out + torch.sigmoid(self.skip_enc_gate) * skip_enc

        # 频率分支
        if self.use_freq_branch:
            freq_enc_out = self.frequency_domain_processing(input_data)  # [B, N, channel]

            freq_text = text_ts
            if freq_text.shape[-1] != self.num_nodes:
                freq_text = F.adaptive_avg_pool1d(freq_text, self.num_nodes)
            freq_gate = self.freq_llm_gate(freq_text.permute(0, 2, 1))  # [B, N, C]
            freq_enc_out = freq_enc_out * (1 + self.freq_gate_scale * freq_gate)

            # Gating Fusion
            fusion_feat = torch.cat(
                [enc_out, freq_enc_out.permute(0, 2, 1)], dim=1
            )
            fusion_feat = self.gate_norm(
                fusion_feat.permute(0, 2, 1)
            ).permute(0, 2, 1)
            local_gate = torch.sigmoid(self.gate_conv(fusion_feat))

            freq_contrib = freq_enc_out.permute(0, 2, 1)  # [B, C, N]
            freq_mixed = local_gate * freq_contrib + (1 - local_gate) * enc_out

            g = torch.sigmoid(self.freq_global_gate)
            enc_out = (1 - g) * enc_out + g * freq_mixed

        # ── CMA ──
        cma_outputs = []
        text_context = text_ts
        if text_context.shape[-1] != self.num_nodes:
            text_context = F.adaptive_avg_pool1d(text_context, self.num_nodes)

        for cma in self.cma_heads:
            out = cma(enc_out, text_context, text_context)
            cma_outputs.append(out)
        fused = self.adaptive_dynamic_heads_cma(cma_outputs)

        # ── Residual Fusion ──
        alpha = self.residual_alpha.view(1, -1, 1)
        cross_out = alpha * fused + (1 - alpha) * enc_out
        cross_out = cross_out.permute(0, 2, 1)  # [B, N, C]

        # 原始输入加到CMA融合后 ──
        skip_cma = self.skip_cma_proj(input_data.permute(0, 2, 1))  # [B, N, C]
        cross_out = cross_out + torch.sigmoid(self.skip_cma_gate) * skip_cma

        # ── Decoder ──
        dec_out = self.decoder(cross_out, cross_out)
        dec_out = self.c_to_length(dec_out).permute(0, 2, 1)  # [B, pred_len, N]

        input_skip = self.input_skip_proj(input_data.permute(0, 2, 1)).permute(0, 2, 1)  # [B, pred_len, N]
        dec_out = dec_out + torch.sigmoid(self.skip_gate) * input_skip

        # ── Denorm ──
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out 

