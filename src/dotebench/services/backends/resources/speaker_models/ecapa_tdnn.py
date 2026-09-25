# Adapted from https://github.com/lawlict/ECAPA-TDNN
# Original: https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification
#
# Changes vs upstream:
#   - Feature extractor for feat_type="wavlm_large" now uses a self-contained copy of
#     the fairseq WavLM code (WavLM.py / modules.py bundled in models/) with weights
#     loaded directly from the fine-tuned checkpoint — no external directory dependency.
#   - Added FairseqWavLMWrapper to match the s3prl UpstreamBase call interface expected
#     by ECAPA_TDNN and to produce byte-identical outputs to the original s3prl path.
#   - Added wavlm_state_dict param to ECAPA_TDNN / ECAPA_TDNN_SMALL (replaces
#     wavlm_model_name / wavlm_cache_dir which needed a separate HF model load).

import os
import scipy.signal as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as trans
from torch.nn.utils.rnn import pad_sequence

DEBUG = os.environ.get("DEBUG", "0") == "1"

# WavLM-large config — matches the config stored in wavlm_large.pt
_WAVLM_LARGE_CFG = {
    "extractor_mode": "layer_norm",
    "encoder_layers": 24,
    "encoder_embed_dim": 1024,
    "encoder_ffn_embed_dim": 4096,
    "encoder_attention_heads": 16,
    "activation_fn": "gelu",
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "activation_dropout": 0.0,
    "encoder_layerdrop": 0.0,
    "dropout_input": 0.0,
    "dropout_features": 0.0,
    "layer_norm_first": True,
    "conv_feature_layers": "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
    "conv_bias": False,
    "feature_grad_mult": 1.0,
    "mask_length": 10,
    "mask_prob": 0.8,
    "mask_selection": "static",
    "mask_other": 0.0,
    "no_mask_overlap": False,
    "mask_min_space": 1,
    "mask_channel_length": 10,
    "mask_channel_prob": 0.0,
    "mask_channel_selection": "static",
    "mask_channel_other": 0.0,
    "no_mask_channel_overlap": False,
    "mask_channel_min_space": 1,
    "conv_pos": 128,
    "conv_pos_groups": 16,
    "relative_position_embedding": True,
    "num_buckets": 320,
    "max_distance": 800,
    "gru_rel_pos": True,
    "normalize": True,
}


# ---------------------------------------------------------------------------
# Self-contained fairseq WavLM wrapper — matches the s3prl call interface
# ---------------------------------------------------------------------------


class FairseqWavLMWrapper(nn.Module):
    """Wraps the bundled fairseq WavLM (WavLM.py) to match the s3prl UpstreamBase
    interface expected by ECAPA_TDNN.

    s3prl interface:
      - Input:  list of 1-D float tensors (one per sample, variable length)
      - Output: dict {"hidden_states": tuple-of-tensors}
                each tensor shape: (batch, time, feat_dim)

    The wrapper exposes .model pointing to the WavLM instance so that
    the attribute check in ECAPA_TDNN.__init__ works:
        len(self.feature_extract.model.encoder.layers) == 24

    Args:
        state_dict: the 'feature_extract.model.*' sub-dict from the finetune
                    checkpoint (with the 'feature_extract.model.' prefix stripped).
                    If None, the model is initialised with random weights (for testing).
    """

    def __init__(self, state_dict: dict | None = None):
        super().__init__()
        from .WavLM import WavLM, WavLMConfig

        cfg = WavLMConfig(_WAVLM_LARGE_CFG)
        self.model = WavLM(cfg)
        # Disable feature gradient multiplier and layer-drop during inference
        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0

        if state_dict is not None:
            missing, unexpected = self.model.load_state_dict(state_dict, strict=True)
            if missing:
                raise RuntimeError(f"FairseqWavLMWrapper: missing keys: {missing}")
            if unexpected:
                raise RuntimeError(
                    f"FairseqWavLMWrapper: unexpected keys: {unexpected}"
                )

        self._cfg_normalize = cfg.normalize

    def forward(self, wavs: list) -> dict:
        if self._cfg_normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        hidden_states = []

        # Inline WavLM.extract_features + TransformerEncoder.forward so hidden
        # states are request-local. This preserves the old s3prl hook semantics:
        # layer inputs for all 24 encoder layers, then the final encoder output.
        with torch.no_grad():
            features = self.model.feature_extractor(padded_wav)

        features = features.transpose(1, 2)
        features = self.model.layer_norm(features)

        padding_mask = self.model.forward_padding_mask(features, wav_padding_mask)

        if self.model.post_extract_proj is not None:
            features = self.model.post_extract_proj(features)

        x = self.model.dropout_input(features)
        encoder = self.model.encoder

        if padding_mask is not None:
            x = x.clone()
            x[padding_mask] = 0

        x_conv = encoder.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not encoder.layer_norm_first:
            x = encoder.layer_norm(x)

        x = F.dropout(x, p=encoder.dropout, training=encoder.training)
        x = x.transpose(0, 1)

        pos_bias = None
        for layer in encoder.layers:
            hidden_states.append(x.transpose(0, 1).detach())
            x, _attn, pos_bias = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_weights=False,
                self_attn_mask=None,
                pos_bias=pos_bias,
            )

        x = x.transpose(0, 1)
        if encoder.layer_norm_first:
            x = encoder.layer_norm(x)
        hidden_states.append(x.detach())

        return {"hidden_states": tuple(hidden_states)}


# ---------------------------------------------------------------------------
# ECAPA-TDNN building blocks (unchanged from upstream)
# ---------------------------------------------------------------------------


class Res2Conv1dReluBn(nn.Module):
    """in_channels == out_channels == channels"""

    def __init__(
        self,
        channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        scale=4,
    ):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=bias,
                )
                for _ in range(self.nums)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)


class Conv1dReluBn(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return x * out.unsqueeze(2)


class SE_Res2Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        scale,
        se_bottleneck_dim,
    ):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(
            out_channels, kernel_size, stride, padding, dilation, scale=scale
        )
        self.Conv1dReluBn2 = Conv1dReluBn(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)
        return x + residual


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att
        in_ch = in_dim * 3 if global_context_att else in_dim
        self.linear1 = nn.Conv1d(in_ch, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)

    def forward(self, x):
        if self.global_context_att:
            ctx_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            ctx_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(
                x
            )
            x_in = torch.cat((x, ctx_mean, ctx_std), dim=1)
        else:
            x_in = x

        if DEBUG:
            import pdb

            pdb.set_trace()

        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


# ---------------------------------------------------------------------------
# ECAPA_TDNN
# ---------------------------------------------------------------------------


class ECAPA_TDNN(nn.Module):
    def __init__(
        self,
        feat_dim=80,
        channels=512,
        emb_dim=192,
        global_context_att=False,
        feat_type="fbank",
        sr=16000,
        feature_selection="hidden_states",
        update_extract=False,
        config_path=None,
        # Used only when feat_type == "wavlm_large":
        # state dict of the fairseq WavLM model (feature_extract.model.* prefix stripped).
        # If None, WavLM is initialised with random weights (weights must be loaded
        # externally via model.feature_extract.model.load_state_dict()).
        wavlm_state_dict: dict | None = None,
    ):
        super().__init__()

        self.feat_type = feat_type
        self.feature_selection = feature_selection
        self.update_extract = update_extract
        self.sr = sr

        if feat_type in ("fbank", "mfcc"):
            self.update_extract = False

        win_len = int(sr * 0.025)
        hop_len = int(sr * 0.01)

        if feat_type == "fbank":
            self.feature_extract = trans.MelSpectrogram(
                sample_rate=sr,
                n_fft=512,
                win_length=win_len,
                hop_length=hop_len,
                f_min=0.0,
                f_max=sr // 2,
                pad=0,
                n_mels=feat_dim,
            )
        elif feat_type == "mfcc":
            melkwargs = dict(
                n_fft=512,
                win_length=win_len,
                hop_length=hop_len,
                f_min=0.0,
                f_max=sr // 2,
                pad=0,
            )
            self.feature_extract = trans.MFCC(
                sample_rate=sr, n_mfcc=feat_dim, log_mels=False, melkwargs=melkwargs
            )
        elif feat_type == "wavlm_large":
            # Self-contained fairseq WavLM — no external directory dependency.
            # weights are loaded from wavlm_state_dict (extracted from the finetune ckpt).
            self.feature_extract = FairseqWavLMWrapper(state_dict=wavlm_state_dict)
        else:
            raise ValueError(
                f"feat_type='{feat_type}' is not supported by this self-contained "
                "implementation. Supported: 'fbank', 'mfcc', 'wavlm_large'."
            )

        # Compatibility check: disable fp32_attention if present (HF WavLM layers have
        # an `.attention` attribute with `fp32_attention`; fairseq WavLM layers use
        # `.self_attn` instead, so this block is a no-op for the fairseq path).
        if feat_type not in ("fbank", "mfcc"):
            layers = self.feature_extract.model.encoder.layers
            if len(layers) == 24:
                layer23 = layers[23]
                if hasattr(layer23, "attention") and hasattr(
                    layer23.attention, "fp32_attention"
                ):
                    layer23.attention.fp32_attention = False
                layer11 = layers[11]
                if hasattr(layer11, "attention") and hasattr(
                    layer11.attention, "fp32_attention"
                ):
                    layer11.attention.fp32_attention = False

            self.feat_num = self.get_feat_num()
            self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))

        if feat_type not in ("fbank", "mfcc"):
            freeze_list = [
                "final_proj",
                "label_embs_concat",
                "mask_emb",
                "project_q",
                "quantizer",
            ]
            for name, param in self.feature_extract.named_parameters():
                for freeze_val in freeze_list:
                    if freeze_val in name:
                        param.requires_grad = False
                        break

        if not self.update_extract:
            for param in self.feature_extract.parameters():
                param.requires_grad = False

        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.channels = [channels] * 4 + [1536]

        self.layer1 = Conv1dReluBn(feat_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(
            self.channels[0],
            self.channels[1],
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            scale=8,
            se_bottleneck_dim=128,
        )
        self.layer3 = SE_Res2Block(
            self.channels[1],
            self.channels[2],
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
            scale=8,
            se_bottleneck_dim=128,
        )
        self.layer4 = SE_Res2Block(
            self.channels[2],
            self.channels[3],
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
            scale=8,
            se_bottleneck_dim=128,
        )

        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, self.channels[-1], kernel_size=1)
        self.pooling = AttentiveStatsPool(
            self.channels[-1],
            attention_channels=128,
            global_context_att=global_context_att,
        )
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)

    def get_feat_num(self):
        self.feature_extract.eval()
        wav = [torch.randn(self.sr).to(next(self.feature_extract.parameters()).device)]
        with torch.no_grad():
            features = self.feature_extract(wav)
        select_feature = features[self.feature_selection]
        if isinstance(select_feature, (list, tuple)):
            return len(select_feature)
        return 1

    def get_feat(self, x):
        if self.update_extract:
            x = self.feature_extract([sample for sample in x])
        else:
            with torch.no_grad():
                if self.feat_type in ("fbank", "mfcc"):
                    x = self.feature_extract(x) + 1e-6
                else:
                    x = self.feature_extract([sample for sample in x])

        if self.feat_type == "fbank":
            x = x.log()

        if self.feat_type not in ("fbank", "mfcc"):
            x = x[self.feature_selection]
            if isinstance(x, (list, tuple)):
                x = torch.stack(x, dim=0)
            else:
                x = x.unsqueeze(0)
            norm_weights = (
                F.softmax(self.feature_weight, dim=-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            x = (norm_weights * x).sum(dim=0)
            x = torch.transpose(x, 1, 2) + 1e-6

        x = self.instance_norm(x)
        return x

    def forward(self, x):
        x = self.get_feat(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn(self.pooling(out))
        out = self.linear(out)

        return out


def ECAPA_TDNN_SMALL(
    feat_dim,
    emb_dim=256,
    feat_type="fbank",
    sr=16000,
    feature_selection="hidden_states",
    update_extract=False,
    config_path=None,
    wavlm_state_dict: dict | None = None,
):
    return ECAPA_TDNN(
        feat_dim=feat_dim,
        channels=512,
        emb_dim=emb_dim,
        feat_type=feat_type,
        sr=sr,
        feature_selection=feature_selection,
        update_extract=update_extract,
        config_path=config_path,
        wavlm_state_dict=wavlm_state_dict,
    )
