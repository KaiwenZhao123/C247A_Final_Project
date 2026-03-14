# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Sequence

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class LiteConvGRUEncoder(nn.Module):
    """A lightweight temporal encoder that applies a small 1D temporal
    convolution before a bidirectional GRU.

    Compared to the TDS baseline, this keeps temporal modeling expressive while
    reducing both parameter count and wall-clock cost on a single GPU VM.
    Inputs and outputs are both of shape (T, N, num_features).
    """

    def __init__(
        self,
        num_features: int,
        conv_channels: int = 192,
        kernel_width: int = 9,
        hidden_size: int = 160,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.temporal_conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=conv_channels,
            kernel_size=kernel_width,
            padding=kernel_width // 2,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.output_proj = nn.Linear(
            hidden_size * (2 if bidirectional else 1), num_features
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs

        # TNC -> NCT for temporal conv, then back to TNC for GRU.
        x = inputs.permute(1, 2, 0)
        x = self.temporal_conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x.permute(2, 0, 1)

        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = self.output_proj(x)
        return self.layer_norm(x + residual)


class DilatedResidualBlock(nn.Module):
    """Residual temporal block with exponentially increasing dilation."""

    def __init__(
        self,
        num_features: int,
        kernel_width: int = 5,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        padding = dilation * (kernel_width // 2)
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=kernel_width,
                padding=padding,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=1,
            ),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        x = inputs.permute(1, 2, 0)
        x = self.block(x)
        x = x.permute(2, 0, 1)
        return self.layer_norm(x + residual)


class DilatedResidualTCNEncoder(nn.Module):
    """Temporal convolutional encoder with dilated residual blocks.

    Unlike self-attention models, this keeps both training and full-session
    test inference linear in sequence length.
    """

    def __init__(
        self,
        num_features: int,
        num_blocks: int = 8,
        kernel_width: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        blocks = [
            DilatedResidualBlock(
                num_features=num_features,
                kernel_width=kernel_width,
                dilation=2**idx,
                dropout=dropout,
            )
            for idx in range(num_blocks)
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(inputs)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequence models."""

    def __init__(self, model_dim: int, max_len: int = 4096) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.max_len = max_len

        self.register_buffer("pe", self._build_pe(max_len), persistent=False)

    def _build_pe(self, length: int) -> torch.Tensor:
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.model_dim, 2)
            * (-math.log(10000.0) / self.model_dim)
        )
        pe = torch.zeros(length, self.model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[0] > self.pe.shape[0]:
            self.pe = self._build_pe(inputs.shape[0]).to(inputs.device)
        return inputs + self.pe[: inputs.shape[0]]


class TemporalTransformerEncoder(nn.Module):
    """A lightweight Transformer encoder over temporal EMG features.

    Inputs and outputs are both shaped (T, N, num_features). A small linear
    bottleneck keeps attention cost low enough for the user's VM.
    """

    def __init__(
        self,
        num_features: int,
        model_dim: int = 192,
        num_layers: int = 3,
        num_heads: int = 4,
        ff_multiplier: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(num_features, model_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(model_dim=model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim),
        )
        self.output_proj = nn.Linear(model_dim, num_features)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        x = self.input_proj(inputs)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.output_proj(x)
        return self.layer_norm(x + residual)


class StridedTemporalTransformerEncoder(nn.Module):
    """Temporal Transformer encoder with convolutional downsampling.

    This reduces sequence length before self-attention so full-session test
    inference remains tractable on the user's VM.
    """

    def __init__(
        self,
        num_features: int,
        conv_channels: int = 256,
        model_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_multiplier: int = 4,
        dropout: float = 0.2,
        stride: int = 4,
    ) -> None:
        super().__init__()

        self.stride = stride
        self.downsample = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=conv_channels,
                kernel_size=5,
                stride=stride,
                padding=2,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=conv_channels,
                out_channels=model_dim,
                kernel_size=5,
                stride=stride,
                padding=2,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.positional_encoding = SinusoidalPositionalEncoding(
            model_dim=model_dim,
            max_len=8192,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim),
        )
        self.output_proj = nn.Linear(model_dim, num_features)
        self.layer_norm = nn.LayerNorm(num_features)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = (input_lengths + self.stride - 1) // self.stride
        lengths = (lengths + self.stride - 1) // self.stride
        return lengths

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.permute(1, 2, 0)
        x = self.downsample(x)
        x = x.permute(2, 0, 1)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.output_proj(x)
        return self.layer_norm(x)
