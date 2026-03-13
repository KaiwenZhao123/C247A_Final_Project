# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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

class EarlyFusionRotationInvariantMLP(nn.Module):
    """
    Early-fusion version:
    input:  (T, N, bands, electrode_channels, freq)
    output: (T, N, mlp_features[-1])

    It fuses both bands from the start by flattening bands and channels together
    before the first learned projection.
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        assert pooling in {"max", "mean"}

        self.num_bands = num_bands
        self.pooling = pooling
        self.offsets = offsets if len(offsets) > 0 else (0,)

        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features * num_bands, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
            num_bands = 1  # only the first layer sees fused raw features
        self.mlp = nn.Sequential(*mlp)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, bands, C, freq)
        # rotate within each band along the electrode-channel dimension
        rotated = torch.stack(
            [inputs.roll(offset, dims=3) for offset in self.offsets],
            dim=2,
        )  # (T, N, rotation, bands, C, freq)

        # fuse both bands from the start
        rotated = rotated.flatten(start_dim=3)   # (T, N, rotation, bands*C*freq)
        x = self.mlp(rotated)                    # (T, N, rotation, mlp_features[-1])

        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class CrossBandFusionMLP(nn.Module):
    """
    Gradual/continuous fusion:
    1) one small per-band encoder
    2) concatenate both bands
    3) learn cross-band mixing with a fusion MLP

    input:  (T, N, bands, C, freq)
    output: (T, N, fusion_features[-1])
    """

    def __init__(
        self,
        in_features: int,
        band_mlp_features: Sequence[int],
        fusion_mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()

        assert len(band_mlp_features) > 0
        assert len(fusion_mlp_features) > 0

        self.band_encoder = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=band_mlp_features,
            pooling=pooling,
            offsets=offsets,
            num_bands=num_bands,
            stack_dim=stack_dim,
        )

        fusion_in = num_bands * band_mlp_features[-1]
        fusion_layers: list[nn.Module] = []
        for out_features in fusion_mlp_features:
            fusion_layers.extend(
                [
                    nn.Linear(fusion_in, out_features),
                    nn.ReLU(),
                ]
            )
            fusion_in = out_features
        self.fusion = nn.Sequential(*fusion_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # (T, N, bands, band_mlp_features[-1])
        x = self.band_encoder(inputs)
        # (T, N, bands * band_mlp_features[-1])
        x = x.flatten(start_dim=2)
        # (T, N, fusion_mlp_features[-1])
        return self.fusion(x)

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

class TemporalConvBiGRUEncoder(nn.Module):
    """
    Temporal encoder for one wrist stream.

    Input:  (T, N, F)
    Output: (T, N, 2 * gru_hidden_size)

    Structure:
        temporal Conv1d -> ReLU -> BiGRU
    """

    def __init__(
        self,
        in_features: int,
        conv_channels: int,
        kernel_size: int,
        gru_hidden_size: int,
        gru_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        assert kernel_size % 2 == 1, "Use odd kernel size to keep time length unchanged."

        self.temporal_conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()

        self.bigru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, F)
        x = inputs.permute(1, 2, 0)          # (N, F, T)
        x = self.temporal_conv(x)            # (N, conv_channels, T)
        x = self.relu(x)
        x = x.permute(2, 0, 1)               # (T, N, conv_channels)

        x, _ = self.bigru(x)                 # (T, N, 2*gru_hidden_size)
        return x


class DualStreamLateFusionEncoder(nn.Module):
    """
    Dual-stream encoder:
      - split left/right wrists
      - per-stream rotation-invariant MLP
      - per-stream temporal CNN + BiGRU
      - late fusion after temporal feature extraction

    Input:  (T, N, 2, C, freq)
    Output: (T, N, fusion_features[-1])
    """

    def __init__(
        self,
        in_features: int,
        stream_mlp_features: Sequence[int],
        conv_channels: int,
        kernel_size: int,
        gru_hidden_size: int,
        gru_layers: int,
        fusion_features: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        assert len(stream_mlp_features) > 0
        assert len(fusion_features) > 0

        stream_feature_dim = stream_mlp_features[-1]
        temporal_out_dim = 2 * gru_hidden_size

        # one encoder per wrist
        self.left_mlp = RotationInvariantMLP(
            in_features=in_features,
            mlp_features=stream_mlp_features,
        )
        self.right_mlp = RotationInvariantMLP(
            in_features=in_features,
            mlp_features=stream_mlp_features,
        )

        self.left_temporal = TemporalConvBiGRUEncoder(
            in_features=stream_feature_dim,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            gru_hidden_size=gru_hidden_size,
            gru_layers=gru_layers,
            dropout=dropout,
        )
        self.right_temporal = TemporalConvBiGRUEncoder(
            in_features=stream_feature_dim,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            gru_hidden_size=gru_hidden_size,
            gru_layers=gru_layers,
            dropout=dropout,
        )

        fusion_layers: list[nn.Module] = []
        fusion_in = 2 * temporal_out_dim
        for out_features in fusion_features:
            fusion_layers.extend(
                [
                    nn.Linear(fusion_in, out_features),
                    nn.ReLU(),
                ]
            )
            fusion_in = out_features
        self.fusion = nn.Sequential(*fusion_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, bands=2, C, freq)
        left = inputs[:, :, 0]   # (T, N, C, freq)
        right = inputs[:, :, 1]  # (T, N, C, freq)

        left = self.left_mlp(left)          # (T, N, stream_feature_dim)
        right = self.right_mlp(right)       # (T, N, stream_feature_dim)

        left = self.left_temporal(left)     # (T, N, 2*gru_hidden_size)
        right = self.right_temporal(right)  # (T, N, 2*gru_hidden_size)

        fused = torch.cat([left, right], dim=-1)   # (T, N, 4*gru_hidden_size)
        fused = self.fusion(fused)                 # (T, N, fusion_features[-1])
        return fused