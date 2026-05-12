from __future__ import annotations

from typing import Any

from .graph import Graph

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, lambda_factor: float) -> torch.Tensor:
        ctx.lambda_factor = float(lambda_factor)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return (-ctx.lambda_factor * grad_output), None

def grad_reverse(inputs: torch.Tensor, lambda_factor: float) -> torch.Tensor:
    return GradientReversalFunction.apply(inputs, float(lambda_factor))

class ConvTemporalGraphical(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        t_kernel_size: int = 1,
        t_stride: int = 1,
        t_padding: int = 0,
        t_dilation: int = 1,
        bias: bool = True) -> None:
        
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * self.kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )

    def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if adjacency.size(0) != self.kernel_size:
            raise ValueError("Adjacency spatial kernel size does not match the temporal graph convolution.")
        outputs = self.conv(inputs)
        batch_size, kernel_channels, time_steps, num_joints = outputs.size()
        outputs = outputs.view(batch_size, self.kernel_size, kernel_channels // self.kernel_size, time_steps, num_joints)
        outputs = torch.einsum("nkctv,kvw->nctw", outputs, adjacency)
        return outputs.contiguous(), adjacency

class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        *,
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True) -> None:
        
        super().__init__()
        if len(kernel_size) != 2 or kernel_size[0] % 2 != 1:
            raise ValueError("kernel_size must be a tuple (odd_temporal_kernel, spatial_kernel).")

        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size[0], 1),
                stride=(stride, 1),
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=False),
        )

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = 0.0 if self.residual is None else self.residual(inputs)
        outputs, adjacency = self.gcn(inputs, adjacency)
        outputs = self.tcn(outputs) + residual
        return self.activation(outputs), adjacency

class PoseSTGCNEncoder(nn.Module):
    """Pose encoder derived from the reference ST-GCN implementation."""

    def __init__(
        self,
        in_channels: int,
        *,
        graph_layout: str = "imugpt22",
        hidden_dim: int = 128,
        dropout: float = 0.1,
        edge_importance_weighting: bool = True
    ) -> None:
        super().__init__()
        self.graph = Graph(layout=graph_layout, strategy="spatial")
        adjacency = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", adjacency)

        spatial_kernel_size = adjacency.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        hidden_dim = int(hidden_dim)

        self.data_bn = nn.BatchNorm1d(in_channels * adjacency.size(1))
        self.blocks = nn.ModuleList(
            [
                STGCNBlock(in_channels, 64, kernel_size, residual=False, dropout=dropout),
                STGCNBlock(64, 64, kernel_size, dropout=dropout),
                STGCNBlock(64, 64, kernel_size, dropout=dropout),
                STGCNBlock(64, hidden_dim, kernel_size, dropout=dropout),
                STGCNBlock(hidden_dim, hidden_dim, kernel_size, dropout=dropout)
            ]
        )
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones_like(self.A)) for _ in self.blocks]
            )
        else:
            self.edge_importance = [1.0] * len(self.blocks)

        self.output_projection = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

    def forward(self, pose_windows: torch.Tensor) -> torch.Tensor:
        if pose_windows.ndim != 4:
            raise ValueError("pose_windows must have shape [batch, time, joints, channels].")

        inputs = pose_windows.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()
        batch_size, channels, time_steps, num_joints, num_instances = inputs.size()
        inputs = inputs.permute(0, 4, 3, 1, 2).contiguous()
        inputs = inputs.view(batch_size * num_instances, num_joints * channels, time_steps)
        inputs = self.data_bn(inputs)
        inputs = inputs.view(batch_size, num_instances, num_joints, channels, time_steps)
        inputs = inputs.permute(0, 1, 3, 4, 2).contiguous()
        inputs = inputs.view(batch_size * num_instances, channels, time_steps, num_joints)

        outputs = inputs
        for block, importance in zip(self.blocks, self.edge_importance):
            outputs, _ = block(outputs, self.A * importance)

        _, encoded_channels, encoded_time_steps, encoded_joints = outputs.size()
        outputs = outputs.view(batch_size, num_instances, encoded_channels, encoded_time_steps, encoded_joints)
        outputs = outputs.mean(dim=1).mean(dim=-1)
        outputs = self.output_projection(outputs)
        return outputs.transpose(1, 2).contiguous()


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int = 1, dropout: float = 0.1, kernel_size: int = 3) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels)
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(inputs) + inputs)


class IMUTCNEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        *,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.1) -> None:
        
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([ResidualTemporalBlock(hidden_dim, dilation=2**block_index, dropout=dropout) for block_index in range(int(num_blocks))])

    def forward(self, imu_windows: torch.Tensor) -> torch.Tensor:
        if imu_windows.ndim != 4:
            raise ValueError("imu_windows must have shape [batch, time, sensors, channels].")
        batch_size, time_steps, num_sensors, num_channels = imu_windows.shape
        inputs = imu_windows.permute(0, 2, 3, 1).reshape(batch_size, num_sensors * num_channels, time_steps)
        outputs = self.input_projection(inputs)
        for block in self.blocks:
            outputs = block(outputs)
        return outputs.transpose(1, 2).contiguous()

class GatedFusionBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.pose_projection = nn.Linear(hidden_dim, hidden_dim)
        self.imu_projection = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, pose_features: torch.Tensor, imu_features: torch.Tensor) -> torch.Tensor:
        pose_projected = self.pose_projection(pose_features)
        imu_projected = self.imu_projection(imu_features)
        gate = self.gate(torch.cat([pose_projected, imu_projected], dim=-1))
        return gate * pose_projected + (1.0 - gate) * imu_projected

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(sequence_features).squeeze(-1), dim=1)
        return torch.sum(sequence_features * weights.unsqueeze(-1), dim=1)

class MultitaskFusionClassifier(nn.Module):
    def __init__(
        self,
        *,
        pose_in_channels: int,
        imu_in_channels: int,
        num_emotions: int,
        num_modalities: int,
        num_stimuli: int,
        num_flat_tags: int | None = None,
        use_pose_branch: bool = True,
        use_imu_branch: bool = True,
        use_domain_head: bool = False,
        graph_layout: str = "imugpt22",
        hidden_dim: int = 128,
        trunk_blocks: int = 2,
        dropout: float = 0.1,
        quality_dim: int = 0,
        modality_dropout_p: float = 0.0) -> None:
        
        super().__init__()
        if not bool(use_pose_branch) and not bool(use_imu_branch):
            raise ValueError("At least one branch must be enabled.")

        self.use_pose_branch = bool(use_pose_branch)
        self.use_imu_branch = bool(use_imu_branch)
        self.use_domain_head = bool(use_domain_head)
        self.quality_dim = int(quality_dim)
        self.modality_dropout_p = float(modality_dropout_p)

        hidden_dim = int(hidden_dim)
        if self.use_pose_branch:
            self.pose_encoder = PoseSTGCNEncoder(
                pose_in_channels,
                graph_layout=graph_layout,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            
        if self.use_imu_branch:
            self.imu_encoder = IMUTCNEncoder(
                imu_in_channels,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            
        if self.use_pose_branch and self.use_imu_branch:
            self.fusion = GatedFusionBlock(hidden_dim)

        self.temporal_trunk = nn.Sequential(*[ResidualTemporalBlock(hidden_dim, dilation=2**block_index, dropout=dropout) for block_index in range(int(trunk_blocks))])
        self.temporal_pool = AttentionPooling(hidden_dim)

        if self.quality_dim > 0:
            self.quality_projection = nn.Sequential(
                nn.Linear(self.quality_dim, max(8, hidden_dim // 4)),
                nn.GELU()
            )
            
            head_input_dim = hidden_dim + max(8, hidden_dim // 4)
        else:
            self.quality_projection = None
            head_input_dim = hidden_dim

        self.embedding_projection = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.emotion_head = self._build_head(hidden_dim, num_emotions, dropout)
        self.modality_head = self._build_head(hidden_dim, num_modalities, dropout)
        self.stimulus_head = self._build_head(hidden_dim, num_stimuli, dropout)
        self.flat_tag_head = None if num_flat_tags is None else self._build_head(hidden_dim, int(num_flat_tags), dropout)
        self.domain_head = None if not self.use_domain_head else self._build_head(hidden_dim, 2, dropout)

    @staticmethod
    def _build_head(input_dim: int, output_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
        )

    def _apply_modality_dropout(self, pose_features: torch.Tensor | None, imu_features: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.training or self.modality_dropout_p <= 0.0:
            return pose_features, imu_features
        if pose_features is None or imu_features is None:
            return pose_features, imu_features

        batch_size = pose_features.shape[0]
        drop_mask = torch.rand(batch_size, device=pose_features.device) < self.modality_dropout_p
        if not bool(torch.any(drop_mask)):
            return pose_features, imu_features
        pose_drop_mask = drop_mask & (torch.rand(batch_size, device=pose_features.device) < 0.5)
        imu_drop_mask = drop_mask & ~pose_drop_mask

        pose_features = pose_features.clone()
        imu_features = imu_features.clone()
        pose_features[pose_drop_mask] = 0.0
        imu_features[imu_drop_mask] = 0.0
        return pose_features, imu_features

    def forward(
        self,
        *,
        pose_inputs: torch.Tensor | None,
        imu_inputs: torch.Tensor | None,
        quality_inputs: torch.Tensor | None = None,
        domain_lambda: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        pose_features = self.pose_encoder(pose_inputs) if self.use_pose_branch and pose_inputs is not None else None
        imu_features = self.imu_encoder(imu_inputs) if self.use_imu_branch and imu_inputs is not None else None
        pose_features, imu_features = self._apply_modality_dropout(pose_features, imu_features)

        if pose_features is not None and imu_features is not None:
            sequence_features = self.fusion(pose_features, imu_features)
        elif pose_features is not None:
            sequence_features = pose_features
        elif imu_features is not None:
            sequence_features = imu_features
        else:
            raise ValueError("No active modality was provided to the classifier.")

        sequence_features = self.temporal_trunk(sequence_features.transpose(1, 2)).transpose(1, 2).contiguous()
        pooled_embedding = self.temporal_pool(sequence_features)

        if self.quality_projection is not None:
            if quality_inputs is None:
                raise ValueError("quality_inputs must be provided when quality_dim > 0.")
            pooled_embedding = torch.cat([pooled_embedding, self.quality_projection(quality_inputs)], dim=1)

        shared_embedding = self.embedding_projection(pooled_embedding)
        outputs = {
            "embedding": shared_embedding,
            "emotion_logits": self.emotion_head(shared_embedding),
            "modality_logits": self.modality_head(shared_embedding),
            "stimulus_logits": self.stimulus_head(shared_embedding),
        }
        if self.flat_tag_head is not None:
            outputs["flat_tag_logits"] = self.flat_tag_head(shared_embedding)
        if self.domain_head is not None:
            outputs["domain_logits"] = self.domain_head(grad_reverse(shared_embedding, domain_lambda))
        return outputs
