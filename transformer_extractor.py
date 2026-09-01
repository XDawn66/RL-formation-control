import torch
import torch.nn as nn

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RobotTransformerExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim=64,
        d_model=64,
        nhead=4,
        num_layers=2
    ):
        super().__init__(
            observation_space,
            features_dim=features_dim
        )

        # observation shape should be:
        # (max_robots, robot_feature_dim)
        max_robots, robot_feature_dim = observation_space.shape

        self.max_robots = max_robots
        self.robot_feature_dim = robot_feature_dim

        # Convert each robot's 8 features into a d_model embedding
        self.robot_embedding = nn.Linear(
            robot_feature_dim,
            d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Transformer output after pooling -> SAC feature vector
        self.output_layer = nn.Linear(
            d_model,
            features_dim
        )

    def forward(self, observations):
        """
        observations shape:
        [batch_size, max_robots, robot_feature_dim]

        Example:
        [256, 20, 8]
        """

        # ------------------------------------------------
        # 1. Identify padded robots
        # ------------------------------------------------

        # Robot is padded if ALL 8 values are zero
        active_mask = torch.any(
            observations != 0,
            dim=-1
        )

        # Transformer expects True = IGNORE token
        padding_mask = ~active_mask

        # ------------------------------------------------
        # 2. Embed each robot token
        # ------------------------------------------------

        x = self.robot_embedding(observations)

        # x:
        # [batch, 20, 64]

        # ------------------------------------------------
        # 3. Self-attention across robots
        # ------------------------------------------------

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        # ------------------------------------------------
        # 4. Masked mean pooling
        # ------------------------------------------------

        mask_float = active_mask.unsqueeze(-1).float()

        x = x * mask_float

        summed = x.sum(dim=1)

        count = mask_float.sum(dim=1).clamp(min=1.0)

        pooled = summed / count

        # pooled:
        # [batch, 64]

        # ------------------------------------------------
        # 5. Final fixed-size representation for SAC
        # ------------------------------------------------

        return self.output_layer(pooled)