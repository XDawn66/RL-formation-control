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
        num_layers=2,
        num_active_robots=None
    ):

        super().__init__(
            observation_space,
            features_dim=features_dim
        )

        self.num_active_robots = num_active_robots

        # observation shape:
        # (max_robots, robot_feature_dim)
        max_robots, robot_feature_dim = observation_space.shape

        self.max_robots = max_robots
        self.robot_feature_dim = robot_feature_dim

        self.d_model = d_model
        # Each robot now has 10 features
        # but this automatically uses observation_space.shape[-1]
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

        # pooled global representation
        self.output_layer = nn.Linear(
            d_model,
            features_dim
        )

    # ========================================================
    # Build mask
    # ========================================================

    def get_masks(self, observations):

        batch_size, num_tokens, _ = observations.shape

        active_mask = torch.zeros(
            (batch_size, num_tokens),
            dtype=torch.bool,
            device=observations.device
        )

        active_mask[:, :self.num_active_robots] = True

        padding_mask = ~active_mask

        return active_mask, padding_mask

    # ========================================================
    # Get PER-ROBOT transformer tokens
    # ========================================================

    def encode_tokens(self, observations):

        """
        Returns:
            tokens:
                [batch, max_robots, d_model]

            active_mask:
                [batch, max_robots]
        """

        active_mask, padding_mask = self.get_masks(
            observations
        )

        # [B, N, robot_feature_dim]
        # ->
        # [B, N, d_model]
        x = self.robot_embedding(observations)

        # Transformer interaction between robots
        tokens = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        # zero padded robot tokens
        tokens = tokens * active_mask.unsqueeze(-1).float()

        return tokens, active_mask

    # ========================================================
    # Pool tokens for GLOBAL gain learning
    # ========================================================

    def pool_tokens(self, tokens, active_mask):

        mask_float = active_mask.unsqueeze(-1).float()

        summed = (
            tokens * mask_float
        ).sum(dim=1)

        count = mask_float.sum(
            dim=1
        ).clamp(min=1.0)

        pooled = summed / count

        return self.output_layer(pooled)

    # ========================================================
    # Normal SB3 feature extractor behavior
    # ========================================================

    def forward(self, observations):

        tokens, active_mask = self.encode_tokens(
            observations
        )

        global_features = self.pool_tokens(
            tokens,
            active_mask
        )

        return global_features