import torch as th
import torch.nn as nn

from stable_baselines3.sac.policies import (
    SACPolicy,
    Actor,
    LOG_STD_MIN,
    LOG_STD_MAX,
)

from stable_baselines3.common.preprocessing import get_action_dim


class PerTokenActor(Actor):
    """
    SAC actor with two branches:

    global pooled Transformer feature
        -> g1, g2

    individual Transformer tokens
        -> [avoid_x, avoid_y] for each robot
    """

    def __init__(
        self,
        *args,
        num_active_robots=5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if self.use_sde:
            raise NotImplementedError(
                "PerTokenActor currently assumes use_sde=False."
            )

        self.num_active_robots = num_active_robots

        # --------------------------------------------------
        # Dimensions
        # --------------------------------------------------

        self.token_dim = self.features_extractor.d_model

        last_layer_dim = (
            self.net_arch[-1]
            if len(self.net_arch) > 0
            else self.features_dim
        )

        action_dim = get_action_dim(self.action_space)

        expected_action_dim = (
            2 + 2 * self.num_active_robots
        )

        assert action_dim == expected_action_dim, (
            f"Action dimension mismatch: "
            f"environment has {action_dim}, "
            f"but PerTokenActor expects {expected_action_dim}"
        )

        # --------------------------------------------------
        # GLOBAL branch
        #
        # pooled feature
        # -> SAC MLP
        # -> g1, g2
        # --------------------------------------------------

        self.gain_mu = nn.Linear(
            last_layer_dim,
            2
        )

        self.gain_log_std = nn.Linear(
            last_layer_dim,
            2
        )

        # --------------------------------------------------
        # LOCAL branch
        #
        # SAME head is applied to every robot token
        #
        # token_i -> [avoid_x_i, avoid_y_i]
        # --------------------------------------------------

        self.avoid_mu = nn.Linear(
            self.token_dim,
            2
        )

        self.avoid_log_std = nn.Linear(
            self.token_dim,
            2
        )

    def get_action_dist_params(self, obs):
        """
        Returns SAC distribution parameters:

        mean_actions:
            [B, 2 + 2*N]

        log_std:
            [B, 2 + 2*N]
        """

        # ==================================================
        # 1. Transformer tokens
        # ==================================================

        tokens, active_mask = (
            self.features_extractor.encode_tokens(obs)
        )

        # tokens:
        # [B, max_robots, d_model]

        # ==================================================
        # 2. Global pooled representation
        # ==================================================

        global_features = (
            self.features_extractor.pool_tokens(
                tokens,
                active_mask
            )
        )

        # global_features:
        # [B, 64]

        # Existing SAC MLP:
        # 64 -> 128 -> 128
        latent_global = self.latent_pi(
            global_features
        )

        # ==================================================
        # 3. GLOBAL gain actions
        # ==================================================

        gain_mu = self.gain_mu(
            latent_global
        )

        gain_log_std = self.gain_log_std(
            latent_global
        )

        # [B, 2]

        # ==================================================
        # 4. PER-ROBOT avoidance actions
        # ==================================================

        active_tokens = tokens[
            :,
            :self.num_active_robots,
            :
        ]

        # Example with 5 robots:
        # [B, 5, 64]

        avoid_mu = self.avoid_mu(
            active_tokens
        )

        avoid_log_std = self.avoid_log_std(
            active_tokens
        )

        # [B, 5, 2]

        batch_size = obs.shape[0]

        avoid_mu = avoid_mu.reshape(
            batch_size,
            -1
        )

        avoid_log_std = avoid_log_std.reshape(
            batch_size,
            -1
        )

        # [B, 10]

        # ==================================================
        # 5. Combine global + local actions
        # ==================================================

        mean_actions = th.cat(
            [
                gain_mu,
                avoid_mu
            ],
            dim=1
        )

        log_std = th.cat(
            [
                gain_log_std,
                avoid_log_std
            ],
            dim=1
        )

        log_std = th.clamp(
            log_std,
            LOG_STD_MIN,
            LOG_STD_MAX
        )

        return mean_actions, log_std, {}


class PerTokenSACPolicy(SACPolicy):
    """
    Standard SACPolicy except that the actor is replaced
    with PerTokenActor.
    """

    def make_actor(
        self,
        features_extractor=None
    ):

        actor_kwargs = (
            self._update_features_extractor(
                self.actor_kwargs,
                features_extractor
            )
        )

        num_active_robots = (
            self.features_extractor_kwargs[
                "num_active_robots"
            ]
        )

        return PerTokenActor(
            **actor_kwargs,
            num_active_robots=num_active_robots
        ).to(self.device)