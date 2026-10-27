"""On-policy buffer for actor."""

import torch
import numpy as np
from harl.utils.envs_tools import get_shape_from_obs_space, get_shape_from_act_space


class OnPolicyActorBuffer:
    """On-policy buffer for actor data storage."""

    def __init__(self, args, obs_space, act_space):
        """Initialize on-policy actor buffer.
        Args:
            args: (dict) arguments
            obs_space: (gym.Space or list) observation space
            act_space: (gym.Space) action space
        """
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]

        obs_shape = get_shape_from_obs_space(obs_space)

        # Buffer for observations of this actor.
        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *obs_shape),
            dtype=np.float32,
        )

        act_shape = get_shape_from_act_space(act_space)
        # Buffer for available actions of this actor.
        self.available_actions = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, act_space.n),
            dtype=np.float32,
        )

        # Buffer for actions of this actor.
        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32
        )

        # Buffer for action log probs of this actor.
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32
        )

        # Episode-boundary masks retained for rollout bookkeeping.
        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)

        # Buffer for active masks of this actor. Active masks denotes whether the agent is alive.
        self.active_masks = np.ones_like(self.masks)

        self.factor = None

        self.step = 0

    def update_factor(self, factor):
        """Save factor for this actor."""
        self.factor = factor.copy()

    def insert(
        self,
        obs,
        actions,
        action_log_probs,
        masks,
        active_masks=None,
        available_actions=None,
    ):
        """Insert data into actor buffer."""
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.masks[self.step + 1] = masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def discount_weights(self, gamma):
        """Episode-start discount, distinct from discounted reward-to-go in GAE.

        Cascade rollouts start at January 1 and contain one complete year.
        Boundary masks also reset the clock independently in each environment
        if a buffer contains more than one episode. Compute in float64 before
        converting to the network dtype, rather than accumulating float32 error.
        """
        if isinstance(gamma, (bool, np.bool_)):
            raise ValueError("gamma must be finite and between 0 and 1")
        try:
            gamma = float(gamma)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("gamma must be finite and between 0 and 1") from exc
        if not np.isfinite(gamma) or not 0. <= gamma <= 1.:
            raise ValueError("gamma must be finite and between 0 and 1")
        weights = np.ones(self.masks[:-1].shape, dtype=np.float64)
        for t in range(1, self.episode_length):
            weights[t] = np.where(self.masks[t] > 0., gamma * weights[t - 1], 1.)
        return weights.astype(np.float32)

    def feed_forward_generator_actor(
        self, advantages, actor_num_mini_batch, objective_weights,
    ):
        """Training data generator for actor that uses MLP network."""

        # get episode_length, n_rollout_threads, mini_batch_size
        episode_length, n_rollout_threads = self.actions.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        if (isinstance(actor_num_mini_batch, (bool, np.bool_)) or
                not isinstance(actor_num_mini_batch, (int, np.integer)) or actor_num_mini_batch < 1):
            raise ValueError('Actor minibatch count must be a positive integer')
        if batch_size < actor_num_mini_batch or batch_size % actor_num_mini_batch:
            raise ValueError('Actor minibatch count must divide the positive rollout size without dropping rows')
        mini_batch_size = batch_size // actor_num_mini_batch

        # shuffle indices
        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(actor_num_mini_batch)
        ]

        # Combine the first two dimensions (episode_length and n_rollout_threads) to form batch.
        # Take obs shape as an example: 
        # (episode_length + 1, n_rollout_threads, *obs_shape) --> (episode_length, n_rollout_threads, *obs_shape)
        # --> (episode_length * n_rollout_threads, *obs_shape)
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        available_actions = (None if self.available_actions is None else
                             self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1]))
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        if self.factor is None:
            raise ValueError("HAPPO likelihood factors must be set before generating actor batches")
        factor = self.factor.reshape(-1, self.factor.shape[-1])
        advantages = advantages.reshape(-1, 1)
        objective_weights = np.asarray(objective_weights, dtype=np.float32)
        if (objective_weights.shape != self.active_masks[:-1].shape or
                not np.isfinite(objective_weights).all() or (objective_weights < 0.).any()):
            raise ValueError("Objective weights must be finite nonnegative rollout scalars")
        objective_weights = objective_weights.reshape(-1, 1)

        
        for indices in sampler:
            # obs shape: 
            # (episode_length * n_rollout_threads, *obs_shape) --> (mini_batch_size, *obs_shape)
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            available_actions_batch = None if available_actions is None else available_actions[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            adv_targ = advantages[indices]
            factor_batch = factor[indices]
            # Apply the same permutation to time weights and action likelihoods.
            yield (obs_batch, actions_batch, masks_batch, active_masks_batch,
                   old_action_log_probs_batch, adv_targ, available_actions_batch,
                   factor_batch, objective_weights[indices])
