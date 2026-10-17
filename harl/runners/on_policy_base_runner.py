"""Shared HARL return computation and network modes used by cascade HAPPO.

Feed-forward networks share the original return computation.
"""
import numpy as np
import torch
from harl.utils.trans_tools import _t2n


class OnPolicyBaseRunner:
    """Common operations inherited by the reservoir runner."""

    @torch.no_grad()
    def compute(self):
        """Compute returns and advantages.
        Compute critic evaluation of the last state,
        and then let buffer compute returns, which will be used during training.
        """
        if self.state_type == "EP":
            next_value = self.critic.get_values(
                self.critic_buffer.share_obs[-1],
            )
            next_value = _t2n(next_value)
        elif self.state_type == "FP":
            next_value = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[-1]),
            )
            next_value = np.array(
                np.split(_t2n(next_value), self.algo_args["train"]["n_rollout_threads"])
            )
        self.critic_buffer.compute_returns(next_value, self.value_normalizer)

    def prep_rollout(self):
        """Prepare for rollout."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_rollout()
        self.critic.prep_rollout()

    def prep_training(self):
        """Prepare for training."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_training()
        self.critic.prep_training()
