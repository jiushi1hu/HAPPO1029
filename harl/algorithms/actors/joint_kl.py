"""A common-rollout KL budget for sequential, independent categorical actors."""
import math

import numpy as np
import torch


class JointKLBudget:
    """Freeze every actor at the start of one sequential HAPPO update.

    At each shared rollout row, KL between the factorized command policies is
    the sum of the actors' categorical KLs. Every component uses the same rows
    and uniform averaging, without actor-specific controllability masks or
    discount weights. The feasibility shield's executed-release distribution
    is not the distribution being measured here.
    """

    def __init__(self, actors, buffers, target_kl):
        try:
            valid_target = (not isinstance(target_kl, (bool, np.bool_)) and
                            target_kl is not None and np.isfinite(float(target_kl)) and float(target_kl) > 0.)
        except (TypeError, ValueError, OverflowError):
            valid_target = False
        if not valid_target:
            raise ValueError('joint_target_kl must be finite and positive')
        if not actors or len(actors) != len(buffers):
            raise ValueError('Joint KL requires one rollout buffer per actor')
        self.target_kl = float(target_kl)
        self.actors, self.buffers = list(actors), list(buffers)
        grid = buffers[0].actions.shape[:2]
        if len(grid) != 2 or min(grid) < 1:
            raise ValueError('Joint KL requires a nonempty common rollout grid')
        parameters = set()
        for actor, buffer in zip(actors, buffers):
            if (buffer.actions.shape[:2] != grid or buffer.obs[:-1].shape[:2] != grid or
                    buffer.active_masks[:-1].shape[:2] != grid or
                    not np.array_equal(buffer.masks, buffers[0].masks)):
                raise ValueError('Joint KL requires aligned common rollout rows and episode starts')
            if actor.act_space.__class__.__name__ != 'Discrete':
                raise ValueError('Joint KL requires categorical Discrete actors')
            actor_parameters = {id(p) for p in actor.actor.parameters()}
            if parameters.intersection(actor_parameters):
                raise ValueError('Joint KL requires independent actor parameters')
            parameters.update(actor_parameters)
        self.row_count = int(np.prod(grid))
        self.before = [actor.rollout_distribution(buffer) for actor, buffer in zip(actors, buffers)]
        if any(reference.batch_shape != torch.Size([self.row_count]) for reference in self.before):
            raise ValueError('Joint KL requires one categorical distribution per common rollout row')
        self.contributions = [0.] * len(actors)

    @property
    def total_kl(self):
        return math.fsum(self.contributions)

    def for_actor(self, actor_id):
        return _ActorJointKLGuard(self, actor_id)


class _ActorJointKLGuard:
    def __init__(self, budget, actor_id):
        if not 0 <= actor_id < len(budget.actors):
            raise ValueError('Invalid joint KL actor index')
        self.budget, self.actor_id = budget, actor_id
        self.target_kl = budget.target_kl

    @property
    def retained_kl(self):
        return self.budget.total_kl

    @property
    def contribution(self):
        return self.budget.contributions[self.actor_id]

    def validate(self, actor, buffer):
        if self.budget.actors[self.actor_id] is not actor or self.budget.buffers[self.actor_id] is not buffer:
            raise ValueError('Joint KL guard must belong to the actor and rollout buffer being trained')

    @torch.no_grad()
    def actor_kl(self):
        budget, i = self.budget, self.actor_id
        current = budget.actors[i].rollout_distribution(budget.buffers[i])
        divergence = torch.distributions.kl_divergence(budget.before[i], current)
        # Clamp only the mean, matching the local guard's floating-point rule.
        # This is the exact categorical KL up to floating-point roundoff.
        value = divergence.mean().clamp_min(0.)
        if not torch.isfinite(divergence).all() or not torch.isfinite(value):
            raise FloatingPointError('Non-finite common-rollout actor KL')
        return float(value)

    def total_with(self, component):
        if not np.isfinite(component) or component < 0.:
            raise FloatingPointError('Invalid joint KL contribution')
        return math.fsum(component if i == self.actor_id else value
                         for i, value in enumerate(self.budget.contributions))

    def commit(self, component):
        if self.total_with(component) > self.target_kl:
            raise ValueError('Cannot commit a candidate above the joint KL budget')
        self.budget.contributions[self.actor_id] = float(component)
