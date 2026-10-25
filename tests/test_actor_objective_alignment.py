"""Episode-start policy gradients, shuffled time weights, and entropy masks."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gym.spaces import Box, Discrete, MultiDiscrete

from examples.train_cascade import load_configuration
from harl.algorithms.actors.happo import HAPPO
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.models.base.act import ACTLayer


class ActorObjectiveAlignmentTests(unittest.TestCase):
    @staticmethod
    def actor(gamma=.99):
        cfg, _ = load_configuration()
        args = {**cfg['model'], **cfg['algo'], 'gamma': gamma,
                'hidden_sizes': [8, 8], 'ppo_epoch': 1, 'entropy_coef': 0.,
                'use_max_grad_norm': False}
        torch.set_num_threads(1)
        actor = HAPPO(args, Box(-1., 1., shape=(1,), dtype=np.float32), Discrete(2))
        # A single shared Bernoulli parameter gives an exactly enumerable
        # gradient without critic error, action-sampling noise, or clipping.
        with torch.no_grad():
            for parameter in actor.actor.parameters():
                parameter.zero_()
        for parameter in actor.actor.parameters():
            parameter.requires_grad_(False)
        bias = actor.actor.act.action_out.linear.bias
        bias.requires_grad_(True)
        actor.actor_optimizer = torch.optim.SGD([bias], lr=.1)
        return actor, bias

    @staticmethod
    def buffer(days, envs=1):
        cfg, _ = load_configuration()
        args = {**cfg['train'], **cfg['model'], 'episode_length': days,
                'n_rollout_threads': envs}
        return OnPolicyActorBuffer(args, Box(-1., 1., shape=(1,), dtype=np.float32), Discrete(2))

    @staticmethod
    def sample(actor, weights, active=None, advantages=None):
        weights = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        n = len(weights)
        obs, actions = np.zeros((n, 1), np.float32), np.ones((n, 1), np.float32)
        with torch.no_grad():
            old_lp, _, _ = actor.evaluate_actions(obs, actions)
        return (obs, actions, np.ones((n, 1), np.float32),
                np.ones((n, 1), np.float32) if active is None else np.asarray(active, np.float32).reshape(n, 1),
                old_lp.numpy(),
                np.ones((n, 1), np.float32) if advantages is None else np.asarray(advantages, np.float32).reshape(n, 1),
                None, np.ones((n, 1), np.float32), weights)

    def test_actual_train_gradient_matches_exact_discounted_objective(self):
        # Day 0 pays a_0; day 364 pays 2*(1-a_364). Both actions share p.
        # J(p)=p+2*gamma**364*(1-p). All four relevant trajectories are
        # enumerated in separate environment columns, each with probability 1/4.
        # At p=1/2, the old uniform-day surrogate moves AGAINST J for gamma=.99.
        for gamma in (0., .5, .99, 1.):
            with self.subTest(gamma=gamma):
                actor, bias = self.actor(gamma)
                b = self.buffer(365, 4)
                first = np.array([0., 0., 1., 1.])
                last = np.array([0., 1., 0., 1.])
                b.actions[0, :, 0], b.actions[-1, :, 0] = first, last
                b.action_log_probs.fill(-math.log(2.))
                b.masks[-1] = 0.
                b.update_factor(np.ones((365, 4, 1), np.float32))
                advantages = np.zeros((365, 4, 1), np.float32)
                advantages[0, :, 0] = first - .5
                advantages[-1, :, 0] = -2. * (last - .5)
                expected_gradient = .25 * (1. - 2. * gamma**364)
                actor.train(b, advantages, 'EP')
                np.testing.assert_allclose(bias.grad.numpy(),
                    [expected_gradient/365., -expected_gradient/365.], rtol=2e-6, atol=1e-9)
                with torch.no_grad():
                    p = float(bias.softmax(0)[1])
                before = .5 + gamma**364
                after = p + 2. * gamma**364 * (1. - p)
                self.assertGreater(after, before)

    def test_weights_reset_on_each_environment_episode_boundary(self):
        b = self.buffer(5, 2)
        b.masks[2, 0] = 0.
        b.masks[3, 1] = 0.
        np.testing.assert_array_equal(b.discount_weights(.5)[:, :, 0],
            [[1., 1.], [.5, .5], [1., .25], [.5, 1.], [.25, .5]])
        np.testing.assert_array_equal(b.discount_weights(0.)[:, :, 0],
            [[1., 1.], [0., 0.], [1., 0.], [0., 1.], [0., 0.]])
        np.testing.assert_array_equal(b.discount_weights(1.), np.ones((5, 2, 1)))

    def test_calendar_weights_cover_common_and_leap_year_and_restart_at_one(self):
        for days in (365, 366, 365):
            b = self.buffer(days)
            np.testing.assert_allclose(b.discount_weights(.99)[:, 0, 0],
                                       .99**np.arange(days), rtol=6e-8)
            self.assertEqual(b.discount_weights(.99)[0, 0, 0], 1.)

    def test_shuffling_keeps_time_weights_with_rows_and_likelihood_factors(self):
        b = self.buffer(5, 2)
        ids = np.arange(10, dtype=np.float32).reshape(5, 2, 1)
        b.obs[:-1] = ids
        b.update_factor(ids + 100.)
        b.masks[2, 0] = b.masks[3, 1] = 0.
        expected = np.array([1., 1., .5, .5, 1., .25, .5, 1., .25, .5])
        for batches in (1, 2, 10):
            seen = []
            for sample in b.feed_forward_generator_actor(ids, batches,
                                                         objective_weights=b.discount_weights(.5)):
                ix = sample[0][:, 0].astype(int)
                seen.extend(ix.tolist())
                np.testing.assert_array_equal(sample[7][:, 0], ix + 100.)
                np.testing.assert_array_equal(sample[8][:, 0], expected[ix])
            self.assertEqual(sorted(seen), list(range(10)))

    def test_one_row_minibatches_retain_weights_and_match_full_batch_gradient(self):
        actor, bias = self.actor(.5)
        b = self.buffer(5)
        b.actions[:, 0, 0] = [0., 1., 1., 0., 1.]
        b.action_log_probs.fill(-math.log(2.))
        b.active_masks[3] = 0.
        b.update_factor(np.arange(1, 6, dtype=np.float32).reshape(5, 1, 1))
        advantages = np.array([2., -1., 3., 4., -2.], np.float32).reshape(5, 1, 1)
        results = []
        for batches in (1, 5):
            losses, gradients = [], []
            for sample in b.feed_forward_generator_actor(advantages, batches,
                                                         objective_weights=b.discount_weights(.5)):
                actor.actor_optimizer.zero_grad()
                with patch.object(actor.actor_optimizer, 'step'):
                    loss, _, _, _ = actor.update(sample)
                losses.append(float(loss))
                gradients.append(np.zeros(2) if bias.grad is None else bias.grad.numpy().copy())
            results.append((np.mean(losses), np.mean(gradients, axis=0)))
        self.assertAlmostEqual(results[0][0], results[1][0], places=7)
        np.testing.assert_allclose(results[0][1], results[1][1], atol=1e-7)

    def test_entropy_uses_same_time_weights_and_forced_action_mask(self):
        actor, bias = self.actor()
        actor.entropy_coef = .4
        with torch.no_grad():
            bias.copy_(torch.tensor([-.2, .2]))
        sample = self.sample(actor, [1., .5, .25, 0.], active=[1., 0., 1., 1.], advantages=[0.]*4)
        # Entropy is an exploration bonus, not a sequential importance factor.
        sample = sample[:7] + (np.full((4, 1), 9., np.float32),) + sample[8:]
        reference = bias.detach().clone().requires_grad_(True)
        expected_entropy = (1.25/4.) * torch.distributions.Categorical(logits=reference).entropy()
        expected_gradient, = torch.autograd.grad(-.4 * expected_entropy, reference)
        with patch.object(actor.actor_optimizer, 'step'):
            _, entropy, _, _ = actor.update(sample)
        self.assertAlmostEqual(float(entropy), float(expected_entropy), places=7)
        torch.testing.assert_close(bias.grad, expected_gradient, rtol=1e-6, atol=1e-8)

    def test_gamma_one_is_full_calendar_mean_with_zero_forced_contribution(self):
        actor, _ = self.actor(1.)
        sample = self.sample(actor, [1., 1.], active=[1., 0.], advantages=[2., 1000.])
        with patch.object(actor.actor_optimizer, 'step'):
            loss, _, _, _ = actor.update(sample)
        self.assertAlmostEqual(float(loss), -1.)

    def test_zero_effective_weight_never_advances_adam_momentum(self):
        actor, bias = self.actor(0.)
        actor.actor_optimizer = torch.optim.Adam([bias], lr=.001)
        actor.update(self.sample(actor, [1.]))  # Populate Adam's moments.
        saved_bias = bias.detach().clone()
        saved_optimizer = copy.deepcopy(actor.actor_optimizer.state_dict())
        samples = [self.sample(actor, [0.]),
                   self.sample(actor, [1.], active=[0.]),
                   self.sample(actor, [1., 0.], active=[0., 1.])]
        for sample in samples:
            with patch.object(actor, 'evaluate_actions') as evaluate:
                loss, entropy, norm, _ = actor.update(sample)
            evaluate.assert_not_called()
            self.assertEqual([float(loss), float(entropy), float(norm)], [0., 0., 0.])
            torch.testing.assert_close(bias, saved_bias, rtol=0, atol=0)
            current = actor.actor_optimizer.state_dict()
            self.assertEqual(current['param_groups'], saved_optimizer['param_groups'])
            for key, value in saved_optimizer['state'][0].items():
                torch.testing.assert_close(current['state'][0][key], value, rtol=0, atol=0)

    def test_gamma_and_sample_weights_are_validated(self):
        b = self.buffer(2)
        for gamma in (-1., 1.1, np.nan, np.inf, True, None):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                b.discount_weights(gamma)
        actor, _ = self.actor()
        sample = self.sample(actor, [1., .5])
        for weights in (np.ones((2,)), np.array([[-1.], [1.]]), np.array([[np.nan], [1.]])):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                actor.update(sample[:-1] + (weights,))

    def test_unreduced_entropy_supports_joint_discrete_and_continuous_actions(self):
        cfg, _ = load_configuration()
        for space in (Discrete(3), MultiDiscrete([2, 3]),
                      Box(-1., 1., shape=(2,), dtype=np.float32)):
            with self.subTest(space=space):
                layer = ACTLayer(space, 8, 'orthogonal_', .01, cfg['model'])
                features = torch.zeros(3, 8)
                actions, _ = layer(features)
                lp, entropy, _ = layer.evaluate_actions(features, actions,
                    active_masks=torch.zeros(3, 1), return_entropy_per_sample=True)
                self.assertEqual(tuple(lp.shape), (3, 1) if isinstance(space, (Discrete, MultiDiscrete)) else (3, 2))
                self.assertEqual(tuple(entropy.shape), (3,))
                if isinstance(space, MultiDiscrete):
                    torch.testing.assert_close(entropy, torch.full((3,), math.log(6.)))
                elif isinstance(space, Discrete):
                    torch.testing.assert_close(entropy, torch.full((3,), math.log(3.)))
                self.assertTrue(torch.isfinite(entropy).all())


if __name__ == '__main__':
    unittest.main()
