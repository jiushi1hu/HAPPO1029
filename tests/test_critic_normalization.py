"""Return units and critic clipping must be explicit and internally consistent."""
import copy
from types import MethodType
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gym.spaces import Box

from examples.train_cascade import load_configuration
from harl.algorithms.critics.v_critic import VCritic, validate_value_normalization
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.valuenorm import ValueNorm
from harl.utils.models_tools import huber_loss, mse_loss


class CriticNormalizationTests(unittest.TestCase):
    @staticmethod
    def normalizer(mean=200., std=100.):
        norm = ValueNorm(1)
        with torch.no_grad():
            norm.running_mean.fill_(mean)
            norm.running_mean_sq.fill_(mean*mean+std*std)
            norm.debiasing_term.fill_(1.)
        return norm

    @staticmethod
    def buffer(*, days=1, proper=True, gae=False, gamma=.9):
        return OnPolicyCriticBufferEP(dict(episode_length=days, n_rollout_threads=1,
            gamma=gamma, gae_lambda=.95, use_gae=gae, use_proper_time_limits=proper),
            Box(-1., 1., (3,), dtype=np.float32))

    @staticmethod
    def critic(*, clipped=False, huber=False):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        cfg, _ = load_configuration()
        args = {**cfg['model'], **cfg['algo'], 'hidden_sizes': [8, 8],
                'use_clipped_value_loss': clipped, 'use_huber_loss': huber,
                'huber_delta': .5, 'critic_epoch': 3, 'critic_num_mini_batch': 2}
        return VCritic(args, Box(-1., 1., (3,), dtype=np.float32))

    def assert_state_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_state_equal(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for x, y in zip(first, second):
                self.assert_state_equal(x, y)
        else:
            self.assertEqual(first, second)

    def test_non_gae_bootstrap_is_denormalized_for_both_time_limit_settings(self):
        for proper in (False, True):
            for use_norm in (False, True):
                with self.subTest(proper=proper, value_normalization=use_norm):
                    cb = self.buffer(proper=proper)
                    cb.rewards.fill(1.)
                    norm = self.normalizer() if use_norm else None
                    before = copy.deepcopy(norm.state_dict()) if norm is not None else None
                    cb.compute_returns(np.array([[.5]], dtype=np.float32), norm)
                    self.assertAlmostEqual(float(cb.returns[0, 0, 0]), 226. if use_norm else 1.45, places=5)
                    self.assertEqual(float(cb.returns[-1, 0, 0]), 250. if use_norm else .5)
                    if norm is not None:
                        self.assert_state_equal(before, norm.state_dict())

    def test_terminal_non_gae_return_ignores_bootstrap_in_both_time_limit_modes(self):
        for proper in (False, True):
            for norm in (None, self.normalizer()):
                with self.subTest(proper=proper, value_normalization=norm is not None):
                    cb = self.buffer(proper=proper)
                    cb.rewards.fill(1.)
                    cb.masks[-1] = 0.
                    cb.compute_returns(np.array([[.5]], dtype=np.float32), norm)
                    self.assertEqual(float(cb.returns[0, 0, 0]), 1.)

    def test_non_gae_internal_terminal_boundaries_stop_denormalized_reward_to_go(self):
        for proper in (False, True):
            cb = self.buffer(days=4, proper=proper)
            cb.rewards[:, 0, 0] = [1., 2., 3., 4.]
            cb.masks[2] = 0.
            cb.compute_returns(np.array([[.5]], dtype=np.float32), self.normalizer())
            np.testing.assert_allclose(cb.returns[:-1, 0, 0], [2.8, 2., 209.1, 229.], rtol=0, atol=2e-5)

    def test_gae_annual_path_retains_the_original_reward_unit_equations(self):
        for proper in (False, True):
            cb = self.buffer(days=3, proper=proper, gae=True)
            cb.rewards[:, 0, 0] = [1., 2., 3.]
            cb.value_preds[:, 0, 0] = [.1, -.1, 0., 0.]
            cb.masks[-1] = 0.
            norm = self.normalizer()
            expected, gae = np.zeros(3), 0.
            raw_values = np.array([210., 190., 200., 250.])
            for step in reversed(range(3)):
                mask = float(cb.masks[step+1, 0, 0])
                delta = cb.rewards[step, 0, 0]+.9*raw_values[step+1]*mask-raw_values[step]
                gae = delta+.9*.95*mask*gae
                expected[step] = gae+raw_values[step]
            cb.compute_returns(np.array([[.5]], dtype=np.float32), norm)
            np.testing.assert_allclose(cb.returns[:-1, 0, 0], expected, rtol=0, atol=3e-5)
            self.assertEqual(float(cb.value_preds[-1, 0, 0]), .5)

    def test_value_scale_validation_rejects_only_the_unsupported_combination(self):
        for clipped, normalization in ((False, False), (False, True), (True, False)):
            self.assertIsNone(validate_value_normalization(clipped, normalization))
        with self.assertRaisesRegex(ValueError, 'Clipped value loss with ValueNorm'):
            validate_value_normalization(True, True)

    def test_clipped_value_loss_without_normalization_matches_hand_loss_and_gradient(self):
        for use_huber, expected_loss, expected_gradient in ((False, 1.1125, .55), (True, .6, .25)):
            critic = self.critic(clipped=True, huber=use_huber)
            values = torch.tensor([[1.], [1.1]], requires_grad=True)
            loss = critic.cal_value_loss(values, torch.tensor([[0.], [1.]]), torch.tensor([[2.], [0.]]))
            self.assertAlmostEqual(float(loss), expected_loss, places=6)
            loss.backward()
            torch.testing.assert_close(values.grad, torch.tensor([[0.], [expected_gradient]]), rtol=0, atol=1e-7)

    def test_clipped_value_loss_with_normalization_rejects_before_any_mutation(self):
        critic = self.critic()
        norm = self.normalizer(mean=7., std=2.)
        cb = self.buffer(days=6)
        cb.share_obs[:] = np.random.RandomState(4).normal(size=cb.share_obs.shape)
        cb.returns[:-1, 0, 0] = np.arange(6)
        sample = next(cb.feed_forward_generator_critic(2))
        critic.update(sample, norm)  # Populate Adam moments and gradients.
        critic.use_clipped_value_loss = True
        snapshot = copy.deepcopy((critic.critic.state_dict(), critic.critic_optimizer.state_dict(),
            norm.state_dict(), [p.grad.clone() for p in critic.critic.parameters()], cb.__dict__))
        rng = torch.get_rng_state().clone()
        with patch.object(norm, 'update', side_effect=AssertionError('normalizer was mutated')), \
                patch.object(critic, 'get_values', side_effect=AssertionError('model forward was evaluated')):
            with self.assertRaisesRegex(ValueError, 'Clipped value loss with ValueNorm'):
                critic.cal_value_loss(torch.ones(2, 1), torch.zeros(2, 1), torch.ones(2, 1), norm)
            with self.assertRaisesRegex(ValueError, 'Clipped value loss with ValueNorm'):
                critic.update(sample, norm)
            with self.assertRaisesRegex(ValueError, 'Clipped value loss with ValueNorm'):
                critic.train(cb, norm)
        after = (critic.critic.state_dict(), critic.critic_optimizer.state_dict(), norm.state_dict(),
                 [p.grad for p in critic.critic.parameters()], cb.__dict__)
        self.assert_state_equal(snapshot, after)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_unclipped_value_normalization_preserves_legacy_learning_state_and_rng_exactly(self):
        current = self.critic()
        legacy = copy.deepcopy(current)
        current_norm, legacy_norm = self.normalizer(mean=7., std=2.), self.normalizer(mean=7., std=2.)
        cb = self.buffer(days=6)
        cb.share_obs[:] = np.random.RandomState(3).normal(size=cb.share_obs.shape)
        cb.returns[:-1, 0, 0] = [9., 11., -3., 4., 1., 17.]
        cb.value_preds[:-1, 0, 0] = [-1., 0., 1., 2., 3., 4.]
        before_buffer = copy.deepcopy(cb.__dict__)

        def original_loss(self, values, old_values, targets, value_normalizer=None):
            clipped = old_values+(values-old_values).clamp(-self.clip_param, self.clip_param)
            value_normalizer.update(targets)
            error_clipped = value_normalizer.normalize(targets)-clipped
            error_original = value_normalizer.normalize(targets)-values
            clipped_loss = huber_loss(error_clipped, self.huber_delta) if self.use_huber_loss else mse_loss(error_clipped)
            original = huber_loss(error_original, self.huber_delta) if self.use_huber_loss else mse_loss(error_original)
            return (torch.max(original, clipped_loss) if self.use_clipped_value_loss else original).mean()

        legacy.cal_value_loss = MethodType(original_loss, legacy)
        rng = torch.get_rng_state().clone()
        with patch.object(legacy_norm, 'update', wraps=legacy_norm.update) as legacy_update:
            expected = legacy.train(cb, legacy_norm)
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        with patch.object(current_norm, 'update', wraps=current_norm.update) as current_update:
            actual = current.train(cb, current_norm)
        self.assertEqual(legacy_update.call_count, 6)
        self.assertEqual(current_update.call_count, 6)
        self.assert_state_equal(expected, actual)
        self.assert_state_equal(legacy.critic.state_dict(), current.critic.state_dict())
        self.assert_state_equal(legacy.critic_optimizer.state_dict(), current.critic_optimizer.state_dict())
        self.assert_state_equal(legacy_norm.state_dict(), current_norm.state_dict())
        self.assert_state_equal([p.grad for p in legacy.critic.parameters()], [p.grad for p in current.critic.parameters()])
        self.assert_state_equal(before_buffer, cb.__dict__)
        torch.testing.assert_close(expected_rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_critic_minibatches_reject_invalid_counts_without_consuming_rng(self):
        cb = self.buffer(days=6)
        for count in (0, -1, 1.5, True, np.bool_(True), None, '2', 4, 7):
            rng = torch.get_rng_state().clone()
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'critic_num_mini_batch'):
                next(cb.feed_forward_generator_critic(count))
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_valid_critic_minibatches_visit_every_row_once_per_epoch(self):
        cb = self.buffer(days=6)
        cb.share_obs[:-1, 0, 0] = np.arange(6)
        for count in (1, 2, 3, 6, np.int64(2)):
            rows = [int(i) for sample in cb.feed_forward_generator_critic(count) for i in sample[0][:, 0]]
            self.assertEqual(sorted(rows), list(range(6)))
            self.assertEqual(len(rows), 6)
        remainder = self.buffer(days=5)
        with self.assertRaisesRegex(ValueError, 'without dropping rows'):
            next(remainder.feed_forward_generator_critic(2))


if __name__ == '__main__':
    unittest.main()
