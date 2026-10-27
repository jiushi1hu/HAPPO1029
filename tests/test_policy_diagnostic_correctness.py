"""Policy measurements must use the effective objective and original action support."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gym.spaces import Box, Discrete

from examples.train_cascade import load_configuration
from harl.algorithms.actors.happo import HAPPO
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.envs.cascade_reservoir.diagnostics import actor_distribution


class PolicyDiagnosticCorrectnessTests(unittest.TestCase):
    def fixture(self, *, gamma=.99, epochs=1, batches=1, lr=0., rollback=False, target=None):
        torch.set_num_threads(1)
        torch.manual_seed(83)
        cfg, _ = load_configuration()
        args = {**cfg['model'], **cfg['algo'], 'hidden_sizes': [4], 'gamma': gamma,
                'ppo_epoch': epochs, 'actor_num_mini_batch': batches, 'lr': lr,
                'kl_rollback': rollback, 'target_kl': target, 'joint_target_kl': None}
        actor = HAPPO(args, Box(-1., 1., (3,), dtype=np.float32), Discrete(3))
        with torch.no_grad():
            for parameter in actor.actor.parameters():
                parameter.zero_()
        buffer = OnPolicyActorBuffer(dict(episode_length=4, n_rollout_threads=1),
                                     actor.obs_space, actor.act_space)
        buffer.actions[:, 0, 0] = [0, 1, 2, 0]
        buffer.update_factor(np.ones_like(buffer.actions))
        self.refresh_likelihoods(actor, buffer)
        advantage = np.arange(1., 5., dtype=np.float32)[:, None, None]
        return actor, buffer, advantage

    @staticmethod
    def refresh_likelihoods(actor, buffer):
        with torch.no_grad():
            logp, _, _ = actor.evaluate_actions(buffer.obs[:-1, 0], buffer.actions[:, 0],
                None if buffer.available_actions is None else buffer.available_actions[:-1, 0])
        buffer.action_log_probs[:, 0] = logp.numpy()
        buffer.action_log_probs[buffer.active_masks[:-1] == 0.] = 0.

    @staticmethod
    def state(actor):
        return copy.deepcopy((actor.actor.state_dict(), actor.actor_optimizer.state_dict(),
            [None if p.grad is None else p.grad.clone() for p in actor.actor.parameters()]))

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
            for a, b in zip(first, second):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(first, second)

    @staticmethod
    def fake_result(ratios):
        zero = torch.tensor(0.)
        return zero, zero, zero, torch.tensor(ratios, dtype=torch.float32)[:, None]

    def test_unchanged_policy_reports_identity_on_controllable_rows(self):
        actor, buffer, advantage = self.fixture()
        buffer.active_masks[2:] = 0.
        self.refresh_likelihoods(actor, buffer)
        parameters = copy.deepcopy(actor.actor.state_dict())
        info = actor.train(buffer, advantage, 'EP')
        self.assertEqual(info['ratio'], 1.)
        self.assertEqual(info['ratio_samples'], 2.)
        self.assert_state_equal(parameters, actor.actor.state_dict())

    def test_ratio_is_row_weighted_across_unequal_effective_minibatch_sizes(self):
        actor, buffer, advantage = self.fixture(batches=2)
        buffer.active_masks[1] = 0.
        self.refresh_likelihoods(actor, buffer)
        with patch('torch.randperm', side_effect=lambda n: torch.arange(n)), \
                patch.object(actor, 'update', side_effect=[
                    self.fake_result([1., 999.]), self.fake_result([3., 5.])]):
            info = actor.train(buffer, advantage, 'EP')
        # Three effective rows have ratios 1, 3 and 5. Averaging the two
        # minibatch means instead would incorrectly report (1+4)/2 = 2.5.
        self.assertEqual(info['ratio'], 3.)
        self.assertEqual(info['ratio_samples'], 3.)

    def test_zero_discount_rows_do_not_contribute_to_ratio(self):
        actor, buffer, advantage = self.fixture(gamma=0.)
        with patch('torch.randperm', side_effect=lambda n: torch.arange(n)), \
                patch.object(actor, 'update', return_value=self.fake_result([2., 50., 100., 200.])):
            info = actor.train(buffer, advantage, 'EP')
        self.assertEqual(info['ratio'], 2.)
        self.assertEqual(info['ratio_samples'], 1.)

    def test_rejected_candidate_does_not_contribute_to_ratio(self):
        actor, buffer, advantage = self.fixture(batches=2, rollback=True, target=100.)
        buffer.active_masks[1] = 0.
        self.refresh_likelihoods(actor, buffer)
        with patch('torch.randperm', side_effect=lambda n: torch.arange(n)), \
                patch.object(actor, 'update', side_effect=[
                    self.fake_result([2., 999.]), self.fake_result([10., 20.])]), \
                patch.object(actor, 'rollout_kl', side_effect=[.1, 200.]):
            info = actor.train(buffer, advantage, 'EP')
        self.assertEqual(info['ratio'], 2.)
        self.assertEqual(info['ratio_samples'], 1.)
        self.assertEqual(info['optimizer_steps'], 1.)
        self.assertEqual(info['optimizer_rejected_steps'], 1.)

    def test_empty_objective_and_all_rejection_keep_explicit_zero_count(self):
        for case in ('forced', 'zero_discount', 'rejected'):
            with self.subTest(case=case):
                actor, buffer, advantage = self.fixture(gamma=0. if case=='zero_discount' else .99,
                    rollback=case=='rejected', target=1e-12 if case=='rejected' else None,
                    lr=.02 if case=='rejected' else 0.)
                if case=='forced':
                    buffer.active_masks.fill(0.)
                elif case=='zero_discount':
                    buffer.active_masks[0] = 0.
                self.refresh_likelihoods(actor, buffer)
                info = actor.train(buffer, advantage, 'EP')
                self.assertEqual(info['ratio'], 0.)
                self.assertEqual(info['ratio_samples'], 0.)

    def test_ratio_measurement_cannot_change_optimizer_gradients_or_rng(self):
        actor, buffer, advantage = self.fixture(epochs=2, batches=2, lr=.005)
        buffer.active_masks[1] = 0.
        self.refresh_likelihoods(actor, buffer)
        actor.train(buffer, advantage, 'EP')  # Populate Adam and gradients.
        self.refresh_likelihoods(actor, buffer)
        outcomes, infos = [], []
        for replace_measurement in (False, True):
            current, rollout = copy.deepcopy((actor, buffer))
            real_update = current.update

            def update(sample):
                result = real_update(sample)
                # Change only the already-computed metric returned after Adam;
                # this must not feed back into any optimization calculation.
                return result[:3]+(torch.full_like(result[3], 17.),)

            torch.manual_seed(119)
            if replace_measurement:
                with patch.object(current, 'update', side_effect=update):
                    infos.append(current.train(rollout, advantage, 'EP'))
            else:
                infos.append(current.train(rollout, advantage, 'EP'))
            outcomes.append((self.state(current), copy.deepcopy(vars(rollout)), torch.get_rng_state().clone()))
        self.assert_state_equal(outcomes[0], outcomes[1])
        self.assertNotEqual(infos[0]['ratio'], infos[1]['ratio'])
        self.assertEqual(infos[1]['ratio'], 17.)
        self.assertEqual(infos[0]['ratio_samples'], 6.)

    def test_distribution_preserves_availability_entropy_mode_and_kl(self):
        actor, buffer, _ = self.fixture()
        buffer.actions.fill(1.)
        buffer.available_actions[:, :, 0] = 0.
        with torch.no_grad():
            actor.actor.act.action_out.linear.bias.copy_(torch.tensor([4., 0., 0.]))
        before = actor_distribution(actor, buffer)
        torch.testing.assert_close(before.probs, torch.tensor([[0., .5, .5]]).repeat(4, 1))
        self.assertEqual(before.mode()[0].item(), 1)
        self.assertAlmostEqual(float(before.entropy()[0]), float(np.log(2.)), places=6)
        with torch.no_grad():
            actor.actor.act.action_out.linear.bias[0] = 100.
        unchanged = actor_distribution(actor, buffer)
        torch.testing.assert_close(before.logits, unchanged.logits, rtol=0, atol=0)
        self.assertEqual(float(torch.distributions.kl_divergence(before, unchanged).max()), 0.)
        with torch.no_grad():
            actor.actor.act.action_out.linear.bias[2] = 2.
        after = actor_distribution(actor, buffer)
        self.assertEqual(after.mode()[0].item(), 2)
        expected = actor.rollout_distribution(buffer)
        torch.testing.assert_close(after.probs, expected.probs, rtol=0, atol=0)
        probabilities = np.array([1., np.exp(2.)]); probabilities /= probabilities.sum()
        independent_kl = float(np.sum(.5*np.log(.5/probabilities)))
        self.assertAlmostEqual(float(torch.distributions.kl_divergence(before, after)[0]), independent_kl, places=6)

    def test_distribution_without_availability_is_observational(self):
        actor, buffer, advantage = self.fixture(lr=.005)
        actor.train(buffer, advantage, 'EP')
        buffer.available_actions = None
        original = self.state(actor), copy.deepcopy(vars(buffer)), torch.get_rng_state().clone(), actor.actor.training
        first, second = actor_distribution(actor, buffer), actor.rollout_distribution(buffer)
        torch.testing.assert_close(first.probs, second.probs, rtol=0, atol=0)
        self.assert_state_equal(original,
            (self.state(actor), vars(buffer), torch.get_rng_state(), actor.actor.training))


if __name__ == '__main__':
    unittest.main()
