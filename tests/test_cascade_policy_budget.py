"""Policy epoch budgets and full-rollout KL stopping preserve HAPPO accounting."""
import copy
import io
from contextlib import redirect_stderr, redirect_stdout
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gym.spaces import Box, Discrete

from cascade_test_utils import load_configuration
from examples.train_cascade import main
from harl.algorithms.actors.happo import HAPPO, validate_policy_optimization
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
import test_cascade_reward_penalties as reward_fixtures


class PolicyBudgetTests(unittest.TestCase):
    def fixture(self, epochs=20, target_kl=.01, lr=.02):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        config, _ = load_configuration()
        args = dict(config['model'], **config['algo'])
        args.update(hidden_sizes=[8, 8], ppo_epoch=epochs, target_kl=target_kl,
                    lr=lr, gamma=.99, actor_num_mini_batch=1)
        obs = Box(-1., 1., shape=(3,), dtype=np.float32)
        act = Discrete(3)
        actor = HAPPO(args, obs, act)
        buffer = OnPolicyActorBuffer(dict(episode_length=24, n_rollout_threads=1), obs, act)
        buffer.obs[:] = np.random.RandomState(5).normal(size=buffer.obs.shape)
        buffer.actions[:, 0, 0] = np.arange(24) % 3
        buffer.update_factor(np.ones_like(buffer.actions))
        with torch.no_grad():
            logp, _, _ = actor.evaluate_actions(buffer.obs[:-1, 0], buffer.actions[:, 0])
        buffer.action_log_probs[:, 0] = logp.numpy()
        advantages = np.where(buffer.actions == 0., 1., -.5).astype(np.float32)
        return actor, buffer, advantages

    def test_policy_budget_validation(self):
        for value in (True, 0, -1, 2.5, '20', None):
            with self.subTest(epochs=value), self.assertRaisesRegex(ValueError, 'positive integer'):
                validate_policy_optimization(value, None)
        for value in (True, False, 0, -1, np.nan, np.inf, 'bad'):
            with self.subTest(kl=value), self.assertRaisesRegex(ValueError, 'finite and positive'):
                validate_policy_optimization(20, value)
        self.assertEqual(validate_policy_optimization(20, None), (20, None))

    def test_early_stop_counts_actual_steps_and_averages_only_executed_updates(self):
        actor, buffer, advantages = self.fixture(target_kl=1e-7)
        old_logp = buffer.action_log_probs.copy()
        original = actor.update
        observed = []

        def update(sample):
            result = original(sample)
            observed.append([float(result[i]) for i in range(3)])
            return result

        before = actor.rollout_distribution(buffer)
        with patch.object(actor, 'update', side_effect=update):
            info = actor.train(buffer, advantages, 'EP')
        self.assertEqual(info['optimizer_steps'], 1.)
        self.assertEqual(info['ppo_epochs_completed'], 1.)
        self.assertEqual(info['kl_early_stopped'], 1.)
        self.assertGreater(info['policy_kl'], actor.target_kl)
        self.assertAlmostEqual(info['policy_kl'], actor.rollout_kl(buffer, before))
        for k, expected in zip(('policy_loss', 'dist_entropy', 'actor_grad_norm'), observed[0]):
            self.assertAlmostEqual(float(info[k]), expected, places=6)
        np.testing.assert_array_equal(buffer.action_log_probs, old_logp)

    def test_small_policy_changes_use_full_budget(self):
        actor, buffer, advantages = self.fixture(lr=1e-7)
        info = actor.train(buffer, advantages, 'EP')
        self.assertEqual(info['optimizer_steps'], 20.)
        self.assertEqual(info['ppo_epochs_completed'], 20.)
        self.assertEqual(info['kl_early_stopped'], 0.)
        self.assertLess(info['policy_kl'], .01)

    def test_disabled_limit_preserves_legacy_updates(self):
        actor, buffer, advantages = self.fixture(epochs=5, target_kl=None, lr=.0003)
        reference = copy.deepcopy(actor)
        expected = {k: 0. for k in ('policy_loss', 'dist_entropy', 'actor_grad_norm', 'ratio')}
        torch.manual_seed(9)
        for _ in range(5):
            for sample in buffer.feed_forward_generator_actor(
                    advantages, 1, objective_weights=buffer.discount_weights(.99)):
                loss, entropy, norm, ratio = reference.update(sample)
                for k, v in zip(expected, (loss, entropy, norm, ratio.mean())):
                    expected[k] += float(v)/5.
        torch.manual_seed(9)
        info = actor.train(buffer, advantages, 'EP')
        for name, value in reference.actor.state_dict().items():
            torch.testing.assert_close(actor.actor.state_dict()[name], value, rtol=0., atol=0.)
        for k, value in expected.items():
            self.assertAlmostEqual(float(info[k]), value, places=6)
        self.assertNotIn('policy_kl', info)
        self.assertEqual(info['optimizer_steps'], 5.)

    def test_exact_kl_uses_all_bins_and_excludes_forced_actions(self):
        actor, buffer, _ = self.fixture()
        buffer.active_masks[:] = 0.
        buffer.active_masks[0] = 1.
        before = torch.distributions.Categorical(probs=torch.tensor([[.2, .3, .5]]*24))
        after = torch.distributions.Categorical(probs=torch.tensor([[.4, .3, .3]] + [[.001, .998, .001]]*23))
        expected = sum(p*np.log(p/q) for p, q in zip((.2, .3, .5), (.4, .3, .3)))
        with patch.object(actor, 'rollout_distribution', return_value=after):
            self.assertAlmostEqual(actor.rollout_kl(buffer, before), expected, places=6)

    def test_kl_measurement_itself_preserves_update_and_rng(self):
        actor, buffer, advantages = self.fixture(epochs=3, target_kl=100., lr=.0003)
        reference = copy.deepcopy(actor)
        reference.target_kl = None
        torch.manual_seed(42)
        actor.train(buffer, advantages, 'EP')
        state = torch.get_rng_state().clone()
        torch.manual_seed(42)
        reference.train(buffer, advantages, 'EP')
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0., atol=0.)
        for name, value in reference.actor.state_dict().items():
            torch.testing.assert_close(actor.actor.state_dict()[name], value, rtol=0., atol=0.)

    def test_fully_forced_batch_has_no_optimizer_steps(self):
        actor, buffer, advantages = self.fixture()
        buffer.active_masks[:] = 0.
        before = copy.deepcopy(actor.actor.state_dict())
        with patch.object(actor, 'update', wraps=actor.update) as update:
            info = actor.train(buffer, advantages, 'EP')
        update.assert_not_called()
        self.assertEqual(info['optimizer_steps'], 0.)
        for name, value in before.items():
            torch.testing.assert_close(actor.actor.state_dict()[name], value, rtol=0., atol=0.)

    def test_cli_rejects_invalid_budget_before_reading_data(self):
        for option in ('--ppo-epochs=0', '--target-kl=nan', '--target-kl=0'):
            with patch('sys.argv', ['train_cascade.py', '--check-data-only', option]), \
                 patch('examples.train_cascade.CascadeData') as data, redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit):
                main()
            data.assert_not_called()

    def test_sequential_factors_use_completed_early_stopped_updates(self):
        config, _ = load_configuration()
        config['model']['hidden_sizes'] = [8, 8]
        config['algo'].update(ppo_epoch=20, target_kl=1e-8, critic_epoch=1, fixed_order=True)
        config['experiment']['rollout_episodes'] = 6
        data, physical, _ = reward_fixtures.RewardPenaltyTests().fixture()
        runner = CascadeReservoirRunner(data, physical, config, 3)
        with redirect_stdout(io.StringIO()):
            runner.collect_batch(6)
        before = []
        for actor, buffer in zip(runner.actor, runner.actor_buffer):
            with torch.no_grad():
                lp, _, _ = actor.evaluate_actions(buffer.obs[:-1, 0], buffer.actions[:, 0])
            before.append(lp.numpy().copy())
        stored = [b.action_log_probs.copy() for b in runner.actor_buffer]
        runner.prep_training()
        actors, _ = runner.train()
        factor = np.ones((runner.length, 1, 1), dtype=np.float32)
        for i, (actor, buffer) in enumerate(zip(runner.actor, runner.actor_buffer)):
            np.testing.assert_allclose(buffer.factor, factor, atol=1e-6, rtol=1e-6)
            np.testing.assert_array_equal(buffer.action_log_probs, stored[i])
            with torch.no_grad():
                lp, _, _ = actor.evaluate_actions(buffer.obs[:-1, 0], buffer.actions[:, 0])
            factor *= np.where(buffer.active_masks[:-1] > 0,
                               np.exp(lp.numpy()-before[i])[:, None], 1.)
        self.assertTrue(any(a['ppo_epochs_completed'] < 20 for a in actors))

    def test_cli_reports_opt_in_profile_and_legacy_override(self):
        config, physical = load_configuration()
        data, _, _ = reward_fixtures.RewardPenaltyTests().fixture()
        for options, epochs, kl in [(['--ppo-epochs', '20', '--target-kl', '.01'], 20, .01),
                                    (['--ppo-epochs', '5', '--no-kl-limit'], 5, None)]:
            output = io.StringIO()
            with patch('sys.argv', ['train_cascade.py', '--check-data-only']+options), \
                 patch('examples.train_cascade.load_configuration', return_value=(copy.deepcopy(config), physical)), \
                 patch('examples.train_cascade.CascadeData', return_value=data), \
                 patch('examples.train_cascade.ReservoirPhysics') as physics, \
                 patch('examples.train_cascade.validate_operating_feasibility', return_value=[]), \
                 redirect_stdout(output):
                # Supply the few scalar metadata fields consumed by check-only mode.
                physics.return_value.reference_energy = 1.
                physics.return_value.head_model = 'fixed'
                physics.return_value.head_data_status = 'test'
                main()
            metadata, _ = json.JSONDecoder().raw_decode(output.getvalue())
            self.assertEqual(metadata['ppo_epochs_max'], epochs)
            self.assertEqual(metadata['actor_target_kl'], kl)


if __name__ == '__main__':
    unittest.main()
