"""Exact surrogate endpoints and observation-only policy diagnosis."""
import copy
import io
import json
import unittest
from contextlib import redirect_stdout

import numpy as np
import torch
from gym.spaces import Box, Discrete

from examples.train_cascade import load_configuration
from harl.algorithms.actors.happo import HAPPO
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
import test_reward_scale_metrics as reward_metric_tests


class SurrogateDiagnosticTests(unittest.TestCase):
    def fixture(self, gamma=.5):
        cfg, _ = load_configuration()
        args = {**cfg['model'], **cfg['algo'], 'hidden_sizes': [8, 8],
                'gamma': gamma, 'entropy_coef': .03, 'ppo_epoch': 2}
        torch.set_num_threads(1)
        actor = HAPPO(args, Box(-1., 1., (1,), dtype=np.float32), Discrete(3))
        with torch.no_grad():
            for p in actor.actor.parameters():
                p.zero_()
            actor.actor.act.action_out.linear.bias.copy_(torch.tensor([-.3, .2, .1]))
        buffer = OnPolicyActorBuffer({**cfg['train'], **args, 'episode_length': 4},
                                     actor.obs_space, actor.act_space)
        buffer.actions[:, 0, 0] = [0, 1, 2, 0]
        buffer.active_masks[1] = 0.
        buffer.masks[2] = buffer.masks[4] = 0.
        buffer.update_factor(np.array([1., 4., 2., .5], dtype=np.float32)[:, None, None])
        with torch.no_grad():
            logp, _, _ = actor.evaluate_actions(buffer.obs[:-1, 0], buffer.actions[:, 0])
        buffer.action_log_probs[:, 0] = logp.numpy()
        advantages = np.array([2., -500., -3., 4.], dtype=np.float32)[:, None, None]
        return actor, buffer, advantages

    def assert_state_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_state_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_state_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_endpoints_match_independent_masked_discounted_sequential_objective(self):
        actor, buffer, advantages = self.fixture()
        logits = np.array([-.1, .9, -.2])
        with torch.no_grad():
            actor.actor.act.action_out.linear.bias.copy_(torch.tensor(logits, dtype=torch.float32))
        weights = buffer.discount_weights(.5)
        np.testing.assert_array_equal(weights[:, 0, 0], [1., .5, 1., .5])
        measured = actor.surrogate_metrics(buffer, advantages, weights)
        probabilities = np.exp(logits)/np.exp(logits).sum()
        actions = buffer.actions[:, 0, 0].astype(int)
        ratio = probabilities[actions]/np.exp(buffer.action_log_probs[:, 0, 0])
        advantage = advantages[:, 0, 0]
        weighted_factor = np.array([1., 0., 1., .5])*buffer.factor[:, 0, 0]
        expected = np.mean(weighted_factor*np.minimum(
            ratio*advantage, np.clip(ratio, .8, 1.2)*advantage))
        entropy = -np.sum(probabilities*np.log(probabilities))*2.5/4.
        self.assertAlmostEqual(measured['clipped_surrogate'], expected, places=6)
        self.assertAlmostEqual(measured['unclipped_surrogate'],
                               np.mean(weighted_factor*ratio*advantage), places=6)
        self.assertAlmostEqual(measured['weighted_entropy'], entropy, places=6)
        self.assertAlmostEqual(measured['regularized_objective'], expected+.03*entropy, places=6)

    def test_train_records_actual_normalized_advantages_and_fixed_factor(self):
        actor, buffer, raw = self.fixture()
        actor.args['normalize_shared_advantages'] = False
        active = raw[buffer.active_masks[:-1] > 0.]
        normalized = (raw-active.mean())/(active.std()+1e-5)
        expected_before = actor.surrogate_metrics(buffer, normalized, buffer.discount_weights(.5))
        factor_before = buffer.factor.copy()
        info = actor.train(buffer, raw, 'EP', diagnostics=True)
        expected_after = actor.surrogate_metrics(buffer, normalized, buffer.discount_weights(.5))
        for key in expected_before:
            self.assertEqual(info[key+'_before'], expected_before[key])
            self.assertEqual(info[key+'_after'], expected_after[key])
            self.assertEqual(info[key+'_delta'], expected_after[key]-expected_before[key])
        np.testing.assert_array_equal(buffer.factor, factor_before)

    def test_measurements_preserve_parameters_adam_gradients_buffers_and_rng(self):
        actor, buffer, advantages = self.fixture()
        actor.actor_num_mini_batch = 2
        actor.train(buffer, advantages, 'EP')  # Exercise existing Adam momentum.
        with torch.no_grad():
            logp, _, _ = actor.evaluate_actions(buffer.obs[:-1, 0], buffer.actions[:, 0])
        buffer.action_log_probs[:, 0] = logp.numpy()
        outcomes = []
        for enabled in (False, True):
            current, rollout = copy.deepcopy((actor, buffer))
            torch.manual_seed(901)
            info = current.train(rollout, advantages.copy(), 'EP', diagnostics=enabled)
            outcomes.append((current.actor.state_dict(), current.actor_optimizer.state_dict(),
                [p.grad for p in current.actor.parameters()], vars(rollout), torch.get_rng_state()))
            self.assertEqual('clipped_surrogate_delta' in info, enabled)
        self.assert_state_equal(outcomes[0], outcomes[1])

    def test_forced_and_zero_discount_objectives_are_zero_without_adam_steps(self):
        for forced in (True, False):
            actor, buffer, advantages = self.fixture(gamma=0.)
            buffer.masks.fill(1.)
            buffer.active_masks[0] = 0.
            if forced:
                buffer.active_masks.fill(0.)
            before = copy.deepcopy(actor.actor_optimizer.state_dict())
            info = actor.train(buffer, advantages, 'EP', diagnostics=True)
            self.assertEqual(info['optimizer_steps'], 0.)
            for key in ('clipped_surrogate', 'unclipped_surrogate',
                        'weighted_entropy', 'regularized_objective'):
                for suffix in ('before', 'after', 'delta'):
                    self.assertEqual(info[key+'_'+suffix], 0.)
            self.assert_state_equal(before, actor.actor_optimizer.state_dict())

    def test_runner_persists_three_measurements_without_changing_learning(self):
        outcomes = []
        for enabled in (False, True):
            data, physical, config = reward_metric_tests.RewardScaleMetricsTests().fixture()
            config['experiment'].update(fixed_training_year=1965, episodes=4, rollout_episodes=2,
                diagnostics=enabled, diagnostic_eval_episodes=[0, 2, 4], diagnostic_stochastic_repeats=2)
            stream = io.StringIO()
            runner = CascadeReservoirRunner(data, physical, config, 7, episode_log=stream)
            with redirect_stdout(io.StringIO()):
                runner.run(4, 1)
            outcomes.append(([a.actor.state_dict() for a in runner.actor], runner.critic.critic.state_dict(),
                [a.actor_optimizer.state_dict() for a in runner.actor], runner.critic.critic_optimizer.state_dict(),
                runner.value_normalizer.state_dict(), torch.get_rng_state(), runner.year_rng.get_state(),
                [b.actions.copy() for b in runner.actor_buffer], runner.critic_buffer.returns.copy()))
            if enabled:
                records = [json.loads(line) for line in stream.getvalue().splitlines()]
                updates = [r for r in records if r['record_type'] == 'training_diagnostics']
                policies = [r for r in records if r['record_type'] == 'policy_evaluation_diagnostics']
                self.assertEqual(len(updates), 2)
                self.assertEqual([r['episode'] for r in policies], [0, 2, 4])
                self.assertEqual(policies[-1]['scope'], 'fixed_training_year')
                self.assertTrue(policies[-1]['comparison_to_previous']['available'])
                for update in updates:
                    for row in update['actors']:
                        self.assertEqual(len(row['surrogate']), 12)
                        self.assertTrue(all(np.isfinite(v) for v in row['surrogate'].values()))
        self.assert_state_equal(outcomes[0], outcomes[1])


if __name__ == '__main__':
    unittest.main()
