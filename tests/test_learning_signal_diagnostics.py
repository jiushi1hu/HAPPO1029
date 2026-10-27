"""Training-signal evidence must use complete trajectories and remain observational."""
import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from gym.spaces import Box, Discrete

from examples.compare_cascade_policy_budget import training_state_fingerprint
from examples.train_cascade import load_configuration
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.valuenorm import ValueNorm
from harl.envs.cascade_reservoir.diagnostics import (
    before_update, monte_carlo_returns, policy_evaluation_report,
)
from harl.envs.cascade_reservoir.learning_signal import learning_signal_report, signal_statistics
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
import test_cascade_reward_penalties as reward_fixtures


class LearningSignalDiagnosticTests(unittest.TestCase):
    def fixture(self, *, shared=True, gamma=.5):
        shape = (6, 1, 1)
        cb = SimpleNamespace(rewards=np.arange(1., 7., dtype=np.float32).reshape(shape),
                             masks=np.ones((7, 1, 1), dtype=np.float32),
                             bad_masks=np.ones((7, 1, 1), dtype=np.float32),
                             gamma=gamma, gae_lambda=.95)
        cb.masks[[3, 6]] = 0.
        buffers = [OnPolicyActorBuffer(dict(episode_length=6, n_rollout_threads=1),
                                      Box(-1., 1., (2,), dtype=np.float32), Discrete(3))
                   for _ in range(2)]
        for buffer in buffers:
            buffer.masks[:] = cb.masks
        buffers[0].active_masks[:-1, 0, 0] = [1., 1., 0., 1., 1., 0.]
        buffers[1].active_masks.fill(0.)
        runner = SimpleNamespace(critic_buffer=cb, actor_buffer=buffers,
                                 actor=[SimpleNamespace(use_policy_active_masks=True) for _ in buffers],
                                 algo_args={'algo': {'normalize_shared_advantages': shared}},
                                 batch_episode_starts=np.array([0, 3]),
                                 batch_episode_lengths=[3, 3], batch_years=[2012, 2013],
                                 sampled_years=[2012, 2013])
        before = dict(mc=monte_carlo_returns(cb.rewards, cb.masks, gamma),
                      prediction=np.array([3., 1., -1., 2., 0., 1.], dtype=np.float32).reshape(shape),
                      gae_targets=np.array([1., -2., 4., 10., -2., 9.], dtype=np.float32).reshape(shape))
        post_prediction = before['mc']+.5
        return runner, before, post_prediction

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

    def test_common_statistics_use_population_spread_and_empty_values_are_json_null(self):
        report = signal_statistics([1., 3., 8.])
        self.assertEqual(report['count'], 3)
        self.assertEqual(report['mean'], 4.)
        self.assertAlmostEqual(report['std'], np.std([1., 3., 8.], ddof=0))
        self.assertEqual(report['median'], 3.)
        self.assertEqual(report['min'], 1.)
        self.assertEqual(report['max'], 8.)
        self.assertEqual(report['p05'], np.percentile([1., 3., 8.], 5))
        empty = signal_statistics([])
        self.assertEqual(empty['count'], 0)
        self.assertTrue(all(value is None for key, value in empty.items() if key != 'count'))
        json.dumps(empty, allow_nan=False)
        with self.assertRaises(FloatingPointError):
            signal_statistics([np.nan])

    def test_frozen_gae_mc_gap_and_sign_disagreement_are_distinct_from_critic_fit(self):
        runner, before, post = self.fixture()
        report = learning_signal_report(runner, before, post)
        raw = before['gae_targets']-before['prediction']
        mc_advantage = before['mc']-before['prediction']
        error = before['gae_targets']-before['mc']
        self.assertEqual(report['scope'], 'training_rollout')
        self.assertEqual(report['gamma'], .5)
        self.assertEqual(report['gae_lambda'], .95)
        self.assertAlmostEqual(report['gae_target_vs_mc']['rmse'], np.sqrt(np.mean(error**2)))
        self.assertEqual(report['gae_target_vs_mc']['bias'], float(error.mean()))
        self.assertEqual(report['advantages']['nonzero_comparison_count'], 6)
        self.assertAlmostEqual(report['advantages']['sign_disagreement_fraction'], 2./6.)
        self.assertEqual(report['advantages']['gae'], signal_statistics(raw))
        self.assertEqual(report['advantages']['monte_carlo'], signal_statistics(mc_advantage))
        self.assertEqual([row['episode_reward'] for row in report['per_episode']], [2.75, 8.])
        self.assertTrue(all(row['critic_mc_post']['rmse'] == .5 for row in report['per_episode']))
        self.assertTrue(all(row['critic_mc_post']['bias'] == .5 for row in report['per_episode']))
        self.assertIn('not true advantages', report['interpretation'])
        json.dumps(report, allow_nan=False)

    def test_year_diversity_counts_annual_trajectories_instead_of_daily_rows(self):
        runner, before, post = self.fixture()
        report = learning_signal_report(runner, before, post)
        batch = report['batch']
        self.assertEqual(batch['annual_trajectories'], 2)
        self.assertEqual(batch['daily_rows'], 6)
        self.assertEqual(batch['episode_lengths'], [3, 3])
        self.assertEqual(batch['unique_year_count'], 2)
        self.assertEqual(batch['year_counts'], {'2012': 1, '2013': 1})
        self.assertEqual(batch['year_diversity_effective_count'], 2.)
        self.assertEqual(batch['trajectory_return_statistics']['count'], 2)
        self.assertNotIn('se', batch['trajectory_return_statistics'])
        runner.batch_years = [2013, 2013]
        batch = learning_signal_report(runner, before, post)['batch']
        self.assertEqual(batch['unique_year_count'], 1)
        self.assertEqual(batch['year_counts'], {'2013': 2})
        self.assertEqual(batch['year_diversity_effective_count'], 1.)
        self.assertIn('not statistical sample size', batch['diversity_definition'])

    def test_actor_summaries_match_actual_shared_and_controllable_normalization(self):
        for shared in (True, False):
            with self.subTest(shared=shared):
                runner, before, post = self.fixture(shared=shared)
                report = learning_signal_report(runner, before, post)
                actor, buffer = report['actors'][0], runner.actor_buffer[0]
                raw = before['gae_targets']-before['prediction']
                active = buffer.active_masks[:-1] > 0.
                scope_values = raw if shared else raw[active]
                expected = (raw-scope_values.mean())/(scope_values.std()+1e-5)
                weights = buffer.discount_weights(.5)*buffer.active_masks[:-1]
                objective = weights > 0.
                self.assertEqual(actor['controllable_rows'], 4)
                self.assertEqual(actor['objective_rows'], 4)
                self.assertEqual(actor['normalization_scope'],
                                 'all_common_rows' if shared else 'actor_controllable_rows')
                self.assertEqual(actor['raw_advantage'], signal_statistics(raw[objective]))
                self.assertEqual(actor['training_advantage'], signal_statistics(expected[objective]))
                self.assertEqual(actor['objective_weight_sum'], 3.)
                self.assertAlmostEqual(actor['weighted_training_advantage_mean'],
                                       float(np.sum(weights[objective].astype(float)*expected[objective])/3.))
                self.assertEqual(actor['gae_mc_sign_disagreement_fraction'], .5)
                forced = report['actors'][1]
                self.assertEqual(forced['controllable_rows'], 0)
                self.assertEqual(forced['objective_rows'], 0)
                self.assertEqual(forced['training_advantage']['count'], 0)
                self.assertIsNone(forced['weighted_training_advantage_mean'])
                self.assertIsNone(forced['gae_mc_sign_disagreement_fraction'])

    def test_zero_discount_removes_later_objective_rows_without_erasing_global_spread(self):
        runner, before, post = self.fixture(gamma=0.)
        runner.actor_buffer[0].active_masks[[0, 3]] = 0.
        report = learning_signal_report(runner, before, post)
        actor = report['actors'][0]
        self.assertEqual(actor['controllable_rows'], 2)
        self.assertEqual(actor['objective_rows'], 0)
        self.assertEqual(actor['objective_weight_sum'], 0.)
        self.assertIsNone(actor['weighted_training_advantage_mean'])
        self.assertEqual(report['advantages']['gae']['count'], 6)
        before['gae_targets'][:] = before['prediction']
        report = learning_signal_report(runner, before, post)
        self.assertEqual(report['advantages']['nonzero_comparison_count'], 0)
        self.assertIsNone(report['advantages']['sign_disagreement_fraction'])

    def test_invalid_shape_incomplete_trajectory_truncation_and_metadata_are_rejected(self):
        mutations = dict(
            nonterminal=lambda r, b, p: r.critic_buffer.masks.__setitem__(-1, 1.),
            truncated=lambda r, b, p: r.critic_buffer.bad_masks.__setitem__(2, 0.),
            starts=lambda r, b, p: setattr(r, 'batch_episode_starts', np.array([0, 2])),
            lengths=lambda r, b, p: setattr(r, 'batch_episode_lengths', [2, 4]),
            years=lambda r, b, p: setattr(r, 'batch_years', [2012]),
            shape=lambda r, b, p: b.__setitem__('mc', b['mc'][:-1]),
            masks=lambda r, b, p: r.critic_buffer.masks.__setitem__(1, .5),
        )
        for key, mutate in mutations.items():
            with self.subTest(key=key):
                runner, before, post = self.fixture()
                mutate(runner, before, post)
                with self.assertRaises(ValueError):
                    learning_signal_report(runner, before, post)
        for key in ('mc', 'prediction', 'gae_targets'):
            runner, before, post = self.fixture()
            before[key][0] = np.nan
            with self.subTest(key=key), self.assertRaises(FloatingPointError):
                learning_signal_report(runner, before, post)

    def test_actor_discount_boundaries_must_agree_with_the_critic_trajectories(self):
        runner, before, post = self.fixture()
        runner.actor_buffer[0].masks[3] = 1.
        with self.assertRaisesRegex(ValueError, 'actor and critic episode boundaries must match'):
            learning_signal_report(runner, before, post)
        runner.actor_buffer[0].masks[:] = runner.critic_buffer.masks
        runner.actor_buffer[1].masks = runner.actor_buffer[1].masks[:-1]
        with self.assertRaisesRegex(ValueError, 'actor and critic episode boundaries must match'):
            learning_signal_report(runner, before, post)

    def test_target_estimator_identifies_the_buffer_setting_before_config_fallback(self):
        for buffer_setting, config_setting, expected in ((False, True, False), (True, False, True),
                                                          (None, False, False), (None, None, True)):
            with self.subTest(buffer=buffer_setting, configured=config_setting):
                runner, before, post = self.fixture()
                if buffer_setting is not None:
                    runner.critic_buffer.use_gae = buffer_setting
                if config_setting is not None:
                    runner.algo_args['algo']['use_gae'] = config_setting
                report = learning_signal_report(runner, before, post)
                self.assertIs(report['use_gae'], expected)
                self.assertEqual(report['target_estimator'], 'gae' if expected else 'discounted_return')

    def test_lambda_one_matches_complete_mc_returns_with_valuenorm_and_internal_boundaries(self):
        cfg, _ = load_configuration()
        args = {**cfg['train'], **cfg['model'], **cfg['algo'], 'episode_length': 6,
                'n_rollout_threads': 1, 'gamma': .5, 'gae_lambda': 1.}
        cb = OnPolicyCriticBufferEP(args, Box(-np.inf, np.inf, (3,), dtype=np.float32))
        cb.masks[[3, 6]] = 0.
        cb.rewards[:, 0, 0] = np.arange(1., 7.)
        norm = ValueNorm(1)
        norm.update(np.array([[-10.], [2.], [30.]], dtype=np.float32))
        expected = np.array([2.75, 3.5, 3., 8., 8., 6.])
        results = []
        for values, terminal in (([3., -2., 8., 1., 7., 9.], 1234.),
                                 ([-8., 20., 30., 5., -7., 2.], -4321.)):
            physical_values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
            cb.value_preds[:-1, 0] = norm.normalize(physical_values).numpy()
            cb.compute_returns(np.array([[terminal]], dtype=np.float32), norm)
            np.testing.assert_allclose(cb.returns[:-1, 0, 0], expected, rtol=0, atol=4e-6)
            results.append(cb.returns[:-1].copy())
        np.testing.assert_allclose(results[0], results[1], rtol=0, atol=4e-6)
        cb.gae_lambda = .95
        cb.compute_returns(np.array([[1000.]], dtype=np.float32), norm)
        self.assertGreater(float(np.max(np.abs(cb.returns[:-1, 0, 0]-expected))), .01)

    def test_enabling_diagnostics_preserves_models_adam_gradients_normalizer_buffers_and_rng(self):
        outcomes = []
        for enabled in (False, True):
            cfg, _ = load_configuration()
            cfg['model']['hidden_sizes'] = [8, 8]
            cfg['algo'].update(ppo_epoch=1, critic_epoch=1)
            cfg['experiment'].update(fixed_training_year=1965, rollout_episodes=2,
                                     diagnostics=enabled, diagnostic_stochastic_repeats=2)
            data, physical, _ = reward_fixtures.RewardPenaltyTests().fixture()
            runner = CascadeReservoirRunner(data, physical, cfg, 9)
            with redirect_stdout(io.StringIO()):
                runner.run(2, 1)
            states = dict(fingerprint=training_state_fingerprint(runner),
                          actor_gradients=[[None if p.grad is None else p.grad.clone()
                                            for p in actor.actor.parameters()] for actor in runner.actor],
                          critic_gradients=[None if p.grad is None else p.grad.clone()
                                            for p in runner.critic.critic.parameters()],
                          actor_buffers=[copy.deepcopy(vars(buffer)) for buffer in runner.actor_buffer],
                          critic_buffer=copy.deepcopy(vars(runner.critic_buffer)))
            if enabled:
                signal = runner.diagnostics_history[-1]['learning_signal']
                self.assertEqual(signal['batch']['annual_trajectories'], 2)
                self.assertEqual(signal['batch']['daily_rows'], 10)
                self.assertEqual(signal['batch']['year_counts'], {'1965': 2})
                json.dumps(signal, allow_nan=False)
                snapshot = before_update(runner)
                self.assertIn('gae_targets', snapshot)
                original = snapshot['gae_targets'].copy()
                runner.critic_buffer.returns[0] += 1.
                np.testing.assert_array_equal(snapshot['gae_targets'], original)
            outcomes.append(states)
        self.assert_state_equal(outcomes[0], outcomes[1])

    def test_training_year_policy_reports_use_only_explicit_training_years(self):
        runner = SimpleNamespace(algo_args={'experiment': {'diagnostic_stochastic_repeats': 2},
                                           'algo': {'gamma': .99}},
                                 actor=[SimpleNamespace(actor=torch.nn.Linear(1, 1))],
                                 data=SimpleNamespace(training_years=(2012, 2013), validation_years=(2020,)),
                                 physical_config={}, seed=3, iteration=0, update_count=0, training_steps=0)
        rows = [dict(year=year, episode_reward=float(i), energy_TWh=100.+i)
                for i, year in enumerate((2012, 2013))]
        calls = []

        def evaluate(*args, **kwargs):
            calls.append(args[3])
            runner.actor[0].actor.eval()
            return dict(episode_reward=float(torch.rand(())), energy_TWh=100.)

        state = torch.get_rng_state().clone()
        with patch('harl.envs.cascade_reservoir.diagnostics.evaluate_policy', side_effect=evaluate), \
                redirect_stdout(io.StringIO()):
            report = policy_evaluation_report(runner, rows, scope='training_years')
        self.assertEqual(calls, [2012, 2013, 2012, 2013])
        self.assertEqual(report['scope'], 'training_years')
        self.assertIn('in-sample training years', report['evaluation_scope'])
        self.assertIn('no generalization claim', report['evaluation_scope'])
        self.assertEqual(report['sampled_repeats'], 2)
        self.assertTrue(runner.actor[0].actor.training)
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
        for bad_rows in ([dict(rows[0], year=2020)], [rows[0], rows[0]],
                         [dict(episode_reward=1., energy_TWh=100.)]):
            with self.subTest(rows=bad_rows), \
                    patch('harl.envs.cascade_reservoir.diagnostics.evaluate_policy') as evaluate, \
                    self.assertRaises(ValueError):
                policy_evaluation_report(runner, bad_rows, scope='training_years')
            evaluate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
