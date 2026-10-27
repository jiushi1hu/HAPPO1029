"""Matched-budget KL experiments must isolate the guard and retain paired evidence."""
import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import examples.compare_cascade_policy_budget as comparison
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
import test_reward_scale_metrics as reward_metric_tests


class CascadeKLComparisonTests(unittest.TestCase):
    def fixture(self, target_kl=None):
        data, physical, config = reward_metric_tests.RewardScaleMetricsTests().fixture()
        data.hashes = {'fixture': 'short-five-day-cascade'}
        config['experiment'].update(fixed_training_year=1965, rollout_episodes=2,
                                   episodes=4, diagnostics=False)
        config['algo'].update(ppo_epoch=5, target_kl=target_kl)
        return data, physical, config

    @staticmethod
    def arm(guarded=False, repeats=3):
        baseline_rewards = [-3., 1., 5.][:repeats]
        rewards = [-4., 3., 10.][:repeats] if guarded else baseline_rewards
        return dict(arm='guarded' if guarded else 'baseline', seed=7,
            initial_training_state_hash='common-initial',
            final_training_state_hash='guarded-final' if guarded else 'baseline-final',
            final_actor_state_hash='guarded-actors' if guarded else 'baseline-actors',
            final_policy_report=dict(seed=7, gamma=.99, scope='fixed_training_year',
                evaluation_years=[1965], sampled_repeats=repeats,
                evaluation_seeds=[[7000+i] for i in range(repeats)],
                stochastic_repeat_means_reward=rewards,
                stochastic_mean_reward=float(np.mean(rewards)),
                deterministic_mean_reward=6. if guarded else 4.,
                stochastic_repeat_means_TWh=[100.+i for i in range(repeats)],
                stochastic_mean_TWh=101., deterministic_mean_TWh=102.,
                episode=4, update=2, training_steps=20),
            actor_optimizer_steps=[8., 10., 10., 10., 10.] if guarded else [10.]*5,
            actor_epochs_saved=[2., 0., 0., 0., 0.] if guarded else [0.]*5,
            kl_early_stops=[1., 0., 0., 1., 0.] if guarded else [0.]*5,
            training_steps=20)

    def test_matched_profile_keeps_five_epochs_and_legacy_budget_remains_available(self):
        self.assertEqual(comparison.resolve_candidate_epochs(None, True), 5)
        self.assertEqual(comparison.resolve_candidate_epochs(5, True), 5)
        self.assertEqual(comparison.resolve_candidate_epochs(None, False), 20)
        self.assertEqual(comparison.resolve_candidate_epochs(12, False), 12)
        self.assertEqual(comparison.resolve_candidate_epochs(np.int64(5), True), 5)
        for requested, matched in ((20, True), (4, True), (0, False), (-1, False),
                                   (True, True), (2.5, False), ('5', True)):
            with self.subTest(requested=requested, matched=matched), self.assertRaises(ValueError):
                comparison.resolve_candidate_epochs(requested, matched)

    def test_fingerprint_is_observational_and_ignores_the_deliberately_different_kl_setting(self):
        hashes = []
        for target in (None, .01):
            data, physical, config = self.fixture(target)
            runner = CascadeReservoirRunner(data, physical, config, 7)
            before = torch.get_rng_state().clone()
            digest = comparison.training_state_fingerprint(runner)
            self.assertEqual(digest, comparison.training_state_fingerprint(runner))
            torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
            hashes.append(digest)
        self.assertEqual(hashes[0], hashes[1])

    def test_fingerprint_covers_networks_optimizers_normalizer_rng_modes_and_counters(self):
        def actor_parameter(r):
            with torch.no_grad():
                next(r.actor[0].actor.parameters()).add_(.25)

        def critic_parameter(r):
            with torch.no_grad():
                next(r.critic.critic.parameters()).add_(.25)

        mutators = dict(
            actor=actor_parameter, critic=critic_parameter,
            actor_optimizer=lambda r: r.actor[0].actor_optimizer.param_groups[0].update(lr=.02),
            critic_optimizer=lambda r: r.critic.critic_optimizer.param_groups[0].update(lr=.02),
            normalizer=lambda r: r.value_normalizer.update(np.array([[1.], [4.]], np.float32)),
            python_rng=lambda r: random.random(), numpy_rng=lambda r: np.random.rand(),
            torch_rng=lambda r: torch.rand(()), year_rng=lambda r: r.year_rng.rand(),
            actor_mode=lambda r: r.actor[0].actor.eval(), critic_mode=lambda r: r.critic.critic.eval(),
            iteration=lambda r: setattr(r, 'iteration', r.iteration+1),
            update_count=lambda r: setattr(r, 'update_count', r.update_count+1),
            training_steps=lambda r: setattr(r, 'training_steps', r.training_steps+5))
        for name, mutate in mutators.items():
            with self.subTest(component=name):
                data, physical, config = self.fixture()
                runner = CascadeReservoirRunner(data, physical, config, 7)
                before = comparison.training_state_fingerprint(runner)
                mutate(runner)
                self.assertNotEqual(before, comparison.training_state_fingerprint(runner))

    def test_paired_result_uses_repeat_differences_and_separates_state_equivalence(self):
        baseline, guarded = self.arm(), self.arm(True)
        result = comparison.paired_kl_result(baseline, guarded)
        self.assertEqual(result['record_type'], 'paired_kl_comparison')
        self.assertEqual(result['seed'], 7)
        self.assertTrue(result['initial_state_match'])
        self.assertFalse(result['final_state_match'])
        self.assertFalse(result['final_actor_state_match'])
        self.assertEqual(result['deterministic_reward_difference'], 2.)
        np.testing.assert_array_equal(result['sampled_repeat_reward_differences'], [-1., 2., 5.])
        self.assertEqual(result['sampled_reward_difference_mean'], 2.)
        self.assertEqual(result['sampled_reward_difference_sd'], 3.)
        self.assertAlmostEqual(result['sampled_reward_difference_se'], np.sqrt(3.))
        self.assertEqual(result['guarded_epochs_saved_by_actor'], [2., 0., 0., 0., 0.])
        self.assertEqual(result['guarded_threshold_crossings_by_actor'], [1., 0., 0., 1., 0.])
        self.assertFalse(result['no_optimizer_steps_saved'])

    def test_last_epoch_crossing_can_save_no_steps_and_single_repeat_has_no_se(self):
        baseline, guarded = self.arm(repeats=1), self.arm(True, repeats=1)
        guarded.update(final_training_state_hash=baseline['final_training_state_hash'],
                       final_actor_state_hash=baseline['final_actor_state_hash'],
                       actor_optimizer_steps=[10.]*5, actor_epochs_saved=[0.]*5,
                       kl_early_stops=[1., 0., 0., 0., 0.])
        result = comparison.paired_kl_result(baseline, guarded)
        self.assertTrue(result['final_state_match'])
        self.assertTrue(result['final_actor_state_match'])
        self.assertTrue(result['no_optimizer_steps_saved'])
        self.assertEqual(result['guarded_threshold_crossings_by_actor'][0], 1.)
        self.assertIsNone(result['sampled_reward_difference_sd'])
        self.assertIsNone(result['sampled_reward_difference_se'])

    def test_paired_result_rejects_mismatched_sampling_grids_and_scopes(self):
        changes = [('gamma', 1.), ('scope', 'validation'), ('seed', 8),
                   ('evaluation_years', [1966]), ('evaluation_seeds', [[1], [2], [3]]),
                   ('sampled_repeats', 2)]
        for key, value in changes:
            with self.subTest(field=key):
                baseline, guarded = self.arm(), self.arm(True)
                guarded['final_policy_report'][key] = value
                with self.assertRaises(ValueError):
                    comparison.paired_kl_result(baseline, guarded)
        baseline, guarded = self.arm(), self.arm(True)
        guarded['seed'] = 8
        with self.assertRaises(ValueError):
            comparison.paired_kl_result(baseline, guarded)

    def test_paired_result_rejects_missing_grid_fields_and_malformed_repeat_arrays(self):
        for values in ([1.], [], [[1.], [2.], [3.]], 1., [1., np.nan, 3.], [1., 2., np.inf]):
            with self.subTest(values=values):
                baseline, guarded = self.arm(), self.arm(True)
                guarded['final_policy_report']['stochastic_repeat_means_reward'] = values
                with self.assertRaises(ValueError):
                    comparison.paired_kl_result(baseline, guarded)
        for count in (True, 0, -1, 3., '3'):
            with self.subTest(repeat_count=count):
                baseline, guarded = self.arm(), self.arm(True)
                for arm in (baseline, guarded):
                    arm['final_policy_report']['sampled_repeats'] = count
                with self.assertRaises(ValueError):
                    comparison.paired_kl_result(baseline, guarded)
        baseline, guarded = self.arm(), self.arm(True)
        for arm in (baseline, guarded):
            del arm['final_policy_report']['evaluation_seeds']
        with self.assertRaises(ValueError):
            comparison.paired_kl_result(baseline, guarded)

    def test_summary_uses_training_seed_differences_not_pooled_action_repeats(self):
        pairs = []
        for i, difference in enumerate((1., 3., 5.)):
            pair = comparison.paired_kl_result(self.arm(), self.arm(True))
            pair.update(seed=i+1, sampled_reward_difference_mean=difference,
                        deterministic_reward_difference=difference+1.,
                        guarded_epochs_saved_by_actor=[float(i), 0., 0., 0., 0.],
                        guarded_threshold_crossings_by_actor=[1., 0., 0., 0., 0.])
            pairs.append(pair)
        summary = comparison.kl_comparison_summary(pairs)
        self.assertEqual(summary['record_type'], 'paired_kl_summary')
        self.assertEqual(summary['seeds'], [1, 2, 3])
        sampled = summary['sampled_reward_difference_across_seeds']
        self.assertEqual(sampled['mean'], 3.)
        self.assertEqual(sampled['sd'], 2.)
        self.assertAlmostEqual(sampled['se'], 2./np.sqrt(3.))
        self.assertEqual(summary['deterministic_reward_difference_across_seeds']['mean'], 4.)
        self.assertEqual(summary['total_guarded_epochs_saved'], 3)
        self.assertEqual(summary['total_guarded_threshold_crossings'], 3)
        self.assertTrue(summary['all_initial_states_match'])
        self.assertFalse(summary['all_final_states_match'])
        self.assertFalse(summary['all_final_actor_states_match'])
        single = comparison.kl_comparison_summary(pairs[:1])
        self.assertIsNone(single['sampled_reward_difference_across_seeds']['sd'])
        self.assertIsNone(single['sampled_reward_difference_across_seeds']['se'])
        for invalid in ([], [pairs[0], pairs[0]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                comparison.kl_comparison_summary(invalid)

    def test_kl_only_cli_runs_equal_budgets_and_persists_step1_diagnostics(self):
        data, physical, config = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'matched.jsonl'
            argv = ['compare_cascade_policy_budget.py', '--kl-only', '--year', '1965',
                    '--updates', '2', '--rollout-episodes', '2', '--seeds', '1',
                    '--evaluation-repeats', '2', '--output', str(path)]
            with patch('sys.argv', argv), \
                    patch.object(comparison, 'load_configuration', return_value=(config, physical)), \
                    patch.object(comparison, 'CascadeData', return_value=data), \
                    patch.object(comparison, 'validate_operating_feasibility', return_value=[1965]) as preflight, \
                    redirect_stdout(io.StringIO()):
                comparison.main()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            metadata = next(r for r in rows if r['record_type'] == 'paired_check_configuration')
            self.assertEqual(metadata['arms']['baseline'], dict(ppo_epochs=5, target_kl=None))
            self.assertEqual(metadata['arms']['guarded'], dict(ppo_epochs=5, target_kl=.01))
            arms = [r for r in rows if r.get('arm') in ('baseline', 'guarded') and 'final_policy_report' in r]
            self.assertEqual(len(arms), 2)
            self.assertEqual({r['arm'] for r in arms}, {'baseline', 'guarded'})
            for arm in arms:
                self.assertEqual(arm['training_steps'], 20)
                self.assertEqual(arm['final_policy_report']['sampled_repeats'], 2)
                self.assertEqual(arm['final_policy_report']['scope'], 'fixed_training_year')
                self.assertEqual(arm['final_policy_report']['evaluation_years'], [1965])
                self.assertEqual(len(arm['training_diagnostics']), 2)
                self.assertEqual(len(arm['training_history']), 2)
                for update in arm['training_diagnostics']:
                    self.assertEqual(len(update['actors']), 5)
                    for actor in update['actors']:
                        self.assertEqual(len(actor['surrogate']), 12)
                        self.assertTrue(np.isfinite(actor['surrogate']['clipped_surrogate_delta']))
            events = [r for r in rows if r['record_type'] == 'training_diagnostics']
            self.assertEqual(len(events), 4)
            for name in ('baseline', 'guarded'):
                self.assertEqual(sum(r['arm'] == name for r in events), 2)
            paired = next(r for r in rows if r['record_type'] == 'paired_kl_comparison')
            self.assertTrue(paired['initial_state_match'])
            if paired['no_optimizer_steps_saved']:
                self.assertTrue(paired['final_state_match'])
                self.assertTrue(paired['final_actor_state_match'])
                np.testing.assert_array_equal(paired['sampled_repeat_reward_differences'], [0., 0.])
                self.assertEqual(paired['deterministic_reward_difference'], 0.)
            preflight.assert_called_once_with(data, physical, years=[1965])
            with patch('sys.argv', argv), patch.object(comparison, 'CascadeData') as loader, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                comparison.main()
            loader.assert_not_called()

    def test_legacy_cli_retains_five_versus_twenty_epoch_budget_comparison(self):
        data, physical, config = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'legacy.jsonl'
            argv = ['compare_cascade_policy_budget.py', '--year', '1965', '--updates', '1',
                    '--rollout-episodes', '1', '--seeds', '1', '--evaluation-repeats', '1',
                    '--output', str(path)]
            with patch('sys.argv', argv), \
                    patch.object(comparison, 'load_configuration', return_value=(config, physical)), \
                    patch.object(comparison, 'CascadeData', return_value=data), \
                    patch.object(comparison, 'validate_operating_feasibility', return_value=[1965]), \
                    redirect_stdout(io.StringIO()):
                comparison.main()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        metadata = rows[0]
        self.assertEqual(metadata['record_type'], 'paired_check_configuration')
        self.assertEqual(metadata['comparison_type'], 'optimization_budget')
        self.assertEqual(metadata['arms']['baseline'], dict(ppo_epochs=5, target_kl=None))
        self.assertEqual(metadata['arms']['guarded'], dict(ppo_epochs=20, target_kl=.01))
        results = rows[1:]
        self.assertEqual([r['arm'] for r in results], ['baseline', 'guarded'])
        for result in results:
            self.assertEqual(result['record_type'], 'paired_check_result')
            self.assertIn('initial_sampled', result)
            self.assertIn('final_sampled', result)
            self.assertEqual(len(result['initial_sampled']['repeat_rewards']), 1)
            self.assertEqual(len(result['final_sampled']['repeat_rewards']), 1)
            self.assertEqual(result['training_steps'], 5)
        self.assertFalse(any(r['record_type'].startswith('paired_kl') for r in rows))

    def test_kl_only_rejects_a_twenty_epoch_candidate_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'invalid.jsonl'
            argv = ['compare_cascade_policy_budget.py', '--kl-only',
                    '--candidate-ppo-epochs', '20', '--output', str(path)]
            with patch('sys.argv', argv), patch.object(comparison, 'CascadeData') as loader, \
                    patch.object(comparison, 'load_configuration') as configuration, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                comparison.main()
            self.assertFalse(path.exists())
        loader.assert_not_called()
        configuration.assert_not_called()


if __name__ == '__main__':
    unittest.main()
