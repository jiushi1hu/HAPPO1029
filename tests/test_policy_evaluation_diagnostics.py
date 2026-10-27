"""Paired policy evaluations must remain observational and statistically explicit."""
import copy
import io
import json
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from harl.envs.cascade_reservoir.diagnostics import (
    policy_evaluation_report, validation_report,
)


class PolicyEvaluationDiagnosticTests(unittest.TestCase):
    @staticmethod
    def runner(repeats=3):
        return SimpleNamespace(
            algo_args={'experiment': {'diagnostic_stochastic_repeats': repeats},
                       'algo': {'gamma': .99}},
            actor=[SimpleNamespace(actor=torch.nn.Linear(1, 1))],
            data=SimpleNamespace(validation_years=(2020, 2021)), physical_config={},
            seed=2, iteration=0, update_count=0, training_steps=0,
            fixed_training_year=2013, offset=0.,
        )

    @staticmethod
    def deterministic_rows(years=(2020, 2021), offset=0.):
        return [dict(year=year, episode_reward=10.+offset+j,
                     energy_TWh=100.+offset+j) for j, year in enumerate(years)]

    @staticmethod
    def evaluation(runner, calls):
        def evaluate(actors, data, physical, year, seed, episode, steps, **kwargs):
            calls.append((year, episode, kwargs))
            actors[0].actor.eval()  # Match the real evaluator's mode preparation.
            draw = float(torch.rand(()))
            return dict(episode_reward=year/1000.+draw+runner.offset*(1.+draw),
                        energy_TWh=year/10.+draw+runner.offset,
                        episode_penalty=.5+draw)
        return evaluate

    def test_repeat_grid_retains_per_year_values_and_preserves_rng_parameters_and_mode(self):
        runner, calls = self.runner(), []
        parameters = copy.deepcopy(runner.actor[0].actor.state_dict())
        rng = torch.get_rng_state().clone()
        rows = self.deterministic_rows()
        with patch('harl.envs.cascade_reservoir.diagnostics.evaluate_policy',
                   side_effect=self.evaluation(runner, calls)), redirect_stdout(io.StringIO()):
            first = policy_evaluation_report(runner, rows)
            second = policy_evaluation_report(runner, rows)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 12)
        self.assertEqual(first['evaluation_years'], [2020, 2021])
        self.assertEqual(first['sampled_repeats'], 3)
        self.assertEqual(first['evaluation_seeds'], [
            [1202020, 1202021], [1212020, 1212021], [1222020, 1222021]])
        means = np.array([[r['episode_reward'] for r in year['repeats']]
                          for year in first['sampled_results_by_year']]).mean(axis=0)
        np.testing.assert_array_equal(first['stochastic_repeat_means_reward'], means)
        self.assertAlmostEqual(first['stochastic_repeat_sd_reward'], means.std(ddof=1))
        self.assertAlmostEqual(first['stochastic_repeat_se_reward'], means.std(ddof=1)/np.sqrt(3))
        self.assertAlmostEqual(first['deterministic_minus_sampled_reward'], 10.5-means.mean())
        self.assertFalse(first['comparison_to_previous']['available'])
        self.assertTrue(all(call[2] == {'deterministic': False, 'gamma': .99} for call in calls))
        self.assertIn('not uncertainty across training seeds', first['uncertainty_scope'])
        self.assertTrue(runner.actor[0].actor.training)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        for key, value in parameters.items():
            torch.testing.assert_close(value, runner.actor[0].actor.state_dict()[key], rtol=0, atol=0)
        json.dumps(first, allow_nan=False)

    def test_checkpoint_deltas_use_paired_repeat_means_and_same_action_seeds(self):
        runner, calls = self.runner(), []
        with patch('harl.envs.cascade_reservoir.diagnostics.evaluate_policy',
                   side_effect=self.evaluation(runner, calls)), redirect_stdout(io.StringIO()):
            before = policy_evaluation_report(runner, self.deterministic_rows())
            runner.iteration, runner.update_count, runner.training_steps, runner.offset = 6, 1, 2191, 2.
            after = policy_evaluation_report(runner, self.deterministic_rows(offset=3.), before)
        delta = np.asarray(after['stochastic_repeat_means_reward'])-before['stochastic_repeat_means_reward']
        comparison = after['comparison_to_previous']
        self.assertTrue(comparison['available'])
        self.assertEqual(comparison['previous_episode'], 0)
        self.assertEqual(comparison['previous_update'], 0)
        self.assertEqual(after['evaluation_seeds'], before['evaluation_seeds'])
        np.testing.assert_array_equal(comparison['stochastic_repeat_reward_deltas'], delta)
        self.assertAlmostEqual(comparison['stochastic_reward_delta_mean'], delta.mean())
        self.assertAlmostEqual(comparison['stochastic_reward_delta_sd'], delta.std(ddof=1))
        self.assertAlmostEqual(comparison['stochastic_reward_delta_se'], delta.std(ddof=1)/np.sqrt(3))
        self.assertEqual(comparison['deterministic_reward_delta'], 3.)
        self.assertEqual(comparison['deterministic_energy_TWh_delta'], 3.)
        np.testing.assert_allclose(comparison['stochastic_repeat_energy_TWh_deltas'], 2.)

    def test_unmatched_evaluations_do_not_claim_paired_deltas(self):
        runner, calls = self.runner(), []
        with patch('harl.envs.cascade_reservoir.diagnostics.evaluate_policy',
                   side_effect=self.evaluation(runner, calls)), redirect_stdout(io.StringIO()):
            before = policy_evaluation_report(runner, self.deterministic_rows())
            for key, value in [('seed', 3), ('gamma', 1.), ('scope', 'fixed_training_year'),
                               ('evaluation_years', [2021, 2020]), ('sampled_repeats', 2),
                               ('evaluation_seeds', [[0, 0]]*3)]:
                with self.subTest(key=key):
                    different = dict(before, **{key: value})
                    report = policy_evaluation_report(runner, self.deterministic_rows(), different)
                    comparison = report['comparison_to_previous']
                    self.assertFalse(comparison['available'])
                    self.assertEqual(comparison['reason'], 'previous_evaluation_grid_or_scope_does_not_match')
                    self.assertNotIn('stochastic_reward_delta_mean', comparison)

    def test_fixed_year_single_repeat_uses_supplied_year_and_has_no_estimated_standard_error(self):
        runner, calls = self.runner(repeats=1), []
        with patch('harl.envs.cascade_reservoir.diagnostics.evaluate_policy',
                   side_effect=self.evaluation(runner, calls)), redirect_stdout(io.StringIO()):
            before = policy_evaluation_report(runner, self.deterministic_rows((2013,)),
                                              scope='fixed_training_year')
            runner.iteration, runner.offset = 6, 1.
            after = policy_evaluation_report(runner, self.deterministic_rows((2013,), offset=1.),
                                             before, scope='fixed_training_year')
        self.assertEqual([call[0] for call in calls], [2013, 2013])
        self.assertIn('in-sample', after['evaluation_scope'])
        self.assertIn('no generalization claim', after['evaluation_scope'])
        self.assertIsNone(after['stochastic_repeat_se_reward'])
        self.assertIsNone(after['sampled_results_by_year'][0]['reward_statistics']['sd'])
        self.assertIsNone(after['comparison_to_previous']['stochastic_reward_delta_sd'])
        self.assertIsNone(after['comparison_to_previous']['stochastic_reward_delta_se'])

    def test_legacy_validation_wrapper_prints_once_and_accepts_old_rows(self):
        runner, calls, output = self.runner(), [], io.StringIO()
        rows = self.deterministic_rows()
        for row in rows:
            del row['year']
        with patch('harl.envs.cascade_reservoir.diagnostics.evaluate_policy',
                   side_effect=self.evaluation(runner, calls)), redirect_stdout(output):
            report = validation_report(runner, rows)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertTrue(output.getvalue().startswith('DIAG_VALIDATION '))
        self.assertNotIn('DIAG_POLICY ', output.getvalue())
        self.assertEqual(report['years'], rows)
        self.assertEqual(report['evaluation_years'], [2020, 2021])

    def test_invalid_grid_and_evaluation_failure_restore_mode_and_rng(self):
        runner = self.runner()
        for value in (0, -1, True, 1.5):
            runner.algo_args['experiment']['diagnostic_stochastic_repeats'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'positive integer'):
                policy_evaluation_report(runner, self.deterministic_rows())
        runner.algo_args['experiment']['diagnostic_stochastic_repeats'] = 3
        rng = torch.get_rng_state().clone()
        def fail(*args, **kwargs):
            runner.actor[0].actor.eval()
            torch.rand(())
            raise RuntimeError('failed evaluation')
        with patch('harl.envs.cascade_reservoir.diagnostics.evaluate_policy', side_effect=fail), \
                self.assertRaisesRegex(RuntimeError, 'failed evaluation'):
            policy_evaluation_report(runner, self.deterministic_rows())
        self.assertTrue(runner.actor[0].actor.training)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
