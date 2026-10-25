"""Fixed-year learning checks stay in-sample and preserve training state."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from examples.train_cascade import main
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.runners.cascade_reservoir_runner import (
    CascadeReservoirRunner, validate_fixed_training_year)
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
import test_reward_scale_metrics as reward_metric_tests


class FixedYearTrainingTests(unittest.TestCase):
    def fixture(self):
        data, physical, cfg = reward_metric_tests.RewardScaleMetricsTests().fixture()
        # The short physical fixture represents one training year only.
        data.validation_years = (2020, 2021, 2022, 2023, 2024)
        data.test_years, data.test_labels = (2006,), ('extremely_dry',)
        cfg['experiment'].update(fixed_training_year=1965, diagnostics=False)
        return data, physical, cfg

    def test_only_integer_training_years_are_accepted(self):
        data, _, _ = self.fixture()
        self.assertIsNone(validate_fixed_training_year(None, data))
        self.assertEqual(validate_fixed_training_year(np.int64(1965), data), 1965)
        for year in (True, np.bool_(False), 1965., '1965', [], 1966, 2006, 2020):
            with self.subTest(year=year), self.assertRaisesRegex(ValueError, 'training split'):
                validate_fixed_training_year(year, data)

    def test_fixed_sampling_does_not_draw_year_rng_and_rejects_other_years(self):
        data, physical, cfg = self.fixture()
        runner = CascadeReservoirRunner(data, physical, cfg, 1)
        self.assertEqual([runner.sample_training_year() for _ in range(10)], [1965]*10)
        self.assertEqual(runner.year_rng.randint(100000), np.random.RandomState(1).randint(100000))
        self.assertEqual(runner.algo_args['algo']['gamma'], .99)
        with self.assertRaisesRegex(ValueError, 'different training year'):
            runner.collect_episode(1966)
        self.assertEqual(runner.episode_attempts, 0)

    def test_evaluation_is_deterministic_and_keeps_buffers_parameters_and_rng(self):
        data, physical, cfg = self.fixture()
        stream = io.StringIO()
        runner = CascadeReservoirRunner(data, physical, cfg, 1, episode_log=stream)
        with redirect_stdout(io.StringIO()):
            runner.collect_episode()
        actions = [buffer.actions.copy() for buffer in runner.actor_buffer]
        rewards, storage = runner.critic_buffer.rewards.copy(), runner.envs[0].storage.copy()
        parameters = [copy.deepcopy(actor.actor.state_dict()) for actor in runner.actor]
        rng = torch.get_rng_state().clone()

        def evaluated(*args, **kwargs):
            self.assertIs(kwargs['deterministic'], True)
            self.assertEqual(args[3], 1965)
            # The fork must protect training even if evaluation consumes RNG.
            torch.rand(3)
            return evaluate_policy(*args, **kwargs)

        with patch('harl.runners.cascade_reservoir_runner.evaluate_policy', side_effect=evaluated), \
                redirect_stdout(io.StringIO()):
            runner.evaluate_training_year()
            runner.evaluate_training_year()
        a, b = runner.fixed_year_history
        self.assertEqual(a, b)
        self.assertEqual(a['record_type'], 'fixed_year_evaluation')
        self.assertEqual(a['evaluation']['phase'], 'training_evaluation')
        self.assertAlmostEqual(a['total_reward'], a['power_reward']-
            sum(a[key+'_penalty'] for key in ('band', 'rise', 'fall')))
        self.assertEqual(runner.validation_history, [])
        self.assertEqual(runner.episode_attempts, 1)
        self.assertEqual(len(runner.episode_history), 1)
        np.testing.assert_array_equal(rewards, runner.critic_buffer.rewards)
        np.testing.assert_array_equal(storage, runner.envs[0].storage)
        for saved, buffer in zip(actions, runner.actor_buffer):
            np.testing.assert_array_equal(saved, buffer.actions)
        for saved, actor in zip(parameters, runner.actor):
            for name, value in saved.items():
                torch.testing.assert_close(value, actor.actor.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(len(stream.getvalue().splitlines()), 3)

    def test_run_checks_untrained_interval_and_last_policy_without_validation(self):
        data, physical, cfg = self.fixture()
        runner = CascadeReservoirRunner(data, physical, cfg, 1)
        checkpoints, years = [], []

        def collect():
            years.append(runner.sample_training_year())

        def evaluate():
            checkpoints.append(runner.iteration)
            return 0.

        with patch.object(runner, 'evaluate_training_year', side_effect=evaluate), \
                patch.object(runner, 'validate') as validation, \
                patch.object(runner, 'collect_episode', side_effect=collect), \
                patch.object(OnPolicyHARunner, 'train', return_value=([{} for _ in range(5)], {})):
            result = runner.run(5, 2)
        self.assertEqual(checkpoints, [0, 2, 4, 5])
        self.assertEqual(years, [1965]*5)
        self.assertNotIn('best_validation_TWh', result)
        validation.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'do not run held-out tests'):
            runner.final_test()

    def test_cli_skips_held_out_and_reference_runs_and_logs_baseline(self):
        data, physical, cfg = self.fixture()
        cfg['experiment']['fixed_training_year'] = None  # Exercise the CLI override.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'fixed.jsonl'
            argv = ['train_cascade.py', '--fixed-training-year', '1965', '--episodes', '1',
                    '--eval-interval', '1', '--seeds', '1', '--gamma', '0.5',
                    '--no-plot', '--episode-log', str(path)]
            output = io.StringIO()
            with patch('sys.argv', argv), \
                    patch('examples.train_cascade.load_configuration', return_value=(cfg, physical)), \
                    patch('examples.train_cascade.CascadeData', return_value=data), \
                    patch('examples.train_cascade.validate_operating_feasibility', return_value=[1965]) as check, \
                    patch('examples.train_cascade.reported_references') as references, \
                    patch.object(CascadeReservoirRunner, 'validate') as validation, \
                    patch.object(CascadeReservoirRunner, 'final_test') as final_test, \
                    redirect_stdout(output):
                main()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([row['record_type'] for row in rows],
                         ['fixed_year_evaluation', 'training_episode', 'fixed_year_evaluation'])
        self.assertEqual([rows[0]['episode'], rows[-1]['episode']], [0, 1])
        startup, _ = json.JSONDecoder().raw_decode(output.getvalue())
        self.assertEqual(startup['active_training_years'], [1965])
        self.assertFalse(startup['held_out_evaluation_enabled'])
        self.assertEqual(startup['gamma'], .5)
        self.assertTrue(all(row['gamma'] == .5 for row in rows))
        self.assertTrue(all('reward_aggregation' in row for row in rows))
        check.assert_called_once_with(data, physical, years=[1965])
        for work in (references, validation, final_test):
            work.assert_not_called()

    def test_cli_rejects_reserved_year_before_feasibility_or_training(self):
        data, physical, cfg = self.fixture()
        with patch('sys.argv', ['train_cascade.py', '--fixed-training-year', '2006', '--no-plot']), \
                patch('examples.train_cascade.load_configuration', return_value=(cfg, physical)), \
                patch('examples.train_cascade.CascadeData', return_value=data), \
                patch('examples.train_cascade.validate_operating_feasibility') as check, \
                patch('examples.train_cascade.CascadeReservoirRunner') as runner, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
            main()
        self.assertEqual(failure.exception.code, 2)
        check.assert_not_called()
        runner.assert_not_called()


if __name__ == '__main__':
    unittest.main()
