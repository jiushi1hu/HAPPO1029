"""Explicit train/validation/test subsets and CLI holdout boundaries."""
import copy
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from examples.train_cascade import load_configuration, main
from harl.envs.cascade_reservoir.data_loader import CascadeData, resolve_year_split
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner, configure_calendar

ROOT = Path(__file__).resolve().parents[1]
TRAIN = [2007, 2008, 2009, 2010, 2011]
VALIDATION = [2012, 2013]
TEST = [2014]
FLAGS = ['--training-years'] + list(map(str, TRAIN)) + ['--validation-years', '2012', '2013', '--test-years', '2014']


class CustomYearSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.physical = load_configuration()
        cls.cfg['model']['hidden_sizes'] = [8, 8]
        cls.cfg['experiment'].update(training_years=TRAIN, validation_years=VALIDATION,
                                     test_years=TEST, test_labels=None, diagnostics=False)
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical, cls.cfg['experiment'])

    def test_default_partition_is_preserved(self):
        cfg, _ = load_configuration()
        split = resolve_year_split(cfg['experiment'])
        self.assertEqual(len(split['training_years']), 50)
        self.assertEqual(split['validation_years'], (2020, 2021, 2022, 2023, 2024))
        self.assertEqual(split['test_years'], (1998, 1999, 1970, 1997, 2006))
        self.assertEqual(split['test_labels'], ('extremely_wet', 'wet', 'normal', 'dry', 'extremely_dry'))

    def test_explicit_subset_and_scaler_ignore_all_unassigned_years(self):
        d = self.data
        self.assertEqual(d.training_years, tuple(TRAIN))
        self.assertEqual(d.validation_years, tuple(VALIDATION))
        self.assertEqual(d.test_years, (2014,))
        self.assertEqual(d.test_labels, ('test_2014',))
        self.assertEqual(len(d.unused_years), 52)
        expected_mask = np.array([day.year in TRAIN for day in d.dates])
        np.testing.assert_array_equal(d.train_mask, expected_mask)
        self.assertEqual(int(d.train_mask.sum()), 1826)
        np.testing.assert_array_equal(d.flow_mean, d.external[expected_mask].mean(0))
        np.testing.assert_array_equal(d.flow_std, np.maximum(d.external[expected_mask].std(0), 1.))

    def test_sampler_and_calendar_only_use_explicit_training_years(self):
        runner = CascadeReservoirRunner(self.data, self.physical, copy.deepcopy(self.cfg), 2)
        samples = [runner.sample_training_year() for _ in range(500)]
        self.assertEqual(set(samples), set(TRAIN))
        self.assertEqual(configure_calendar(self.cfg, self.data, 2008)['train']['episode_length'], 366)
        for year in [2006, 2012, 2013, 2014, 2024]:
            with self.subTest(year=year), self.assertRaisesRegex(ValueError, 'Only training years'):
                configure_calendar(self.cfg, self.data, year)

    def test_invalid_year_lists_and_labels_are_rejected(self):
        base = dict(training_years=TRAIN, validation_years=VALIDATION, test_years=TEST)
        invalid = [('training_years', []), ('validation_years', []), ('test_years', []),
                   ('training_years', [2007, 2007]), ('validation_years', [2012, 2012]),
                   ('test_years', [2014, 2014]), ('training_years', [2012]),
                   ('training_years', [2014]), ('validation_years', [2014]),
                   ('training_years', [1964]), ('test_years', [2025]),
                   ('training_years', [2007.]), ('validation_years', [True]),
                   ('test_years', ['2014']), ('test_labels', ['wet', 'dry']),
                   ('test_labels', [''])]
        for key, values in invalid:
            with self.subTest(key=key, values=values), self.assertRaises(ValueError):
                resolve_year_split(dict(base, **{key: values}))

    def test_cli_overrides_lists_and_limits_preflight_to_selected_years(self):
        output = io.StringIO()
        with patch('sys.argv', ['train_cascade.py'] + FLAGS + ['--check-data-only', '--seeds', '1', '2', '3']), \
                patch('examples.train_cascade.validate_operating_feasibility', return_value=list(range(2007, 2015))) as check, \
                redirect_stdout(output):
            main()
        startup, _ = json.JSONDecoder().raw_decode(output.getvalue())
        self.assertEqual(startup['training_years'], TRAIN)
        self.assertEqual(startup['training_year_count'], 5)
        self.assertEqual(startup['validation_years'], VALIDATION)
        self.assertEqual(startup['test_years'], {'test_2014': 2014})
        self.assertEqual(startup['checkpoint_evaluation'], 'held-out validation on 2 years')
        self.assertIsNone(startup['fixed_training_year'])
        self.assertEqual(check.call_args.kwargs['years'], list(range(2007, 2015)))

    def test_cli_rejects_overlapping_lists_before_data_loading(self):
        with patch('sys.argv', ['train_cascade.py', '--training-years', '2014',
                               '--validation-years', '2012', '2013', '--test-years', '2014']), \
                patch('examples.train_cascade.CascadeData') as data, redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            main()
        data.assert_not_called()

    def test_two_year_validation_selects_checkpoint_then_tests_only_2014(self):
        runner = CascadeReservoirRunner(self.data, self.physical, copy.deepcopy(self.cfg), 1)
        events = []

        def evaluate(actors, data, physical, year, seed, episode, steps, *, gamma):
            events.append((year, episode))
            self.assertEqual(gamma, .99)
            reward = ({2012: 2., 2013: 4.}.get(year, 3.) if episode == 0 else 1.)
            return dict(year=year, energy_MWh=reward*1e6, energy_TWh=reward,
                        episode_reward=reward, episode_penalty=0.)

        with patch('harl.runners.cascade_reservoir_runner.evaluate_policy', side_effect=evaluate), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.validate(), 3.)
            saved = [copy.deepcopy(a.actor.state_dict()) for a in runner.actor]
            with torch.no_grad():
                for actor in runner.actor:
                    next(actor.actor.parameters()).add_(1.)
            runner.iteration = 1
            self.assertEqual(runner.validate(), 1.)
            self.assertEqual(runner.best_iteration, 0)
            rows = runner.final_test()
        self.assertEqual(events, [(2012, 0), (2013, 0), (2012, 1), (2013, 1), (2014, 0)])
        self.assertEqual([row['year'] for row in rows], [2014])
        self.assertEqual(rows[0]['typical_label'], 'test_2014')
        for actor, state in zip(runner.actor, saved):
            for key, value in state.items():
                torch.testing.assert_close(actor.actor.state_dict()[key], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
