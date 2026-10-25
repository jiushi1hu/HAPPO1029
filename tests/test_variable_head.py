"""Storage/head feedback, capacity limits, input provenance and annual regression."""
import copy
import csv
import hashlib
import shutil
import tempfile
from pathlib import Path
import unittest

import numpy as np

from cascade_test_utils import load_configuration
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
from harl.envs.cascade_reservoir.data_loader import CascadeData, HEAD_SENSITIVITY_FILE, IDS
from harl.envs.cascade_reservoir.evaluation import evaluate_policy, reference_summary
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics, InfeasibleCascade
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data/cascade_reservoir_new'


class HeadInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.config = load_configuration()
        cls.config['head_model'] = 'storage_sensitivity'
        cls.data = CascadeData(DATA_ROOT, cls.config, cls.cfg['experiment'])

    def test_assumptions_are_loaded_by_id_and_hashed(self):
        d = self.data
        np.testing.assert_array_equal(d.head_sensitivity['reference_net_head_m'], d.head)
        self.assertEqual(d.hashes[HEAD_SENSITIVITY_FILE],
                         hashlib.sha256((DATA_ROOT / HEAD_SENSITIVITY_FILE).read_bytes()).hexdigest())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for path in DATA_ROOT.glob('*.csv'):
                shutil.copy2(path, root / path.name)
            path = root / HEAD_SENSITIVITY_FILE
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(reversed(rows))
            shuffled = CascadeData(root, self.config, self.cfg['experiment'])
            for key, value in d.head_sensitivity.items():
                np.testing.assert_array_equal(shuffled.head_sensitivity[key], value)

    def test_missing_supplement_fails_only_for_variable_head(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for path in DATA_ROOT.glob('*.csv'):
                if path.name != HEAD_SENSITIVITY_FILE:
                    shutil.copy2(path, root / path.name)
            with self.assertRaisesRegex(ValueError, 'requires.*reservoir_head_sensitivity.csv'):
                CascadeData(root, self.config, self.cfg['experiment'])
            fixed = dict(self.config, head_model='fixed')
            old_data = CascadeData(root, fixed, self.cfg['experiment'])
            self.assertIsNone(old_data.head_sensitivity)
            self.assertNotIn(HEAD_SENSITIVITY_FILE, old_data.hashes)
            with self.assertRaisesRegex(ValueError, 'Load the sensitivity data'):
                ReservoirPhysics(old_data, self.config)

    def test_stale_malformed_or_mislabelled_assumptions_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for path in DATA_ROOT.glob('*.csv'):
                shutil.copy2(path, root / path.name)
            path = root / HEAD_SENSITIVITY_FILE
            with path.open(newline='') as stream:
                original = list(csv.DictReader(stream))
            changes = [('reservoir_id', 'BHT'), ('data_status', 'measured'),
                       ('reference_net_head_m', '144'), ('operating_level_max_m', '976'),
                       ('assumed_reference_forebay_level_m', '961'),
                       ('assumed_effective_tailwater_plus_loss_m', '818'),
                       ('head_slope_m_per_m', '0'), ('scenario_net_head_min_m', '127'),
                       ('scenario_net_head_max_m', '159'), ('efficiency', '0.8'),
                       ('reference_net_head_m', 'nan'), ('efficiency', 'inf'),
                       ('head_slope_m_per_m', '')]
            for key, value in changes:
                with self.subTest(field=key, value=value):
                    rows = copy.deepcopy(original)
                    rows[0][key] = value
                    with path.open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    with self.assertRaises(ValueError):
                        CascadeData(root, self.config, self.cfg['experiment'])

    def test_unknown_mode_and_changed_efficiency_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown head model'):
            CascadeData(DATA_ROOT, dict(self.config, head_model='variable_typo'))
        with self.assertRaisesRegex(ValueError, 'Unknown head model'):
            ReservoirPhysics(self.data, dict(self.config, head_model='variable_typo'))
        with self.assertRaisesRegex(ValueError, 'efficiency must match'):
            CascadeData(DATA_ROOT, dict(self.config, efficiency=.8))


class VariableHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.config = load_configuration()
        cls.config['head_model'] = 'storage_sensitivity'
        cls.data = CascadeData(DATA_ROOT, cls.config, cls.cfg['experiment'])
        cls.physics = ReservoirPhysics(cls.data, cls.config)

    def test_higher_pool_increases_energy_per_unit_of_water(self):
        d, p = self.data, self.physics
        release = np.ones(5)  # Far below every power or turbine-flow limit.
        low = p.generation(release, d.vmin, d.vmin)
        high = p.generation(release, d.vmax, d.vmax)
        np.testing.assert_allclose(low['net_head_m'], [128., 172., 155., 95., 65.6])
        np.testing.assert_allclose(high['net_head_m'], [158., 232., 215., 105., 95.6])
        self.assertTrue((high['energy'] > low['energy']).all())
        np.testing.assert_allclose(high['energy']/low['energy'], high['net_head_m']/low['net_head_m'])
        # Independent units: Q=1 m3/s, H in m -> MW -> 24-hour MWh.
        np.testing.assert_allclose(high['energy'], 24.*1000.*9.81*.9*high['net_head_m']/1e6)

    def test_mean_is_taken_after_converting_storage_to_elevation(self):
        d = self.data
        row = self.physics.generation(np.ones(5), d.vmin, d.vmax)
        np.testing.assert_allclose(row['net_head_m'], d.head)
        expected = np.array([(self.config['level_bounds_m'][i][0] +
                              self.config['level_bounds_m'][i][1])/2. for i in IDS])
        np.testing.assert_allclose(row['forebay_level_mean_m'], expected)
        # The real storage-elevation curves are not all linear across this range.
        self.assertGreater(np.max(np.abs(d.levels((d.vmin+d.vmax)/2.)-expected)), .1)

    def test_higher_head_reduces_power_limited_flow_and_preserves_spill(self):
        d, p = self.data, self.physics
        release = d.turbine_cap + 1000.
        rows = [p.generation(release, storage, storage) for storage in (d.vmin, d.vmax)]
        self.assertTrue((rows[1]['effective_turbine_cap_m3s'] < rows[0]['effective_turbine_cap_m3s']).all())
        self.assertTrue((rows[1]['spill'] > rows[0]['spill']).all())
        for row in rows:
            np.testing.assert_allclose(row['turbine'] + row['spill'], release)
            self.assertTrue((row['turbine'] <= d.turbine_cap).all())
            self.assertTrue((row['power'] <= d.power_cap + 1e-7).all())
        np.testing.assert_allclose(rows[1]['power'], d.power_cap)

    def test_transition_uses_its_actual_beginning_and_ending_storage(self):
        d = self.data
        storage = (d.vmin+d.vmax)/2.
        external = d.external[0]
        row = self.physics.transition(storage, external, np.linspace(.1, .9, 5))
        mean_level = (d.levels(storage) + d.levels(row['storage_end']))/2.
        offset = np.array([817., 593., 385., 275., 79.4])
        np.testing.assert_allclose(row['net_head_m'], mean_level-offset, rtol=0., atol=1e-12)
        expected_power = np.minimum(row['release'], d.turbine_cap)*1000.*9.81*.9*(mean_level-offset)/1e6
        np.testing.assert_allclose(row['power'], np.minimum(expected_power, d.power_cap))
        self.assertAlmostEqual(row['reward'], row['energy'].sum()/1624800.)

    def test_generation_requires_valid_explicit_storage_and_never_changes_it(self):
        d, p = self.data, self.physics
        for storage in (None, np.zeros(4), np.full(5, np.nan), d.vmin-10., d.vmax+10.):
            with self.subTest(storage=storage):
                with self.assertRaises((ValueError, InfeasibleCascade)):
                    p.generation(np.ones(5), storage, storage)
        with self.assertRaises(ValueError):
            p.constant_storage(np.zeros((2, 5)))
        # Tolerated numerical deviations affect the lookup only, not water.
        storage = d.vmin.copy()
        storage[0] -= .0005
        before = storage.copy()
        row = p.transition(storage, np.zeros(5), np.zeros(5))
        np.testing.assert_array_equal(storage, before)
        np.testing.assert_array_equal(row['storage_end'], before)
        np.testing.assert_array_equal(row['release'], np.zeros(5))
        np.testing.assert_array_equal(row['energy'], np.zeros(5))

    def test_fixed_mode_retains_constant_head_at_every_storage(self):
        d = self.data
        p = ReservoirPhysics(d, dict(self.config, head_model='fixed'))
        low = p.generation(np.ones(5), d.vmin, d.vmin)
        high = p.generation(np.ones(5), d.vmax, d.vmax)
        np.testing.assert_array_equal(low['net_head_m'], d.head)
        np.testing.assert_array_equal(low['energy'], high['energy'])
        self.assertEqual(p.head_data_status, 'supplied_typical_head')

    def test_reference_uses_the_years_initial_storage_and_selected_physics(self):
        d = self.data
        for mode in ('fixed', 'storage_sensitivity'):
            p = ReservoirPhysics(d, dict(self.config, head_model=mode))
            for year in (1965, 1968):
                with self.subTest(mode=mode, year=year):
                    start, stop = d.indices(year, year)
                    release, energy = p.constant_storage(d.external[start:stop], d.initial_storage(year))
                    np.testing.assert_allclose(release, np.cumsum(d.external[start:stop], axis=-1))
                    h = d.head if mode == 'fixed' else np.array([158.,232.,215.,105.,95.6])
                    expected = 24.*np.minimum(np.minimum(release, d.turbine_cap)*1000.*9.81*.9*h/1e6,
                                              d.power_cap)
                    np.testing.assert_allclose(energy, expected)
                    row = reference_summary(d, p, [year],gamma=1.)[0]
                    self.assertAlmostEqual(row['energy_TWh'], expected.sum()/1e6)
                    self.assertEqual(row['head_model'], mode)

    def test_annual_water_trajectories_and_closure_match_fixed_mode(self):
        d = self.data
        for year, days in ((1965, 365), (1968, 366)):
            fixed = CascadeReservoirEnv(d, dict(self.config, head_model='fixed'), start_year=year, end_year=year)
            variable = CascadeReservoirEnv(d, self.config, start_year=year, end_year=year)
            rng = np.random.RandomState(year)
            energy_difference = 0.
            for _ in range(days):
                actions = rng.randint(0, 200, size=5)
                fixed.step(actions)
                variable.step(actions)
                a, b = fixed.last_transition, variable.last_transition
                for key in ('storage_start', 'storage_end', 'release', 'inflow', 'controls', 'residual'):
                    np.testing.assert_array_equal(a[key], b[key])
                np.testing.assert_allclose(b['release'], b['turbine']+b['spill'])
                self.assertTrue((b['power'] <= d.power_cap+1e-7).all())
                self.assertTrue((b['net_head_m'] >= d.head_sensitivity['scenario_net_head_min_m']-1e-9).all())
                self.assertTrue((b['net_head_m'] <= d.head_sensitivity['scenario_net_head_max_m']+1e-9).all())
                energy_difference += b['energy'].sum()-a['energy'].sum()
            self.assertNotEqual(energy_difference, 0.)
            np.testing.assert_allclose(variable.storage, d.initial_storage(year), rtol=0., atol=4.)

    def test_evaluation_identifies_assumptions_and_reports_head_ranges(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['model']['hidden_sizes'] = [8, 8]
        runner = CascadeReservoirRunner(self.data, self.config, cfg, 1)
        row = evaluate_policy(runner.actor, self.data, self.config, 2024, 1, 0, 0, diagnostics=True,
                              gamma=runner.algo_args['algo']['gamma'])
        self.assertEqual(row['head_model'], 'storage_sensitivity')
        self.assertEqual(row['head_data_status'], 'assumed_sensitivity_only')
        low, mean, high = [np.array(row[key]) for key in ('min_net_head_m', 'mean_net_head_m', 'max_net_head_m')]
        self.assertTrue((low <= mean+1e-10).all())
        self.assertTrue((mean <= high+1e-10).all())
        self.assertTrue((low >= self.data.head_sensitivity['scenario_net_head_min_m']-1e-9).all())
        self.assertTrue((high <= self.data.head_sensitivity['scenario_net_head_max_m']+1e-9).all())


if __name__ == '__main__':
    unittest.main()
