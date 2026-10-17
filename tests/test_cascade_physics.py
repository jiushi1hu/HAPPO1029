"""Analytical physical checks, independent of the HAPPO optimizer."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from examples.train_cascade import load_configuration
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics, InfeasibleCascade
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv

ROOT = Path(__file__).resolve().parents[1]


class PhysicalInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.physical = load_configuration()
        cls.data = CascadeData(ROOT / 'data/cascade_reservoir_new', cls.physical, cls.cfg['experiment'])

    def test_density_and_gravity_must_be_finite_positive(self):
        for key in ('water_density_kg_m3', 'gravity_m_s2'):
            for value in (float('nan'), float('inf'), 0., -1.):
                with self.subTest(key=key, value=value):
                    config = copy.deepcopy(self.physical)
                    config[key] = value
                    with self.assertRaises(ValueError):
                        ReservoirPhysics(self.data, config)

    def test_operating_levels_must_be_two_finite_ordered_values(self):
        for bounds in ([float('nan'), 975.], [945., float('nan')],
                       [945., float('inf')], [945.], [945., 950., 975.], [975., 945.]):
            with self.subTest(bounds=bounds):
                config = copy.deepcopy(self.physical)
                config['level_bounds_m']['WDD'] = bounds
                with self.assertRaises(ValueError):
                    CascadeData(ROOT / 'data/cascade_reservoir_new', config, self.cfg['experiment'])

    def test_reference_inflows_must_be_finite_nonnegative_five_vectors(self):
        physics = ReservoirPhysics(self.data, self.physical)
        for flows in (np.full(5, -1.), np.full(5, np.nan), np.full(5, np.inf),
                      np.zeros(4), np.zeros((2, 4)), np.array(0.)):
            with self.subTest(flows=flows):
                with self.assertRaises(ValueError):
                    physics.constant_storage(flows)
        release, energy = physics.constant_storage(np.zeros((2, 5)), self.data.initial_storage(1965))
        np.testing.assert_array_equal(release, np.zeros((2, 5)))
        np.testing.assert_array_equal(energy, np.zeros((2, 5)))


class AnalyticalPhysicsTests(unittest.TestCase):
    def setUp(self):
        _, self.config = load_configuration()
        self.config['head_model'] = 'fixed'  # Analytical fixtures specify constant head.
        self.data = SimpleNamespace(vmin=np.full(5, 1e9), vmax=np.full(5, 2e9),
            release_cap=np.full(5, 20000.), turbine_cap=np.full(5, 10000.),
            power_cap=np.full(5, 10000.), head=np.full(5, 100.))
        self.physics = ReservoirPhysics(self.data, self.config)
        self.storage = np.full(5, 1.5e9)

    def actions_for(self, storage, external, releases):
        # Independent local balance inversion for fixtures with ample downstream capacity.
        inflows = external + np.concatenate(([0.], releases[:-1]))
        low = np.maximum(0., inflows + (storage-self.data.vmax)/86400.)
        high = np.minimum(self.data.release_cap, inflows + (storage-self.data.vmin)/86400.)
        return np.sqrt((releases-low)/(high-low))

    def assert_balance(self, row, storage, external, releases):
        inflows = external + np.concatenate(([0.], releases[:-1]))
        np.testing.assert_allclose(row['release'], releases, rtol=0, atol=1e-10)
        np.testing.assert_allclose(row['inflow'], inflows, rtol=0, atol=1e-10)
        np.testing.assert_allclose(row['storage_end'], storage+(inflows-releases)*86400.,
                                   rtol=0, atol=1e-5)
        independent_residual = row['storage_end']-storage-(row['inflow']-row['release'])*86400.
        np.testing.assert_allclose(independent_residual, 0., rtol=0, atol=1e-5)
        self.assertLess(abs(row['storage_end'].sum()-storage.sum()-
                            (external.sum()-row['release'][-1])*86400.), 1e-5)

    def test_zero_flow_preserves_storage_and_has_zero_energy(self):
        row = self.physics.transition(self.storage, np.zeros(5), np.zeros(5))
        self.assert_balance(row, self.storage, np.zeros(5), np.zeros(5))
        np.testing.assert_array_equal(row['energy'], np.zeros(5))
        self.assertEqual(row['reward'], 0.)

    def test_constant_incremental_inflows_have_no_double_counting(self):
        external = np.arange(1., 6.)
        releases = np.array([1., 3., 6., 10., 15.])
        storage = self.storage.copy()
        for _ in range(3):
            row = self.physics.transition(storage, external, self.actions_for(storage, external, releases))
            self.assert_balance(row, storage, external, releases)
            np.testing.assert_allclose(row['storage_end'], self.storage, rtol=0, atol=1e-5)
            storage = row['storage_end']

    def test_positive_negative_and_zero_storage_changes_in_cubic_metres(self):
        external = np.arange(1., 6.)
        releases = np.array([0., 5., 0., 4., 2.])
        row = self.physics.transition(self.storage, external,
                                      self.actions_for(self.storage, external, releases))
        self.assert_balance(row, self.storage, external, releases)
        expected = np.array([86400., -259200., 691200., 0., 604800.])
        np.testing.assert_allclose(row['storage_end']-self.storage, expected, rtol=0, atol=1e-5)
        # 1 m3/s for one day = 86400 m3 = 8.64 units of 10000 m3.
        self.assertAlmostEqual((row['storage_end'][0]-self.storage[0])/10000., 8.64)

    def test_upper_boundary_releases_inflows_and_lower_boundary_cannot_supply_water(self):
        external = np.arange(1., 6.)
        row = self.physics.transition(self.data.vmax, external, np.zeros(5))
        self.assert_balance(row, self.data.vmax, external, np.cumsum(external))
        np.testing.assert_array_equal(row['storage_end'], self.data.vmax)
        row = self.physics.transition(self.data.vmin, np.zeros(5), np.ones(5))
        self.assert_balance(row, self.data.vmin, np.zeros(5), np.zeros(5))
        np.testing.assert_array_equal(row['controls'], np.zeros(5))
        # These actions deliberately reach both storage boundaries without clipping storage.
        # Draining five reservoirs in one day requires ample cumulative outlet capacity.
        self.data.release_cap.fill(100000.)
        for actions, target in ((np.ones(5), self.data.vmin), (np.zeros(5), self.data.vmax)):
            storage = self.data.vmax if actions[0] else self.data.vmin
            external = np.zeros(5) if actions[0] else np.full(5, 1e9/86400.)
            row = self.physics.transition(storage, external, actions)
            np.testing.assert_allclose(row['storage_end'], target, rtol=0, atol=1e-5)
            self.assert_balance(row, storage, external, row['release'])

    def test_pulse_routes_on_same_day_and_is_not_replayed_next_day(self):
        external = np.array([1., 0., 0., 0., 0.])
        release = np.ones(5)
        row = self.physics.transition(self.storage, external, self.actions_for(self.storage, external, release))
        self.assert_balance(row, self.storage, external, release)
        np.testing.assert_allclose(row['inflow'], np.ones(5), rtol=0, atol=1e-10)
        next_row = self.physics.transition(row['storage_end'], np.zeros(5), np.zeros(5))
        np.testing.assert_array_equal(next_row['inflow'], np.zeros(5))
        np.testing.assert_array_equal(next_row['release'], np.zeros(5))

    def test_net_head_efficiency_power_and_daily_energy_units(self):
        external = np.array([1., 0., 0., 0., 0.])
        row = self.physics.transition(self.storage, external,
            self.actions_for(self.storage, external, np.ones(5)))
        # rho=1000 kg/m3, g=9.81 m/s2, eta=.9, Q=1 m3/s, H=100 m.
        np.testing.assert_allclose(row['power']*1e6, np.full(5, 882900.), rtol=0, atol=1e-6)
        np.testing.assert_allclose(row['power']*1000., np.full(5, 882.9), rtol=0, atol=1e-8)
        np.testing.assert_allclose(row['power'], np.full(5, .8829), rtol=0, atol=1e-10)
        np.testing.assert_allclose(row['energy'], np.full(5, 21.1896), rtol=0, atol=1e-9)
        np.testing.assert_allclose(row['energy'], row['power']*(86400./3600.))
        altered = copy.deepcopy(self.config); altered['efficiency'] = .45
        half = ReservoirPhysics(self.data, altered).transition(self.storage, external,
            self.actions_for(self.storage, external, np.ones(5)))
        np.testing.assert_allclose(half['energy'], row['energy']/2.)

    def test_turbine_power_and_outlet_limits_preserve_spill_mass(self):
        self.data.turbine_cap[0] = 2.
        self.data.power_cap[1] = 3.*.8829
        physics = ReservoirPhysics(self.data, self.config)
        external = np.array([5., 0., 0., 0., 0.])
        row = physics.transition(self.storage, external,
            self.actions_for(self.storage, external, np.full(5, 5.)))
        np.testing.assert_allclose(row['turbine'], [2., 3., 5., 5., 5.])
        np.testing.assert_allclose(row['spill'], [3., 2., 0., 0., 0.], atol=1e-10)
        np.testing.assert_allclose(row['release'], row['turbine']+row['spill'])
        self.assert_balance(row, self.storage, external, np.full(5, 5.))
        with self.assertRaises(InfeasibleCascade):
            physics.transition(self.data.vmax, np.full(5, 30000.), np.zeros(5))

    def test_transition_does_not_mutate_inputs_and_infeasible_states_raise(self):
        storage, external, actions = self.storage.copy(), np.zeros(5), np.zeros(5)
        copies = [a.copy() for a in (storage, external, actions)]
        self.physics.transition(storage, external, actions)
        for actual, expected in zip((storage, external, actions), copies):
            np.testing.assert_array_equal(actual, expected)
        for bad in (self.data.vmin-2., self.data.vmax+2.):
            with self.assertRaises(InfeasibleCascade):
                self.physics.transition(bad, external, actions)

    def test_roundoff_at_lower_boundary_never_creates_negative_release_or_power(self):
        storage = self.data.vmin.copy(); storage[0] -= .0005
        row = self.physics.transition(storage, np.zeros(5), np.zeros(5))
        np.testing.assert_array_equal(row['release'], np.zeros(5))
        np.testing.assert_array_equal(row['power'], np.zeros(5))
        np.testing.assert_array_equal(row['storage_end'], storage)
        self.assert_balance(row, storage, np.zeros(5), np.zeros(5))

    def test_roundoff_at_upper_boundary_never_exceeds_outlet_capacity(self):
        storage = self.data.vmax.copy(); storage[0] += .0005
        external = np.array([20000., 0., 0., 0., 0.])
        row = self.physics.transition(storage, external, np.ones(5))
        np.testing.assert_array_equal(row['release'], np.full(5, 20000.))
        np.testing.assert_array_equal(row['storage_end'], storage)
        self.assert_balance(row, storage, external, np.full(5, 20000.))


class EnvironmentPhysicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.config = load_configuration()
        cls.data = CascadeData(ROOT / 'data/cascade_reservoir_new', cls.config, cls.cfg['experiment'])

    def test_elevation_interpolation_matches_anchors_and_storage_units(self):
        # XJB has a known linear table: 370 m -> 4.074e9 m3; 380 m -> 4.977e9 m3.
        storage = self.data.vmin.copy(); storage[3] = 4.5255e9
        self.assertEqual(self.data.levels(storage)[3], 375.)
        for i, reservoir in enumerate(self.config['reservoir_ids']):
            for level, volume in self.data.anchors[reservoir]:
                storage = self.data.vmin.copy(); storage[i] = volume
                self.assertEqual(self.data.levels(storage)[i], level)
        np.testing.assert_allclose(self.data.levels(self.data.vmin), [945.,765.,540.,370.,145.])
        np.testing.assert_allclose(self.data.levels(self.data.vmax), [975.,825.,600.,380.,175.])
        with self.assertRaises(ValueError):
            self.data.levels(self.data.vmin-2.)

    def test_reset_step_observations_termination_and_failed_step_are_consistent(self):
        env = CascadeReservoirEnv(self.data, self.config, start_year=1968, end_year=1968)
        original = env.storage.copy()
        obs, state, available = env.reset()
        self.assertEqual(obs.shape, (5, 18)); self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_array_equal(obs, state); self.assertIsNone(available)
        self.assertIsNone(env.last_transition)
        before_index = env.index
        with self.assertRaises(ValueError):
            env.step(np.full(5, np.nan))
        self.assertEqual(env.index, before_index)
        np.testing.assert_array_equal(env.storage, original)
        obs, state, reward, done, info, _ = env.step(np.zeros((5, 1)))
        row = env.last_transition
        np.testing.assert_allclose(obs[0,:5], (row['storage_end']-self.data.vmin)/(self.data.vmax-self.data.vmin), atol=1e-7)
        np.testing.assert_allclose(obs[0,5:10], (self.data.external[env.index]-self.data.flow_mean)/self.data.flow_std, atol=1e-7)
        np.testing.assert_allclose(obs[0,10:15], row['release']/self.data.release_cap, atol=1e-7)
        np.testing.assert_array_equal(obs, state)
        np.testing.assert_allclose(reward, row['reward']); self.assertFalse(done.any())
        self.assertTrue(all(not item['bad_transition'] for item in info))
        for _ in range(365):
            obs, state, _, done, _, _ = env.step(np.zeros(5))
        self.assertTrue(done.all()); self.assertEqual(env.index, env.stop)
        np.testing.assert_array_equal(obs, np.zeros((5,18)))
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(5))
        env.reset()
        np.testing.assert_array_equal(env.storage, original)
        np.testing.assert_array_equal(env.previous_release, np.zeros(5))
        self.assertEqual(env.index, env.start); self.assertIsNone(env.last_transition)
        self.assertAlmostEqual(float(env.observation()[-1]), 1.)


if __name__ == '__main__':
    unittest.main()
