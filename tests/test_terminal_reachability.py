"""Water-budget boundary regressions and an independent cumulative-balance LP oracle."""
from pathlib import Path
import unittest
import numpy as np
from scipy.optimize import linprog
from cascade_test_utils import load_configuration
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.envs.cascade_reservoir.terminal_storage import TerminalReleaseBounds

ROOT = Path(__file__).resolve().parents[1]


class ReachabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg, cls.config = load_configuration()
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.config, cfg['experiment'])

    def boundary_env(self, shortage, upper_target=True):
        env = CascadeReservoirEnv(self.data, self.config, start_year=2022, end_year=2022)
        env.index = env.stop-3
        if not upper_target:
            env.terminal_storage_target = (self.data.vmin+self.data.vmax)/2.
        env.storage = env.terminal_storage_target-self.data.external[env.index:env.stop].sum(0)*86400.
        env.storage[0] -= shortage
        return env

    def test_sub_litre_water_budget_error_keeps_mass_and_terminal_tolerance(self):
        for upper_target in (True, False):
            with self.subTest(upper_target=upper_target):
                env = self.boundary_env(.0001, upper_target)
                while env.index < env.stop:
                    before = env.storage.copy()
                    env.step(np.zeros(5))
                    row = env.last_transition
                    np.testing.assert_allclose(env.storage-before,
                        (row['inflow']-row['release'])*86400., atol=1e-5, rtol=0)
                    self.assertTrue((row['release'] >= 0.).all())
                error = np.abs(env.storage-env.terminal_storage_target)
                self.assertTrue((error <= np.maximum(1.,1e-10*env.terminal_storage_target)).all())
                self.assertGreater(error[0], 0.)  # Missing water is not invented or clipped away.

    def test_real_shortage_is_rejected_with_replay_context(self):
        env = self.boundary_env(10.)
        before, index = env.storage.copy(), env.index
        with self.assertRaisesRegex(InfeasibleCascade, 'date=2022-12-29.*storage_m3=.*actions='):
            env.step(np.zeros(5))
        np.testing.assert_array_equal(env.storage, before)
        self.assertEqual(env.index,index)

    def test_fast_intervals_match_independent_cumulative_balance_lp(self):
        rng = np.random.RandomState(717)
        d, dt = self.data, 86400.
        for _ in range(24):
            days = int(rng.randint(1,8))
            _, stop = d.indices(2022,2022)
            flows = d.external[stop-days:stop]
            target = d.vmax
            storage = target-rng.uniform() * flows.sum(0)*dt
            bounds = TerminalReleaseBounds(d,dt,storage,flows,target)
            self.assertIsNotNone(bounds.fast_bounds)
            # Eliminate storage variables: cumulative outflow minus upstream
            # releases must respect every daily storage bound and final balance.
            chain = np.eye(5); chain[np.arange(1,5),np.arange(4)] = -1.
            matrix = np.kron(np.tril(np.ones((days,days))),chain)
            inflows = np.cumsum(flows,axis=0).reshape(-1)
            upper = inflows+np.tile((storage-d.vmin)/dt,days)
            lower = inflows+np.tile((storage-d.vmax)/dt,days)
            final = flows.sum(0)+(storage-target)/dt
            q_bounds = list(zip(np.zeros(days*5),np.tile(d.release_cap,days)))
            release = np.zeros(5)
            for i in range(5):
                objective = np.zeros(days*5);objective[i] = 1.
                expected = []
                for sign in (1.,-1.):
                    result = linprog(sign*objective,A_ub=np.vstack((matrix,-matrix)),
                        b_ub=np.concatenate((upper,-lower)),A_eq=matrix[-5:],b_eq=final,
                        bounds=q_bounds,method='highs')
                    self.assertTrue(result.success,result.message)
                    expected.append(result.x[i])
                actual = bounds.interval(i,release)
                np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)
                release[i] = actual[0]+rng.uniform()**2*(actual[1]-actual[0])
                q_bounds[i] = (release[i],release[i])

    def test_outlet_condition_selects_general_lp(self):
        flows = np.zeros((2,5));flows[0,0] = self.data.release_cap[0]+1.
        bounds = TerminalReleaseBounds(self.data,86400.,self.data.vmax,flows,self.data.vmax)
        self.assertIsNone(bounds.fast_bounds)
        with self.assertRaises(InfeasibleCascade):
            bounds.interval(0,np.zeros(5))


if __name__ == '__main__':
    unittest.main()
