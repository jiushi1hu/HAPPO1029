"""Release-bound queries must depend on their supplied prefix, never call history."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog

from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.envs.cascade_reservoir.terminal_storage import TerminalReleaseBounds


class TerminalQueryStateTests(unittest.TestCase):
    def fixture(self, *, closed=True, fast=False):
        dt = 86400.
        data = SimpleNamespace(vmin=np.zeros(5), vmax=np.full(5, 10.*dt),
                               release_cap=np.full(5, 10.))
        storage = np.full(5, (10. if fast else 5.)*dt)
        flows = np.zeros((2, 5)); flows[:, 0] = 1.
        return TerminalReleaseBounds(data, dt, storage, flows,
                                     storage.copy() if closed else None)

    @staticmethod
    def releases(prefix):
        # The unresolved suffix is intentionally invalid: only the prefix is read.
        releases = np.full(5, np.nan)
        releases[:len(prefix)] = prefix
        return releases

    def test_earlier_reservoir_query_does_not_retain_a_later_query_prefix(self):
        bounds = self.fixture()
        self.assertIsNone(bounds.fast_bounds)
        bounds.interval(1, self.releases([0.]))
        np.testing.assert_array_equal(bounds.interval(0, self.releases([])), [0., 2.])

    def test_arbitrary_query_order_matches_fresh_instances_and_keeps_base_bounds(self):
        queries = [(4, [1., 1., 1., 1.]), (0, []), (2, [0., 1.]),
                   (1, [2.]), (3, [1., 0., 1.]), (0, [])]
        for closed in (True, False):
            with self.subTest(terminal_target=closed):
                bounds = self.fixture(closed=closed)
                original = copy.deepcopy(bounds.bounds)
                for agent, prefix in queries:
                    releases = self.releases(prefix)
                    untouched = releases.copy()
                    expected = self.fixture(closed=closed).interval(agent, releases)
                    np.testing.assert_array_equal(bounds.interval(agent, releases), expected)
                    self.assertEqual(bounds.bounds, original)
                    np.testing.assert_array_equal(releases, untouched)

    def test_repeated_reservoir_queries_use_the_new_prefix(self):
        bounds = self.fixture(closed=False)
        np.testing.assert_array_equal(bounds.interval(1, self.releases([3.])), [0., 8.])
        np.testing.assert_array_equal(bounds.interval(1, self.releases([0.])), [0., 5.])
        np.testing.assert_array_equal(bounds.interval(1, self.releases([1.])), [0., 6.])

    def test_infeasible_prefix_does_not_contaminate_feasibility_or_later_queries(self):
        bounds = self.fixture()
        original = copy.deepcopy(bounds.bounds)
        # Three units of first-day release exceed the entire two-day water budget.
        with self.assertRaises(InfeasibleCascade):
            bounds.interval(1, self.releases([3.]))
        self.assertEqual(bounds.bounds, original)
        bounds.check_feasible()
        np.testing.assert_array_equal(bounds.interval(0, self.releases([])), [0., 2.])
        np.testing.assert_array_equal(bounds.interval(1, self.releases([1.])), [0., 2.])

    def test_default_monotone_queries_match_the_original_conditioned_lp(self):
        bounds = self.fixture()
        legacy_bounds = copy.deepcopy(bounds.bounds)
        prefix = np.zeros(5)
        for agent in range(5):
            objective = np.zeros(2*bounds.n); objective[agent] = 1.
            expected = []
            for sign in (1., -1.):
                result = linprog(sign*objective, A_eq=bounds.matrix, b_eq=bounds.rhs,
                    bounds=legacy_bounds, method='highs', options={
                        'primal_feasibility_tolerance': 1e-9,
                        'dual_feasibility_tolerance': 1e-9})
                self.assertTrue(result.success, result.message)
                expected.append(float(result.x[agent]))
            actual = bounds.interval(agent, prefix)
            np.testing.assert_array_equal(actual, expected)
            prefix[agent] = (actual[0]+actual[1])/2.
            legacy_bounds[agent] = (prefix[agent], prefix[agent])

    def test_terminal_tolerance_retry_retains_query_prefix_and_base_bounds(self):
        bounds = self.fixture()
        original = copy.deepcopy(bounds.bounds)
        failure = SimpleNamespace(success=False, status=2, message='infeasible')
        solution = np.zeros(2*bounds.n); solution[0] = 1.; solution[1] = .5
        success = SimpleNamespace(success=True, status=0, message='ok', x=solution)
        calls = []

        def solve(*args, **kwargs):
            calls.append(copy.deepcopy(kwargs['bounds']))
            return failure if len(calls) < 3 else success

        with patch('harl.envs.cascade_reservoir.terminal_storage.linprog', side_effect=solve):
            np.testing.assert_array_equal(bounds.interval(1, self.releases([1.])), [.5, .5])
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(query[0] == (1., 1.) for query in calls))
        self.assertEqual(calls[0][-5:], [(0., 0.)]*5)
        self.assertTrue(any(low < 0. < high for low, high in calls[2][-5:]))
        self.assertEqual(bounds.bounds, original)

    def test_fast_prefix_budget_path_remains_independent_of_query_order(self):
        bounds = self.fixture(fast=True)
        self.assertIsNotNone(bounds.fast_bounds)
        for agent, prefix in [(4, [1., 1., 1., 1.]), (0, []),
                              (2, [1.5, 1.5]), (1, [1.]), (0, [])]:
            releases = self.releases(prefix)
            expected = self.fixture(fast=True).interval(agent, releases)
            np.testing.assert_array_equal(bounds.interval(agent, releases), expected)
        bounds.check_feasible()


if __name__ == '__main__':
    unittest.main()
