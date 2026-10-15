"""Exact remaining-horizon release feasibility for a hard terminal storage target."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade


class TerminalReleaseBounds:
    """Daily zero-delay reachability, using the supplied remaining inflows.

    Upper-bound targets use prefix water budgets when outlet capacity permits;
    other cases use a mass balance LP with storage relative to the target.
    Only feasibility is solved, never energy. Earlier current-day releases are
    fixed as the environment resolves the cascade upstream to downstream.
    """

    def __init__(self, data, dt, storage, future_external, target):
        self.data, self.dt = data, dt
        flows = np.asarray(future_external, dtype=np.float64)
        self.storage, self.target = np.asarray(storage), np.asarray(target)
        days = len(flows)
        self.n = n = days * 5
        self.tolerance = np.maximum(1., 1e-10*np.abs(target))
        self.fast_bounds = None
        # At an upper-bound target, a feasible continuation exists iff each
        # upstream-prefix water budget can refill its prefix of reservoirs.
        # Sufficiency: fill upstream deficits first, passing excess downstream.
        # Releases then never exceed cumulative daily external inflows. This
        # argument requires the following outlet-capacity condition.
        if (np.array_equal(target, data.vmax) and
                (np.cumsum(flows, axis=1) <= data.release_cap).all()):
            budget = np.cumsum(flows.astype(np.longdouble).sum(axis=0) +
                (storage.astype(np.longdouble)-target)/dt).astype(np.float64)
            if (budget < -np.cumsum(self.tolerance)/dt).any():
                raise InfeasibleCascade('Terminal storage target is unreachable: negative prefix water budget')
            lower = np.zeros(5)
            upper = np.minimum(data.release_cap, np.maximum(0., budget))
            for i in range(3, -1, -1):
                lower[i] = max(0., lower[i+1]-flows[0,i+1]+(data.vmin[i+1]-storage[i+1])/dt)
                upper[i] = min(upper[i], upper[i+1]-flows[0,i+1]+(data.vmax[i+1]-storage[i+1])/dt)
            self.fast_bounds = lower, upper, flows[0]
            return
        indices = np.arange(n)
        rows = [indices, indices, indices[5:], indices[indices % 5 > 0]]
        cols = [indices, n+indices, n+indices[5:]-5, indices[indices % 5 > 0]-1]
        vals = [np.ones(n), np.ones(n), -np.ones(n-5), -np.ones(4*days)]
        self.matrix = coo_matrix((np.concatenate(vals),
            (np.concatenate(rows), np.concatenate(cols))), shape=(n, 2*n)).tocsr()
        self.rhs = flows.reshape(-1).copy()
        # Use storage deviation from the target, avoiding cancellation between
        # large absolute storage values at successive days and at the endpoint.
        self.rhs[:5] += (storage-target) / dt
        self.bounds = list(zip(np.zeros(n), np.tile(data.release_cap, days))) + \
            list(zip(np.tile((data.vmin-target)/dt, days), np.tile((data.vmax-target)/dt, days)))
        for i in range(5):
            self.bounds[-5+i] = (0., 0.)

    def _solve(self, objective):
        options = {'primal_feasibility_tolerance': 1e-9,
                   'dual_feasibility_tolerance': 1e-9}
        result = linprog(objective, A_eq=self.matrix, b_eq=self.rhs,
                         bounds=self.bounds, method='highs', options=options)
        if not result.success and result.status == 2:
            result = linprog(objective, A_eq=self.matrix, b_eq=self.rhs,
                bounds=self.bounds, method='highs', options={**options, 'presolve': False})
        if not result.success and result.status == 2:
            # Recover only within the final-storage tolerance already enforced
            # by the environment. Never alter the state or discard water.
            bounds = self.bounds.copy()
            for i in range(5):
                bounds[-5+i] = (max((self.data.vmin[i]-self.target[i])/self.dt, -self.tolerance[i]/self.dt),
                               min((self.data.vmax[i]-self.target[i])/self.dt, self.tolerance[i]/self.dt))
            result = linprog(objective, A_eq=self.matrix, b_eq=self.rhs,
                bounds=bounds, method='highs', options={**options, 'presolve': False})
        return result

    def interval(self, agent, releases):
        if self.fast_bounds is not None:
            lower, upper, external = self.fast_bounds
            inflow = external[agent] + (releases[agent-1] if agent else 0.)
            return [max(lower[agent], inflow+(self.storage[agent]-self.data.vmax[agent])/self.dt),
                    min(upper[agent], inflow+(self.storage[agent]-self.data.vmin[agent])/self.dt)]
        for i in range(agent):
            self.bounds[i] = (releases[i], releases[i])
        objective = np.zeros(2*self.n)
        objective[agent] = 1.
        limits = []
        for sign in (1., -1.):
            result = self._solve(sign*objective)
            if not result.success:
                if result.status == 2:
                    raise InfeasibleCascade('Terminal storage target is unreachable: agent={} days={} prefix={} {}'.format(
                        agent, self.n//5, releases[:agent].tolist(), result.message))
                raise RuntimeError('Terminal feasibility solver failed: agent={} days={} {}'.format(
                    agent, self.n//5, result.message))
            limits.append(float(result.x[agent]))
        return limits
