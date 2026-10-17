"""Remaining-horizon release feasibility for daily and terminal storage bounds."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade


class TerminalReleaseBounds:
    """Daily zero-delay reachability, using the supplied remaining inflows.

    Daily bounds constrain END-of-day storage and include any intermediate
    deadlines compiled by the data loader. Soft references are not read here.
    Constant global bounds and upper-bound terminal targets use prefix water
    budgets when outlet capacity permits; other cases use a mass balance LP.
    Only feasibility is solved, never energy. Earlier current-day releases are
    fixed as the environment resolves the cascade upstream to downstream.
    """

    def __init__(self, data, dt, storage, future_external, target=None,
                 storage_lower=None, storage_upper=None):
        self.data, self.dt = data, dt
        flows = np.asarray(future_external, dtype=np.float64)
        self.storage = np.asarray(storage, dtype=np.float64).copy()
        self.target = None if target is None else np.asarray(target, dtype=np.float64).copy()
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Feasibility timestep must be finite and positive')
        if (flows.ndim != 2 or flows.shape[1] != 5 or len(flows) == 0 or
                not np.isfinite(flows).all() or (flows < 0).any()):
            raise ValueError('Expected nonempty finite nonnegative future inflows with shape (days, 5)')
        if self.storage.shape != (5,) or not np.isfinite(self.storage).all():
            raise ValueError('Starting storage must be a finite five-vector')
        if self.target is not None and (self.target.shape != (5,) or not np.isfinite(self.target).all()):
            raise ValueError('Terminal storage target must be a finite five-vector')
        days = len(flows)
        self.n = n = days * 5
        if (storage_lower is None) != (storage_upper is None):
            raise ValueError('Supply both daily storage_lower and storage_upper')
        if storage_lower is None:
            self.storage_lower = np.broadcast_to(data.vmin, flows.shape).copy()
            self.storage_upper = np.broadcast_to(data.vmax, flows.shape).copy()
        else:
            self.storage_lower = np.asarray(storage_lower, dtype=np.float64).copy()
            self.storage_upper = np.asarray(storage_upper, dtype=np.float64).copy()
        if (self.storage_lower.shape != flows.shape or self.storage_upper.shape != flows.shape or
                not np.isfinite(self.storage_lower).all() or not np.isfinite(self.storage_upper).all() or
                (self.storage_lower > self.storage_upper).any()):
            raise ValueError('Daily storage bounds must be finite ordered arrays with shape (days, 5)')
        if (self.storage_lower < data.vmin).any() or (self.storage_upper > data.vmax).any():
            raise ValueError('Daily storage bounds must lie within global operating bounds')
        self.reference = self.storage if self.target is None else self.target
        self.tolerance = np.maximum(1., 1e-10*np.abs(self.reference))
        if ((self.storage < data.vmin-self.tolerance) | (self.storage > data.vmax+self.tolerance)).any():
            raise InfeasibleCascade('Starting storage violates operating bounds')
        if self.target is not None and ((self.target < self.storage_lower[-1]) |
                                        (self.target > self.storage_upper[-1])).any():
            raise InfeasibleCascade('Terminal storage target conflicts with final daily storage bounds')
        self.fast_bounds = None
        # At an upper-bound target, a feasible continuation exists iff each
        # upstream-prefix water budget can refill its prefix of reservoirs.
        # Sufficiency: fill upstream deficits first, passing excess downstream.
        # Releases then never exceed cumulative daily external inflows. This
        # argument requires the following outlet-capacity condition.
        global_bounds = (np.array_equal(self.storage_lower, np.broadcast_to(data.vmin, flows.shape)) and
                         np.array_equal(self.storage_upper, np.broadcast_to(data.vmax, flows.shape)))
        if (global_bounds and np.array_equal(self.target, data.vmax) and
                (np.cumsum(flows, axis=1) <= data.release_cap).all()):
            budget = np.cumsum(flows.astype(np.longdouble).sum(axis=0) +
                (self.storage.astype(np.longdouble)-self.target)/dt).astype(np.float64)
            if (budget < -np.cumsum(self.tolerance)/dt).any():
                raise InfeasibleCascade('Terminal storage target is unreachable: negative prefix water budget')
            lower = np.zeros(5)
            upper = np.minimum(data.release_cap, np.maximum(0., budget))
            for i in range(3, -1, -1):
                lower[i] = max(0., lower[i+1]-flows[0,i+1]+(data.vmin[i+1]-self.storage[i+1])/dt)
                upper[i] = min(upper[i], upper[i+1]-flows[0,i+1]+(data.vmax[i+1]-self.storage[i+1])/dt)
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
        self.rhs[:5] += (self.storage-self.reference) / dt
        self.bounds = list(zip(np.zeros(n), np.tile(data.release_cap, days))) + \
            list(zip(((self.storage_lower-self.reference)/dt).reshape(-1),
                     ((self.storage_upper-self.reference)/dt).reshape(-1)))
        if self.target is not None:
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
        if not result.success and result.status == 2 and self.target is not None:
            # Recover only within the final-storage tolerance already enforced
            # by the environment. Never alter the state or discard water.
            bounds = self.bounds.copy()
            for i in range(5):
                bounds[-5+i] = (max((self.storage_lower[-1,i]-self.target[i])/self.dt, -self.tolerance[i]/self.dt),
                               min((self.storage_upper[-1,i]-self.target[i])/self.dt, self.tolerance[i]/self.dt))
            result = linprog(objective, A_eq=self.matrix, b_eq=self.rhs,
                bounds=bounds, method='highs', options={**options, 'presolve': False})
        return result

    def check_feasible(self):
        """Check the entire horizon without selecting or optimizing a release."""
        if self.fast_bounds is not None:
            return
        result = self._solve(np.zeros(2*self.n))
        if not result.success:
            message = 'Operating constraints are unreachable: ' + result.message
            if result.status == 2:
                raise InfeasibleCascade(message)
            raise RuntimeError('Operating feasibility solver failed: ' + result.message)

    def interval(self, agent, releases):
        if isinstance(agent, bool) or not isinstance(agent, (int, np.integer)) or not 0 <= agent < 5:
            raise ValueError('Reservoir index must be an integer in [0, 4]')
        releases = np.asarray(releases, dtype=np.float64)
        # Only the resolved prefix is read; callers may leave later entries uninitialized.
        if releases.shape != (5,) or not np.isfinite(releases[:agent]).all():
            raise ValueError('Expected a five-vector with a finite resolved release prefix')
        if ((releases[:agent] < 0) | (releases[:agent] > self.data.release_cap[:agent])).any():
            raise InfeasibleCascade('Resolved release prefix violates outlet capacities')
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
                    reason = ('Daily storage bounds are unreachable' if self.target is None else
                              'Terminal storage target is unreachable under the daily storage bounds')
                    raise InfeasibleCascade((reason + ': agent={} days={} prefix={} {}').format(
                            agent, self.n//5, releases[:agent].tolist(), result.message))
                raise RuntimeError('Terminal feasibility solver failed: agent={} days={} {}'.format(
                    agent, self.n//5, result.message))
            limits.append(float(result.x[agent]))
        return limits
