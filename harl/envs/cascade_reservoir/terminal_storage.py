"""Exact remaining-horizon release feasibility for a hard terminal storage target."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade


class TerminalReleaseBounds:
    """Daily zero-delay mass balance LP, using the supplied remaining inflows.

    Storage variables are expressed as volume / timestep_seconds so equality
    coefficients stay near one. Only feasibility is solved; energy and policy
    actions are not optimized here. Earlier current-day releases are fixed as
    the environment resolves the cascade upstream to downstream.
    """

    def __init__(self, data, dt, storage, future_external, target):
        self.data, self.dt = data, dt
        flows = np.asarray(future_external, dtype=np.float64)
        days = len(flows)
        self.n = n = days * 5
        indices = np.arange(n)
        rows = [indices, indices, indices[5:], indices[indices % 5 > 0]]
        cols = [indices, n+indices, n+indices[5:]-5, indices[indices % 5 > 0]-1]
        vals = [np.ones(n), np.ones(n), -np.ones(n-5), -np.ones(4*days)]
        self.matrix = coo_matrix((np.concatenate(vals),
            (np.concatenate(rows), np.concatenate(cols))), shape=(n, 2*n)).tocsr()
        self.rhs = flows.reshape(-1).copy()
        self.rhs[:5] += storage / dt
        self.bounds = list(zip(np.zeros(n), np.tile(data.release_cap, days))) + \
            list(zip(np.tile(data.vmin/dt, days), np.tile(data.vmax/dt, days)))
        for i in range(5):
            self.bounds[-5+i] = (target[i]/dt, target[i]/dt)

    def interval(self, agent, releases):
        for i in range(agent):
            self.bounds[i] = (releases[i], releases[i])
        objective = np.zeros(2*self.n)
        objective[agent] = 1.
        limits = []
        for sign in (1., -1.):
            result = linprog(sign*objective, A_eq=self.matrix, b_eq=self.rhs,
                bounds=self.bounds, method='highs', options={
                    'primal_feasibility_tolerance': 1e-9,
                    'dual_feasibility_tolerance': 1e-9})
            if not result.success:
                if result.status == 2:
                    raise InfeasibleCascade('Terminal storage target is unreachable: ' + result.message)
                raise RuntimeError('Terminal feasibility solver failed: ' + result.message)
            limits.append(float(result.x[agent]))
        return limits
