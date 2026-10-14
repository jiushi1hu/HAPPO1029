"""Double-precision water balance and exact current-day chain feasibility."""
import numpy as np


class InfeasibleCascade(RuntimeError):
    pass


class ReservoirPhysics:
    def __init__(self, data, config):
        self.data = data
        if config.get("release_mapping", "quadratic") != "quadratic":
            raise ValueError("This experiment uses quadratic normalized-action mapping")
        self.dt = float(config["timestep_seconds"])
        if self.dt != 86400.0:
            raise ValueError("This model is daily")
        eta = float(config["efficiency"])
        if not 0 < eta <= 1:
            raise ValueError("Efficiency must lie in (0, 1]")
        density = float(config["water_density_kg_m3"])
        gravity = float(config["gravity_m_s2"])
        if not np.isfinite([density, gravity]).all() or density <= 0 or gravity <= 0:
            raise ValueError("Water density and gravity must be finite and positive")
        self.coefficient = density * gravity * eta * data.head / 1e6
        self.effective_turbine_cap = np.minimum(data.turbine_cap, data.power_cap / self.coefficient)
        self.reference_energy = float(24 * data.power_cap.sum())

    def downstream_intervals(self, storage, external):
        d = self.data
        lo, hi = np.zeros(5), d.release_cap.copy()
        for i in range(3, -1, -1):
            lo[i] = max(0., lo[i + 1] - external[i + 1] + (d.vmin[i + 1] - storage[i + 1]) / self.dt)
            hi[i] = min(d.release_cap[i], hi[i + 1] - external[i + 1] + (d.vmax[i + 1] - storage[i + 1]) / self.dt)
            if lo[i] > hi[i] + 1e-8:
                raise InfeasibleCascade("Empty downstream-compatible release interval")
        return lo, hi

    def transition(self, storage, external, actions, terminal_bounds=None):
        storage, external, actions = [np.asarray(x, dtype=np.float64) for x in (storage, external, actions)]
        if any(x.shape != (5,) or not np.isfinite(x).all() for x in (storage, external, actions)):
            raise ValueError("Expected three finite five-vectors")
        if ((actions < 0) | (actions > 1)).any() or (external < 0).any():
            raise ValueError("Invalid normalized action or external flow")
        d = self.data
        if ((storage < d.vmin - 1) | (storage > d.vmax + 1)).any():
            raise InfeasibleCascade("Starting storage violates operating bounds")
        lower, upper = self.downstream_intervals(storage, external)
        inflow, release, low, high = [np.empty(5) for _ in range(4)]
        controls, executed_actions = np.ones(5, dtype=np.float32), actions.copy()
        for i in range(5):
            inflow[i] = external[i] + (release[i - 1] if i else 0.)
            low[i] = max(lower[i], inflow[i] + (storage[i] - d.vmax[i]) / self.dt)
            high[i] = min(upper[i], inflow[i] + (storage[i] - d.vmin[i]) / self.dt)
            if terminal_bounds is not None:
                terminal_low, terminal_high = terminal_bounds.interval(i, release)
                low[i] = max(low[i], terminal_low)
                high[i] = min(high[i], terminal_high)
            if low[i] > high[i] + 1e-8:
                raise InfeasibleCascade("Empty feasible release interval at reservoir " + str(i))
            if high[i] - low[i] <= 1e-8:
                controls[i] = 0
                executed_actions[i] = 0.5
                # A tolerance-sized interval inversion can put the midpoint
                # just outside [0, outlet capacity]. Bound the actual flow;
                # storage is still computed from it and checked below.
                release[i] = min(d.release_cap[i], max(0., 0.5 * (low[i] + high[i])))
            else:
                release[i] = low[i] + actions[i] ** 2 * (high[i] - low[i])
        next_storage = storage + (inflow - release) * self.dt
        turbine = np.minimum(release, self.effective_turbine_cap)
        spill = release - turbine
        power = turbine * self.coefficient
        energy = 24. * power
        residual = next_storage - storage - (inflow - release) * self.dt
        tolerance = np.maximum(1., 1e-10 * np.abs(storage))
        if ((next_storage < d.vmin - tolerance) | (next_storage > d.vmax + tolerance)).any():
            raise InfeasibleCascade("Storage bound violation after execution")
        if (release < -1e-8).any() or (release > d.release_cap + 1e-8).any():
            raise InfeasibleCascade("Release capacity violation")
        if (np.abs(residual) > tolerance).any() or (power > d.power_cap + 1e-7).any():
            raise AssertionError("Water/energy calculation failed")
        system_residual = next_storage.sum() - storage.sum() - (external.sum() - release[-1]) * self.dt
        if abs(system_residual) > tolerance.sum():
            raise AssertionError("Cascade water conservation failed")
        return dict(storage_start=storage.copy(), storage_end=next_storage, inflow=inflow,
                    external=external.copy(), actions=executed_actions, release=release,
                    turbine=turbine, spill=spill, power=power, energy=energy,
                    low=low, high=high, controls=controls, residual=residual,
                    reward=float(energy.sum() / self.reference_energy),
                    system_residual=float(system_residual))

    def constant_storage(self, external):
        external = np.asarray(external, dtype=np.float64)
        if (external.ndim < 1 or external.shape[-1] != 5 or
                not np.isfinite(external).all() or (external < 0).any()):
            raise ValueError("Reference inflows must be finite nonnegative five-vectors")
        release = np.cumsum(external, axis=-1)
        if (release > self.data.release_cap + 1e-8).any():
            raise InfeasibleCascade("Constant-storage reference exceeds outlet capacities")
        turbine = np.minimum(release, self.effective_turbine_cap)
        return release, 24. * self.coefficient * turbine
