"""Read/validate source data without changing hydrology or applying policies."""
import csv
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

IDS = ("WDD", "BHT", "XLD", "XJB", "THR")
FLOW_COLUMNS = ("WDD_external_inflow_m3s",) + tuple(
    name + "_interval_inflow_m3s" for name in IDS[1:]
)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty input: " + str(path))
    return rows


class CascadeData:
    def __init__(self, root, config, experiment=None):
        self.root = Path(root).resolve()
        experiment = experiment or {}
        self.validation_years = tuple(int(y) for y in experiment.get("validation_years", range(2020, 2025)))
        self.test_years = tuple(int(y) for y in experiment.get("test_years", [1998, 1999, 1970, 1997, 2006]))
        self.test_labels = tuple(experiment.get("test_labels", ["extremely_wet", "wet", "normal", "dry", "extremely_dry"]))
        if len(self.validation_years) != 5 or len(set(self.validation_years)) != 5:
            raise ValueError("Require five unique validation years")
        if len(self.test_years) != 5 or len(set(self.test_years)) != 5 or len(self.test_labels) != 5:
            raise ValueError("Require five unique test years and five corresponding labels")
        if set(self.validation_years) & set(self.test_years):
            raise ValueError("Validation and test years must be disjoint")
        if tuple(config["reservoir_ids"]) != IDS:
            raise ValueError("Reservoir order must be WDD/BHT/XLD/XJB/THR")
        rels = ["daily_incremental_inflows_1965_2024.csv", "initial_storage_1965_2024.csv",
                "reservoir_stage_storage.csv", "reservoir_parameters.csv", "daily_operation_periods.csv"]
        self.hashes = {r: hashlib.sha256((self.root / r).read_bytes()).hexdigest() for r in rels}
        flows = read_csv(self.root / rels[0])
        self.dates = tuple(datetime.strptime(r["date"].strip(), "%Y-%m-%d").date() for r in flows)
        if any(b - a != timedelta(days=1) for a, b in zip(self.dates, self.dates[1:])):
            raise ValueError("Inflow dates must be unique, increasing and consecutive")
        self.external = np.array([[float(r[c]) for c in FLOW_COLUMNS] for r in flows], dtype=np.float64)
        if not np.isfinite(self.external).all() or (self.external < 0).any():
            raise ValueError("Inflow must be finite and nonnegative")
        expected = (datetime(1965, 1, 1).date(), datetime(2024, 12, 31).date())
        if (self.dates[0], self.dates[-1]) != expected or len(self.dates) != (expected[1] - expected[0]).days + 1:
            raise ValueError("Require complete daily inflows for 1965 through 2024")
        self.start_year, self.end_year = 1965, 2024
        self.years = np.array([d.year for d in self.dates])
        all_years = set(range(self.start_year, self.end_year + 1))
        reserved = set(self.validation_years) | set(self.test_years)
        if not reserved <= all_years:
            raise ValueError("Holdout years lie outside the dataset")
        self.training_years = tuple(sorted(all_years - reserved))
        if len(self.training_years) != 50:
            raise ValueError("Require exactly 50 training years")
        self.train_mask = np.isin(self.years, self.training_years)
        train_flow = self.external[self.train_mask]
        self.flow_mean = train_flow.mean(axis=0)
        self.flow_std = np.maximum(train_flow.std(axis=0), 1.0)
        periods = read_csv(self.root / rels[4])
        if len(periods) != len(flows) or [r["date"] for r in periods] != [r["date"] for r in flows]:
            raise ValueError("Operation-period dates must match daily inflows exactly")
        for day, row in zip(self.dates, periods):
            values = (day.year, day.month, day.day, day.timetuple().tm_yday)
            if tuple(int(row[k]) for k in ("year", "month", "day", "day_of_year")) != values:
                raise ValueError("Inconsistent calendar fields in operation periods")
        self.period_labels = tuple(r["period_en"] for r in periods)

        initial = read_csv(self.root / rels[1])
        physics = read_csv(self.root / rels[3])
        if len(physics) != 5 or {r["reservoir_id"] for r in physics} != set(IDS):
            raise ValueError("Physics must contain exactly one row per reservoir")
        self.initial_storage_by_year = {}
        keys = set()
        for row in initial:
            date = datetime.strptime(row["date"], "%Y-%m-%d").date()
            key = (date.year, row["reservoir_id"])
            if date.month != 1 or date.day != 1 or key in keys or row["reservoir_id"] not in IDS:
                raise ValueError("Invalid or duplicate annual initial-storage record")
            keys.add(key)
            self.initial_storage_by_year.setdefault(date.year, {})[row["reservoir_id"]] = float(row["initial_storage_m3"])
        if set(self.initial_storage_by_year) != all_years or any(set(v) != set(IDS) for v in self.initial_storage_by_year.values()):
            raise ValueError("Require five initial storages for every year 1965--2024")
        self.initial_storage_by_year = {year: np.array([v[id] for id in IDS])
                                        for year, v in self.initial_storage_by_year.items()}
        records = {r["reservoir_id"]: r for r in physics}
        if [int(records[i]["order"]) for i in IDS] != [1, 2, 3, 4, 5]:
            raise ValueError("Invalid hydraulic order")
        self.power_cap = np.array([float(config.get("installed_capacity_overrides_mw", {}).get(
            i, records[i]["installed_capacity_mw"])) for i in IDS])
        self.turbine_cap = np.array([float(records[i]["max_turbine_flow_m3s"]) for i in IDS])
        self.release_cap = np.array([float(records[i]["max_total_release_m3s"]) for i in IDS])
        self.head = np.array([float(records[i]["typical_net_head_m"]) for i in IDS])
        self.anchors = {}
        anchor_rows = read_csv(self.root / rels[2])
        if {r["reservoir_id"] for r in anchor_rows} != set(IDS):
            raise ValueError("Unexpected anchor reservoir IDs")
        for i in IDS:
            pairs = sorted((float(r["level_m"]), float(r["storage_m3"]))
                           for r in anchor_rows if r["reservoir_id"] == i)
            points = np.asarray(pairs, dtype=np.float64)
            if len(points) < 2 or not np.isfinite(points).all() or (np.diff(points, axis=0) <= 0).any():
                raise ValueError("Non-monotone or invalid anchors: " + i)
            self.anchors[i] = points
        lower, upper = [], []
        for i in IDS:
            z = np.asarray(config["level_bounds_m"][i], dtype=float)
            points = self.anchors[i]
            if (z.shape != (2,) or not np.isfinite(z).all() or
                    z[0] >= z[1] or z[0] < points[0, 0] or z[1] > points[-1, 0]):
                raise ValueError("Operating bounds outside curve: " + i)
            v = np.interp(z, points[:, 0], points[:, 1])
            lower.append(v[0]); upper.append(v[1])
        self.vmin, self.vmax = np.array(lower), np.array(upper)
        for name in ["power_cap", "turbine_cap", "release_cap", "head"]:
            a = getattr(self, name)
            if not np.isfinite(a).all() or (a <= 0).any():
                raise ValueError("Invalid positive physics input: " + name)
        for year, storage in self.initial_storage_by_year.items():
            if not np.isfinite(storage).all() or ((storage < self.vmin) | (storage > self.vmax)).any():
                raise ValueError("Initial storage outside operating bounds in year " + str(year))

    def initial_storage(self, year):
        """Return a copy of the selected year's supplied initial storage."""
        if int(year) not in self.initial_storage_by_year:
            raise ValueError("Initial storage year outside the dataset")
        return self.initial_storage_by_year[int(year)].copy()

    def indices(self, start_year, end_year):
        if not self.start_year <= start_year <= end_year <= self.end_year:
            raise ValueError("Requested years lie outside the dataset")
        selected = np.flatnonzero((self.years >= start_year) & (self.years <= end_year))
        if not len(selected):
            raise ValueError("Empty year selection")
        return int(selected[0]), int(selected[-1]) + 1

    def levels(self, storage):
        result = []
        for i, value in zip(IDS, storage):
            p = self.anchors[i]
            if not p[0, 1] - 1.0 <= value <= p[-1, 1] + 1.0:
                raise ValueError("Storage outside interpolation domain")
            result.append(np.interp(value, p[:, 1], p[:, 0]))
        return np.asarray(result)
