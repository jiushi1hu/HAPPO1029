"""Read/validate source data without changing hydrology or applying policies."""
import csv
import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

IDS = ("WDD", "BHT", "XLD", "XJB", "THR")
HEAD_MODELS = ("fixed", "storage_sensitivity")
HEAD_SENSITIVITY_FILE = "reservoir_head_sensitivity.csv"
SEASONAL_FILES = (
    "reservoir_five_key_operation_parameters.csv",
    "reservoir_period_operation_parameters.csv",
    "reservoir_reference_level_nodes.csv",
    "reservoir_daily_period_reference_1965_2024.csv",
)
OPERATION_PERIODS = ("drawdown", "flood_season", "impoundment", "dry_season")
FLOW_COLUMNS = ("WDD_external_inflow_m3s",) + tuple(
    name + "_interval_inflow_m3s" for name in IDS[1:]
)
DEFAULT_VALIDATION_YEARS = (2020, 2021, 2022, 2023, 2024)
DEFAULT_TEST_YEARS = (1998, 1999, 1970, 1997, 2006)
DEFAULT_TEST_LABELS = ("extremely_wet", "wet", "normal", "dry", "extremely_dry")


def resolve_year_split(experiment):
    """Validate disjoint year lists before reading data or fitting a scaler.

    Omitted training years retain the historical complement-of-holdouts default.
    Explicit training years leave every other unassigned year unused.
    """
    available = set(range(1965, 2025))

    def years(key, default):
        values = experiment.get(key, default)
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(key + ' must be a nonempty list of integer years')
        if any(isinstance(y, (bool, np.bool_)) or not isinstance(y, (int, np.integer)) for y in values):
            raise ValueError(key + ' must contain integer years')
        values = tuple(int(y) for y in values)
        if len(set(values)) != len(values):
            raise ValueError(key + ' must contain unique years')
        if not set(values) <= available:
            raise ValueError(key + ' must lie within 1965--2024')
        return values

    validation = years('validation_years', DEFAULT_VALIDATION_YEARS)
    tests = years('test_years', DEFAULT_TEST_YEARS)
    if set(validation) & set(tests):
        raise ValueError('Validation and test years must be disjoint')
    reserved = set(validation) | set(tests)
    training = (tuple(sorted(available - reserved)) if experiment.get('training_years') is None
                else tuple(sorted(years('training_years', ()))))
    if not training:
        raise ValueError('training_years must be nonempty')
    if set(training) & reserved:
        raise ValueError('Training years must be disjoint from validation and test years')
    labels = experiment.get('test_labels')
    if labels is None:
        labels = DEFAULT_TEST_LABELS if tests == DEFAULT_TEST_YEARS else tuple('test_'+str(y) for y in tests)
    if (not isinstance(labels, (list, tuple)) or len(labels) != len(tests) or
            any(not isinstance(label, str) or not label.strip() for label in labels) or
            len(set(labels)) != len(labels)):
        raise ValueError('Require one unique nonempty test label per test year')
    return dict(training_years=training, validation_years=validation,
                test_years=tests, test_labels=tuple(labels))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty input: " + str(path))
    return rows


class CascadeData:
    def __init__(self, root, config, experiment=None):
        self.root = Path(root).resolve()
        self.head_model = config.get("head_model", "fixed")
        if self.head_model not in HEAD_MODELS:
            raise ValueError("Unknown head model: " + str(self.head_model))
        experiment = experiment or {}
        for name, values in resolve_year_split(experiment).items():
            setattr(self, name, values)
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
        selected = set(self.training_years + self.validation_years + self.test_years)
        self.unused_years = tuple(sorted(all_years - selected))
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
        self.head_sensitivity = None
        if self.head_model == "storage_sensitivity":
            self._load_head_sensitivity(config)
        self.seasonal_operation = None
        load_seasonal = config.get("load_seasonal_operation", False)
        if not isinstance(load_seasonal, bool):
            raise ValueError("load_seasonal_operation must be a boolean")
        if load_seasonal:
            self.load_seasonal_operation()

    def load_seasonal_operation(self):
        """Explicitly load seasonal inputs without activating environment rules.

        The optional configuration key ``load_seasonal_operation: true`` calls
        this method during construction. Existing callers keep their original
        behavior. Consumers must explicitly pass storage_bounds(start, stop)
        to the reachability solver; loading never changes global vmin/vmax.
        """
        if self.seasonal_operation is None:
            seasonal = SeasonalOperationData(self)
            self.hashes.update(seasonal.hashes)
            self.seasonal_operation = seasonal
        return self.seasonal_operation

    def _load_head_sensitivity(self, config):
        """Validate the explicitly assumed scenario against its source inputs."""
        path = self.root / HEAD_SENSITIVITY_FILE
        if not path.is_file():
            raise ValueError("storage_sensitivity requires " + str(path))
        rows = read_csv(path)
        if len(rows) != 5 or {r.get("reservoir_id") for r in rows} != set(IDS):
            raise ValueError("Head sensitivity must contain exactly one row per reservoir")
        records = {r["reservoir_id"]: r for r in rows}
        if any(records[i].get("data_status") != "assumed_sensitivity_only" for i in IDS):
            raise ValueError("Head sensitivity must be labelled assumed_sensitivity_only")
        columns = ("operating_level_min_m", "operating_level_max_m",
                   "assumed_reference_forebay_level_m", "reference_net_head_m",
                   "assumed_effective_tailwater_plus_loss_m", "head_slope_m_per_m",
                   "scenario_net_head_min_m", "scenario_net_head_max_m", "efficiency")
        try:
            values = {key: np.array([float(records[i][key]) for i in IDS]) for key in columns}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Missing or nonnumeric head sensitivity field") from exc
        if not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError("Head sensitivity fields must be finite")
        low, high = np.array([config["level_bounds_m"][i] for i in IDS], dtype=float).T
        reference_level = (low + high) / 2.
        expected = {
            "operating_level_min_m": low,
            "operating_level_max_m": high,
            "assumed_reference_forebay_level_m": reference_level,
            "reference_net_head_m": self.head,
            "assumed_effective_tailwater_plus_loss_m": reference_level - self.head,
            "scenario_net_head_min_m": self.head + low - reference_level,
            "scenario_net_head_max_m": self.head + high - reference_level,
        }
        for key, value in expected.items():
            if not np.allclose(values[key], value, rtol=0., atol=1e-6):
                raise ValueError("Head sensitivity inconsistent with source inputs: " + key)
        if not np.allclose(values["head_slope_m_per_m"], 1., rtol=0., atol=1e-12):
            raise ValueError("This head scenario requires a constant offset (head slope 1)")
        if ((values["scenario_net_head_min_m"] <= 0).any() or
                (values["scenario_net_head_max_m"] <= 0).any()):
            raise ValueError("Scenario net head must remain positive throughout operating bounds")
        eta = float(config["efficiency"])
        if not 0. < eta <= 1. or not np.allclose(values["efficiency"], eta, rtol=0., atol=1e-12):
            raise ValueError("Head sensitivity efficiency must match the physical configuration")
        self.head_sensitivity = values
        self.hashes[HEAD_SENSITIVITY_FILE] = hashlib.sha256(path.read_bytes()).hexdigest()

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


class SeasonalOperationData:
    """End-of-day constraints and separate soft references in hydraulic order.

    The supplied key table defines a flood ceiling from the drawdown deadline
    THROUGH the flood-limit end date, inclusive. The refill level is a lower
    bound ON its deadline, not a requirement to track a prescribed daily path
    or to hold that level forever. Annual closure remains a solver argument.
    Reference nodes describe start-of-day storage and interpolate in storage,
    with actual calendar-day lengths (including leap days). Reference bands,
    rate suggestions and reward weights are never promoted to hard bounds.
    Provenance labels are retained verbatim, without asserting official status.
    """

    def __init__(self, data):
        self.data = data
        self.hashes = {}
        self.key_parameters = {}
        self.period_parameters = {}
        self.reference_nodes = {}
        shape = (len(data.dates), len(IDS))
        self.storage_lower = np.broadcast_to(data.vmin, shape).copy()
        self.storage_upper = np.broadcast_to(data.vmax, shape).copy()
        self.flood_limit_active = np.zeros(shape, dtype=bool)
        self.refill_deadline = np.zeros(shape, dtype=bool)
        self._load_keys()
        self._load_periods()
        self._load_nodes()
        self._load_daily_reference()
        for array in (self.storage_lower, self.storage_upper,
                      self.flood_limit_active, self.refill_deadline,
                      self.reference_subphases, *self.reference.values()):
            array.setflags(write=False)

    def _rows(self, filename, required):
        path = self.data.root / filename
        if not path.is_file():
            raise ValueError("Seasonal operation requires " + str(path))
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames or []
            if (len(headers) != len(set(headers)) or not set(required) <= set(headers)):
                raise ValueError("Missing or duplicate seasonal columns: " + filename)
            count = 0
            for row in reader:
                count += 1
                if None in row or any(row.get(k) is None or not row[k].strip() for k in required):
                    raise ValueError("Missing seasonal value: {} row {}".format(filename, reader.line_num))
                yield row
            if not count:
                raise ValueError("Empty input: " + str(path))
        self.hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _number(row, key):
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid seasonal number: " + key) from exc
        if not np.isfinite(value):
            raise ValueError("Nonfinite seasonal number: " + key)
        return value

    @staticmethod
    def _month_day(value):
        if re.fullmatch(r"\d{2}-\d{2}", value):
            month, day = map(int, value.split("-"))
        elif re.fullmatch(r"\d{1,2}月\d{1,2}日", value):
            month, day = map(int, value[:-1].split("月"))
        else:
            raise ValueError("Invalid recurring seasonal date: " + value)
        try:
            # A recurring rule must also have a defined date in common years.
            datetime(2001, month, day)
        except ValueError as exc:
            raise ValueError("Invalid common-year seasonal date: " + value) from exc
        return month, day

    def _volume(self, reservoir, level):
        j = IDS.index(reservoir)
        p = self.data.anchors[reservoir]
        low, high = self.data.levels(self.data.vmin)[j], self.data.levels(self.data.vmax)[j]
        if not low <= level <= high:
            raise ValueError("Seasonal level outside operating bounds: " + reservoir)
        return float(np.interp(level, p[:, 0], p[:, 1]))

    def _load_keys(self):
        fields = ("reservoir_id", "flood_control_limit_level_m", "drawdown_deadline_mmdd",
                  "flood_control_limit_end_mmdd", "impoundment_target_level_m",
                  "impoundment_deadline_mmdd", "date_interpretation", "value_status")
        for row in self._rows(SEASONAL_FILES[0], fields):
            id = row["reservoir_id"]
            if id not in IDS or id in self.key_parameters:
                raise ValueError("Invalid or duplicate seasonal reservoir: " + id)
            if row["date_interpretation"] != "end_of_day":
                raise ValueError("Seasonal key dates must use end_of_day")
            draw, end, refill = [self._month_day(row[k]) for k in fields[2:4] + fields[5:6]]
            if not draw <= end < refill:
                raise ValueError("Require drawdown deadline <= flood end < refill deadline: " + id)
            flood_v = self._volume(id, self._number(row, fields[1]))
            refill_v = self._volume(id, self._number(row, fields[4]))
            self.key_parameters[id] = dict(row)
            j = IDS.index(id)
            for year in range(self.data.start_year, self.data.end_year + 1):
                start, _ = self.data.indices(year, year)
                origin = self.data.dates[start]
                a, b, c = [start + (datetime(year, *md).date() - origin).days
                           for md in (draw, end, refill)]
                self.storage_upper[a:b+1, j] = flood_v
                self.flood_limit_active[a:b+1, j] = True
                self.storage_lower[c, j] = refill_v
                self.refill_deadline[c, j] = True
        if set(self.key_parameters) != set(IDS):
            raise ValueError("Seasonal keys require exactly one row per reservoir")

    def _load_periods(self):
        numeric = ("period_days_common_year", "period_days_leap_year",
                   "reference_start_level_m", "reference_end_level_m",
                   "reference_start_storage_m3", "reference_end_storage_m3",
                   "reference_storage_change_1e8m3", "mean_target_net_storage_rate_m3s",
                   "reservoir_min_operating_level_m", "reservoir_max_operating_level_m",
                   "reservoir_flood_limit_level_m", "max_turbine_flow_m3s", "max_total_release_m3s",
                   "suggested_soft_daily_rise_limit_m", "suggested_soft_daily_fall_limit_m",
                   "suggested_target_half_band_m", "weight_flood_safety", "weight_generation",
                   "weight_storage", "weight_downstream_support")
        required = ("reservoir_id", "period_en", "start_month_day", "end_month_day", "model_status") + numeric
        for row in self._rows(SEASONAL_FILES[1], required):
            id, period = row["reservoir_id"], row["period_en"]
            key = (id, period)
            if id not in IDS or period not in OPERATION_PERIODS or key in self.period_parameters:
                raise ValueError("Invalid or duplicate seasonal period: " + str(key))
            j = IDS.index(id)
            record = dict(row)
            record.update({k: self._number(row, k) for k in numeric})
            a, b = self._month_day(row["start_month_day"]), self._month_day(row["end_month_day"])
            if a > b:
                raise ValueError("Period dates must be within one calendar year")
            for year, field in ((2001, "period_days_common_year"), (2000, "period_days_leap_year")):
                days = (datetime(year, *b)-datetime(year, *a)).days + 1
                if record[field] != days:
                    raise ValueError("Incorrect period day count: " + str(key))
            for k in ("start", "end"):
                expected = self._volume(id, record["reference_"+k+"_level_m"])
                if abs(expected-record["reference_"+k+"_storage_m3"]) > 1.:
                    raise ValueError("Period level/storage mismatch: " + str(key))
            expected = (record["reference_end_storage_m3"]-record["reference_start_storage_m3"])
            if (abs(expected/1e8-record["reference_storage_change_1e8m3"]) > 1e-5 or
                    abs(expected/(record["period_days_common_year"]*86400.)-
                        record["mean_target_net_storage_rate_m3s"]) > .001):
                raise ValueError("Period storage change/rate mismatch: " + str(key))
            expected = (self.data.levels(self.data.vmin)[j], self.data.levels(self.data.vmax)[j],
                        float(self.key_parameters[id]["flood_control_limit_level_m"]),
                        self.data.turbine_cap[j], self.data.release_cap[j])
            if not np.allclose([record[k] for k in numeric[8:13]], expected, rtol=0., atol=1e-6):
                raise ValueError("Period physical parameters mismatch: " + str(key))
            weights = [record[k] for k in numeric[-4:]]
            if min(weights) < 0 or max(weights) > 1 or abs(sum(weights)-1.) > 1e-8:
                raise ValueError("Invalid reference reward weights: " + str(key))
            if (record["suggested_soft_daily_rise_limit_m"] <= 0 or
                    record["suggested_soft_daily_fall_limit_m"] <= 0 or
                    record["suggested_target_half_band_m"] < 0):
                raise ValueError("Invalid soft rate or target band: " + str(key))
            self.period_parameters[key] = record
        if len(self.period_parameters) != 5*len(OPERATION_PERIODS):
            raise ValueError("Require all four periods for every reservoir")

    def _load_nodes(self):
        required = ("reservoir_id", "node_sequence", "month_day", "reference_target_level_m",
                    "reference_target_storage_m3", "node_type")
        for row in self._rows(SEASONAL_FILES[2], required):
            id = row["reservoir_id"]
            if id not in IDS:
                raise ValueError("Invalid reference-node reservoir: " + id)
            level, volume = [self._number(row, k) for k in required[3:5]]
            if abs(self._volume(id, level)-volume) > 1.:
                raise ValueError("Reference node level/storage mismatch: " + id)
            record = dict(row, month_day=self._month_day(row["month_day"]),
                          level=level, storage=volume, sequence=self._number(row, "node_sequence"))
            self.reference_nodes.setdefault(id, []).append(record)
        for id in IDS:
            nodes = sorted(self.reference_nodes.get(id, []), key=lambda r: r["sequence"])
            if (len(nodes) < 2 or [r["sequence"] for r in nodes] != list(range(1, len(nodes)+1)) or
                    nodes[0]["month_day"] != (1, 1) or
                    any(a["month_day"] >= b["month_day"] for a, b in zip(nodes, nodes[1:]))):
                raise ValueError("Invalid reference node sequence/dates: " + id)
            self.reference_nodes[id] = tuple(nodes)

    def _load_daily_reference(self):
        fields = ("target_start_of_day_level_m", "target_end_of_day_level_m",
                  "target_start_of_day_storage_m3", "target_end_of_day_storage_m3",
                  "target_storage_change_m3", "target_net_storage_rate_m3s",
                  "soft_target_lower_level_m", "soft_target_upper_level_m",
                  "reservoir_operating_min_level_m", "reservoir_operating_max_level_m",
                  "reservoir_flood_limit_level_m", "zero_lag_cumulative_natural_inflow_m3s",
                  "reference_flood_limit_active", "reference_target_feasible_zero_lag",
                  "period_day_index", "period_progress_fraction", "weight_flood_safety",
                  "weight_generation", "weight_storage", "weight_downstream_support")
        required = ("date", "reservoir_id", "period_en", "reference_subphase_en") + fields
        shape = self.storage_lower.shape
        self.reference = ref = {k: np.empty(shape) for k in fields}
        self.reference_subphases = np.empty(shape, dtype="U8")
        seen = np.zeros(shape, dtype=bool)
        dates = {day.isoformat(): t for t, day in enumerate(self.data.dates)}
        for row in self._rows(SEASONAL_FILES[3], required):
            if row["date"] not in dates or row["reservoir_id"] not in IDS:
                raise ValueError("Unknown daily reference date/reservoir")
            t, j = dates[row["date"]], IDS.index(row["reservoir_id"])
            if seen[t, j]:
                raise ValueError("Duplicate daily reference date/reservoir")
            if row["period_en"] != self.data.period_labels[t]:
                raise ValueError("Daily reference period mismatch: " + row["date"])
            if row["reference_subphase_en"] not in ("drawdown", "hold", "refill"):
                raise ValueError("Invalid reference subphase")
            self.reference_subphases[t, j] = row["reference_subphase_en"]
            for k in fields:
                ref[k][t, j] = self._number(row, k)
            seen[t, j] = True
        if not seen.all():
            raise ValueError("Daily reference must cover every date and reservoir")
        start, end = ref[fields[2]], ref[fields[3]]
        if not np.allclose(start[1:], end[:-1], rtol=0., atol=1.):
            raise ValueError("Daily reference storage is discontinuous")
        if (not np.allclose(ref[fields[4]], end-start, rtol=0., atol=1.) or
                not np.allclose(ref[fields[5]], (end-start)/86400., rtol=0., atol=1e-5)):
            raise ValueError("Daily reference storage change/rate mismatch")
        for j, id in enumerate(IDS):
            p = self.data.anchors[id]
            for volumes, levels in ((start[:, j], ref[fields[0]][:, j]), (end[:, j], ref[fields[1]][:, j])):
                if ((volumes < self.data.vmin[j]-1.).any() or (volumes > self.data.vmax[j]+1.).any() or
                        not np.allclose(levels, np.interp(volumes, p[:, 1], p[:, 0]), rtol=0., atol=1e-6)):
                    raise ValueError("Daily reference level/storage mismatch: " + id)
            low, high = self.data.levels(self.data.vmin)[j], self.data.levels(self.data.vmax)[j]
            for year in range(self.data.start_year, self.data.end_year+1):
                a, b = self.data.indices(year, year)
                nodes = self.reference_nodes[id]
                origin = self.data.dates[a]
                node_days = [(datetime(year, *r["month_day"]).date()-origin).days for r in nodes] + [b-a]
                node_v = [r["storage"] for r in nodes] + [nodes[0]["storage"]]
                if (not np.allclose(start[a:b, j], np.interp(np.arange(b-a), node_days, node_v), rtol=0., atol=1.1) or
                        not np.allclose(end[a:b, j], np.interp(np.arange(1,b-a+1), node_days, node_v), rtol=0., atol=1.1) or
                        abs(start[a, j]-self.data.initial_storage(year)[j]) > 1.):
                    raise ValueError("Daily reference disagrees with nodes/initial storage: " + id)
                for period in OPERATION_PERIODS:
                    r = self.period_parameters[(id, period)]
                    x, y = [a+(datetime(year, *self._month_day(r[k])).date()-origin).days
                            for k in ("start_month_day", "end_month_day")]
                    if (any(p != period for p in self.data.period_labels[x:y+1]) or
                            not np.array_equal(ref["period_day_index"][x:y+1, j], np.arange(1,y-x+2)) or
                            not np.allclose(ref["period_progress_fraction"][x:y+1, j],
                                            np.arange(1,y-x+2)/(y-x+1), rtol=0., atol=1e-6) or
                            abs(start[x, j]-r["reference_start_storage_m3"]) > 1. or
                            abs(end[y, j]-r["reference_end_storage_m3"]) > 1.):
                        raise ValueError("Daily reference disagrees with period summary: " + str((id,period)))
                    band = r["suggested_target_half_band_m"]
                    reference_ceiling = np.where(ref["reference_flood_limit_active"][x:y+1,j] > 0,
                                                 r["reservoir_flood_limit_level_m"], high)
                    for k, expected in (("soft_target_lower_level_m", np.maximum(low, ref[fields[1]][x:y+1,j]-band)),
                                        ("soft_target_upper_level_m", np.minimum(reference_ceiling, ref[fields[1]][x:y+1,j]+band))):
                        if not np.allclose(ref[k][x:y+1,j], expected, rtol=0., atol=1e-6):
                            raise ValueError("Daily reference target band mismatch: " + id)
                    for k in fields[-4:]:
                        if not np.allclose(ref[k][x:y+1,j], r[k], rtol=0., atol=1e-8):
                            raise ValueError("Daily reference reward weight mismatch: " + id)
            for k, expected in ((fields[8], low), (fields[9], high),
                                (fields[10], float(self.key_parameters[id]["flood_control_limit_level_m"]))):
                if not np.allclose(ref[k][:,j], expected, rtol=0., atol=1e-6):
                    raise ValueError("Daily reference physical parameter mismatch: " + id)
        cumulative = np.cumsum(self.data.external, axis=1)
        if not np.allclose(ref[fields[11]], cumulative, rtol=0., atol=1e-6):
            raise ValueError("Daily reference natural inflow disagrees with source inflows")
        required_q = cumulative-np.cumsum((end-start)/86400., axis=1)
        feasible = (required_q >= -1e-5) & (required_q <= self.data.release_cap+1e-5)
        if not np.array_equal(ref[fields[13]], feasible.astype(float)):
            raise ValueError("Daily reference feasibility flags disagree with cascade water balance")
        # These are whole-day reference flags; the key deadline constrains
        # end-of-day storage even when that day's whole-day flag is zero.
        expected_active = self.flood_limit_active.copy()
        expected_active[1:] &= self.flood_limit_active[:-1]
        expected_active[0] = False
        if not np.array_equal(ref[fields[12]], expected_active.astype(float)):
            raise ValueError("Daily reference flood-limit flags disagree with key dates")

    def storage_bounds(self, start, stop):
        """Copies of inclusive-day-end bounds for the half-open index range."""
        if (any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in (start, stop)) or
                not 0 <= start < stop <= len(self.data.dates)):
            raise ValueError("Invalid seasonal horizon indices")
        return self.storage_lower[start:stop].copy(), self.storage_upper[start:stop].copy()
