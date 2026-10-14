"""Real-data partitioning and calendar regression checks."""
import csv
import shutil
import tempfile
from pathlib import Path
import unittest
import numpy as np
from examples.train_cascade import load_configuration
from harl.envs.cascade_reservoir.data_loader import CascadeData, FLOW_COLUMNS
from harl.runners.cascade_reservoir_runner import configure_calendar

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/cascade_reservoir_new"


class CalendarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.physical = load_configuration()
        cls.data = CascadeData(DATA_ROOT, cls.physical, cls.cfg["experiment"])

    def test_exact_disjoint_partition(self):
        d = self.data
        self.assertEqual(len(d.dates), 21915)
        self.assertEqual(d.validation_years, (2020, 2021, 2022, 2023, 2024))
        self.assertEqual(d.test_years, (1998, 1999, 1970, 1997, 2006))
        self.assertEqual(d.test_labels, ("extremely_wet", "wet", "normal", "dry", "extremely_dry"))
        expected = set(range(1965, 2025)) - set(d.validation_years) - set(d.test_years)
        self.assertEqual(set(d.training_years), expected)
        self.assertEqual(len(expected), 50)
        np.testing.assert_allclose(d.flow_mean, d.external[d.train_mask].mean(axis=0))
        self.assertEqual(d.power_cap[2], 12600.)

    def test_holdout_flows_cannot_change_training_scaler(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for p in DATA_ROOT.glob("*.csv"):
                shutil.copy2(p, root / p.name)
            p = root / "daily_incremental_inflows_1965_2024.csv"
            with p.open(newline="") as f:
                rows = list(csv.DictReader(f))
            heldout = set(self.data.validation_years + self.data.test_years)
            for row in rows:
                if int(row["date"][:4]) in heldout:
                    for key in FLOW_COLUMNS:
                        row[key] = str(float(row[key]) * 10.)
            with p.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            changed = CascadeData(root, self.physical, self.cfg["experiment"])
            np.testing.assert_array_equal(self.data.flow_mean, changed.flow_mean)
            np.testing.assert_array_equal(self.data.flow_std, changed.flow_std)

    def test_calendar_and_all_initial_states(self):
        d = self.data
        for year in range(1965, 2025):
            start, stop = d.indices(year, year)
            self.assertEqual(d.dates[start].isoformat(), str(year)+"-01-01")
            self.assertEqual(d.dates[stop-1].isoformat(), str(year)+"-12-31")
            import calendar
            self.assertEqual(stop-start, 366 if calendar.isleap(year) else 365)
            np.testing.assert_array_equal(d.initial_storage(year), d.vmax)
        with self.assertRaises(ValueError):
            d.indices(2025, 2025)

    def test_year_specific_buffer_length_no_holdout_sampling(self):
        for year, days in [(1965, 365), (1968, 366)]:
            cfg = configure_calendar(self.cfg, self.data, year)
            self.assertEqual(cfg["train"]["episode_length"], days)
            self.assertEqual(days % cfg["algo"]["actor_num_mini_batch"], 0)
        for year in self.data.validation_years + self.data.test_years:
            with self.assertRaises(ValueError):
                configure_calendar(self.cfg, self.data, year)


if __name__ == "__main__":
    unittest.main()
