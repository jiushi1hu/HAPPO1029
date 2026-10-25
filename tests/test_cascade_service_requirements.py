"""Paper Table A.1 transcription, input validation, and data-only integration."""
import copy
import csv
import hashlib
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from examples.train_cascade import load_configuration, main
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
from harl.envs.cascade_reservoir.data_loader import (
    CascadeData, IDS, SERVICE_REQUIREMENTS_FILE, ServiceRequirementsData)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/cascade_reservoir_new'
FIELDS = ('reservoir_id', 'paper_reservoir_id', 'guaranteed_output_mw',
          'ecological_flow_m3s', 'source_doi', 'source_table')
EXPECTED_POWER = np.array([2290., 4567.5, 5345.7, 2750., 4990.])
EXPECTED_FLOW = np.array([900., 1160., 1160., 1200., 6000.])


class ServiceRequirementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.physical = load_configuration()
        cls.data = CascadeData(DATA, cls.physical, cls.config['experiment'])
        with (DATA / SERVICE_REQUIREMENTS_FILE).open(newline='') as stream:
            cls.rows = list(csv.DictReader(stream))

    def fixture(self, root):
        return SimpleNamespace(root=root, power_cap=self.data.power_cap.copy(),
                               release_cap=self.data.release_cap.copy())

    def write_rows(self, root, rows, fields=FIELDS):
        with (root / SERVICE_REQUIREMENTS_FILE).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_exact_paper_values_units_mapping_and_provenance(self):
        requirements = self.data.service_requirements
        np.testing.assert_array_equal(requirements.guaranteed_output_mw, EXPECTED_POWER)
        np.testing.assert_array_equal(requirements.ecological_flow_m3s, EXPECTED_FLOW)
        metadata = requirements.metadata()
        self.assertEqual(metadata['reservoir_ids'], list(IDS))
        self.assertEqual(metadata['paper_reservoir_ids'], ['WDD', 'BHT', 'XLD', 'XJB', 'TGR'])
        self.assertEqual(metadata['time_basis'], 'constant_daily_requirement')
        self.assertEqual(metadata['value_status'], 'paper_reported')
        for name in IDS:
            self.assertEqual(metadata['sources'][name],
                             dict(source_doi='10.1016/j.jhydrol.2026.136160', source_table='Table A.1'))
        digest = hashlib.sha256((DATA / SERVICE_REQUIREMENTS_FILE).read_bytes()).hexdigest()
        self.assertEqual(metadata['sha256'], digest)
        self.assertEqual(self.data.hashes[SERVICE_REQUIREMENTS_FILE], digest)
        self.assertIs(self.data.load_service_requirements(), requirements)
        with self.assertRaises(ValueError):
            requirements.guaranteed_output_mw[0] = 0.
        metadata['sources']['WDD']['source_table'] = 'changed'
        self.assertEqual(requirements.metadata()['sources']['WDD']['source_table'], 'Table A.1')
        json.dumps(requirements.metadata(), allow_nan=False)

    def test_shuffled_rows_are_aligned_by_reservoir_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_rows(root, list(reversed(self.rows)))
            requirements = ServiceRequirementsData(self.fixture(root))
            np.testing.assert_array_equal(requirements.guaranteed_output_mw, EXPECTED_POWER)
            np.testing.assert_array_equal(requirements.ecological_flow_m3s, EXPECTED_FLOW)

    def test_zero_is_an_explicit_inactive_requirement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = copy.deepcopy(self.rows)
            rows[0]['guaranteed_output_mw'] = '0'
            rows[1]['ecological_flow_m3s'] = '0'
            self.write_rows(root, rows)
            requirements = ServiceRequirementsData(self.fixture(root))
            self.assertEqual(requirements.guaranteed_output_mw[0], 0.)
            self.assertEqual(requirements.ecological_flow_m3s[1], 0.)

    def test_rejects_missing_duplicate_unknown_and_mismapped_reservoirs(self):
        changes = ('missing', 'duplicate', 'unknown', 'mapping')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for change in changes:
                with self.subTest(change=change):
                    rows = copy.deepcopy(self.rows)
                    if change == 'missing':
                        rows.pop()
                    elif change == 'duplicate':
                        rows.append(dict(rows[0]))
                    elif change == 'unknown':
                        rows[0]['reservoir_id'] = 'OTHER'
                    else:
                        rows[-1]['paper_reservoir_id'] = 'XJB'
                    self.write_rows(root, rows)
                    with self.assertRaisesRegex(ValueError, 'reservoir'):
                        ServiceRequirementsData(self.fixture(root))

    def test_rejects_invalid_numbers_and_capacity_excess(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for field in ('guaranteed_output_mw', 'ecological_flow_m3s'):
                for value in ('-1', 'nan', 'inf', '-inf', 'invalid', '1e100'):
                    with self.subTest(field=field, value=value):
                        rows = copy.deepcopy(self.rows)
                        rows[0][field] = value
                        self.write_rows(root, rows)
                        with self.assertRaisesRegex(ValueError, '[Ss]ervice requirement'):
                            ServiceRequirementsData(self.fixture(root))
            self.write_rows(root, self.rows)
            fixture = self.fixture(root)
            fixture.power_cap[0] = EXPECTED_POWER[0] - 1.
            with self.assertRaisesRegex(ValueError, 'configured capacity'):
                ServiceRequirementsData(fixture)

    def test_rejects_bad_headers_missing_values_and_missing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / SERVICE_REQUIREMENTS_FILE
            with self.assertRaisesRegex(ValueError, 'require'):
                ServiceRequirementsData(self.fixture(root))
            for fields in (FIELDS[:-1], FIELDS + ('source_table',)):
                with self.subTest(fields=fields):
                    rows = [{key: row[key] for key in fields} for row in self.rows]
                    self.write_rows(root, rows, fields)
                    with self.assertRaisesRegex(ValueError, 'columns'):
                        ServiceRequirementsData(self.fixture(root))
            for field in FIELDS:
                with self.subTest(field=field):
                    rows = copy.deepcopy(self.rows)
                    rows[0][field] = ''
                    self.write_rows(root, rows)
                    with self.assertRaisesRegex(ValueError, 'Missing service requirement value'):
                        ServiceRequirementsData(self.fixture(root))
            path.write_text(','.join(FIELDS) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'exactly one row'):
                ServiceRequirementsData(self.fixture(root))

    def test_explicit_loading_leaves_hydrology_bounds_and_transition_unchanged(self):
        physical = dict(self.physical, load_service_requirements=False)
        physical['reward_penalties'] = dict(physical['reward_penalties'], enabled=False)
        data = CascadeData(DATA, physical, self.config['experiment'])
        self.assertIsNone(data.service_requirements)
        self.assertNotIn(SERVICE_REQUIREMENTS_FILE, data.hashes)
        arrays = ('power_cap', 'release_cap', 'vmin', 'vmax', 'external', 'flow_mean', 'flow_std')
        before = {name: getattr(data, name).copy() for name in arrays}
        old = CascadeReservoirEnv(data, physical, start_year=2007, end_year=2007)
        old.step(np.full(5, 100))
        old_transition = old.last_transition
        data.load_service_requirements()
        for name, expected in before.items():
            np.testing.assert_array_equal(getattr(data, name), expected)
        new = CascadeReservoirEnv(data, physical, start_year=2007, end_year=2007)
        new.step(np.full(5, 100))
        for key in ('power', 'energy', 'release', 'storage_end', 'reward'):
            np.testing.assert_array_equal(new.last_transition[key], old_transition[key])
        self.assertEqual(new.last_transition['generation_reward'], old_transition['reward'])
        self.assertEqual(new.last_transition['reward_penalty'], 0.)
        self.assertEqual(new.reward_settings, old.reward_settings)

    def test_loading_flag_requires_a_boolean(self):
        for value in ('true', 1, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'load_service_requirements must be a boolean'):
                    CascadeData(DATA, dict(self.physical, load_service_requirements=value),
                                self.config['experiment'])

    def test_cli_reports_source_values_and_hash_without_changing_reward_settings(self):
        output = io.StringIO()
        with patch('sys.argv', ['train_cascade.py', '--check-data-only']), \
             patch('examples.train_cascade.CascadeData', return_value=self.data), \
             patch('examples.train_cascade.validate_operating_feasibility', return_value=[]), \
             redirect_stdout(output):
            main()
        startup, _ = json.JSONDecoder().raw_decode(output.getvalue())
        self.assertEqual(startup['service_requirements'], self.data.service_requirements.metadata())
        self.assertEqual(startup['reward_penalties'], self.physical['reward_penalties'])
        self.assertEqual(startup['gamma'], self.config['algo']['gamma'])


if __name__ == '__main__':
    unittest.main()
