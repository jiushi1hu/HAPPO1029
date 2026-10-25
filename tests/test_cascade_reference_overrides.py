"""Explicit annual references retain edited paths without weakening hard rules."""
import copy
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cascade_test_utils import load_configuration
from harl.envs.cascade_reservoir.data_loader import (
    CascadeData, IDS, REFERENCE_OVERRIDE_FILE, SEASONAL_FILES, SeasonalOperationData)

DATA = Path(__file__).resolve().parents[1] / 'data/cascade_reservoir_new'


class AnnualReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config, physical = load_configuration()
        cls.data = CascadeData(DATA, physical, config['experiment'])
        cls.seasonal = cls.data.load_seasonal_operation()
        cls.indices = {str(day): i for i, day in enumerate(cls.data.dates)}

    def load_mutated(self, filename, transform):
        original = SeasonalOperationData._rows

        def rows(instance, name, required):
            for row in original(instance, name, required):
                yield transform(dict(row)) if name == filename else row

        with patch.object(SeasonalOperationData, '_rows', rows):
            return SeasonalOperationData(self.data)

    def test_retained_paths_and_default_paths_are_independently_checked(self):
        s, d = self.seasonal, self.data
        self.assertEqual(set(s.reference_year_nodes),
                         {('THR', 2006), ('THR', 2011), ('THR', 2022), ('XJB', 2022)})
        examples = [('THR', '2006-09-07', 17515680558.),
                    ('THR', '2011-09-08', 17150000011.),
                    ('THR', '2022-09-30', 26949585918.),
                    ('XJB', '2022-09-30', 4930232800.)]
        for id, day, expected in examples:
            self.assertEqual(s.reference['target_end_of_day_storage_m3'][self.indices[day], IDS.index(id)],
                             expected)
        self.assertEqual(d.hashes[REFERENCE_OVERRIDE_FILE],
                         hashlib.sha256((DATA/REFERENCE_OVERRIDE_FILE).read_bytes()).hexdigest())
        for id, year in [('THR', 2008), ('XJB', 2023), ('WDD', 2022)]:
            a, b = d.indices(year, year)
            origin = d.dates[a]
            nodes = s.reference_nodes[id]
            days = [(origin.replace(month=n['month_day'][0], day=n['month_day'][1])-origin).days
                    for n in nodes] + [b-a]
            volumes = [n['storage'] for n in nodes] + [nodes[0]['storage']]
            np.testing.assert_allclose(s.reference['target_end_of_day_storage_m3'][a:b, IDS.index(id)],
                                       np.interp(np.arange(1, b-a+1), days, volumes), rtol=0., atol=1.1)

    def test_corrected_endpoint_and_hard_deadlines_remain_independent(self):
        s = self.seasonal
        j = IDS.index('XJB')
        self.assertEqual(s.reference['target_end_of_day_storage_m3'][self.indices['2022-10-24'], j],
                         4977000000.)
        self.assertEqual(s.reference['target_start_of_day_storage_m3'][self.indices['2022-10-25'], j],
                         4977000000.)
        t = self.indices['2022-09-30']
        self.assertEqual(s.storage_lower[t, j], 4977000000.)
        self.assertLess(s.reference['target_end_of_day_storage_m3'][t, j], s.storage_lower[t, j])
        # The early THR refill remains a soft target above an unchanged flood ceiling.
        t, j = self.indices['2006-09-07'], IDS.index('THR')
        self.assertGreater(s.reference['target_end_of_day_storage_m3'][t, j], s.storage_upper[t, j])
        self.assertEqual(s.reference['soft_target_lower_level_m'][t, j], 145.)
        self.assertEqual(s.reference['soft_target_upper_level_m'][t, j], 145.)
        self.assertTrue((s.reference['soft_target_lower_level_m'] <=
                         s.reference['soft_target_upper_level_m']).all())

    def test_deleting_override_does_not_silently_accept_edited_daily_data(self):
        with patch.object(SeasonalOperationData, '_load_year_nodes', lambda self: None):
            with self.assertRaisesRegex(ValueError, 'nodes/initial storage: XJB 2022'):
                SeasonalOperationData(self.data)

    def test_invalid_override_keys_dates_levels_and_sequences_are_rejected(self):
        changes = [('reservoir_id', 'unknown'), ('year', '2006.5'), ('year', '1964'),
                   ('date', '2005-12-31'), ('date', '2006-02-30'),
                   ('date', '2006-1-1'), ('node_sequence', '2'),
                   ('reference_target_storage_m3', '39300000002'),
                   ('reference_target_level_m', '174')]
        for key, value in changes:
            def transform(row):
                if row['reservoir_id'] == 'THR' and row['year'] == '2006' and row['node_sequence'] == '1':
                    row[key] = value
                return row
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.load_mutated(REFERENCE_OVERRIDE_FILE, transform)

    def test_missing_required_override_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for path in DATA.glob('*.csv'):
                if path.name != REFERENCE_OVERRIDE_FILE:
                    (root/path.name).symlink_to(path)
            (root/REFERENCE_OVERRIDE_FILE).write_text('reservoir_id,year\nTHR,2006\n')
            data = copy.copy(self.data)
            data.root = root
            with self.assertRaisesRegex(ValueError, 'Missing or duplicate seasonal columns'):
                SeasonalOperationData(data)

    def test_individually_valid_node_edits_still_must_match_daily_path(self):
        def transform(row):
            if row['reservoir_id'] == 'THR' and row['year'] == '2006' and row['date'] == '2006-09-07':
                volume = float(row['reference_target_storage_m3']) + 100.
                curve = self.data.anchors['THR']
                row['reference_target_storage_m3'] = str(volume)
                row['reference_target_level_m'] = str(np.interp(volume, curve[:, 1], curve[:, 0]))
            return row
        with self.assertRaisesRegex(ValueError, 'nodes/initial storage: THR 2006'):
            self.load_mutated(REFERENCE_OVERRIDE_FILE, transform)

    def test_derived_bands_and_subphases_are_not_exempt_from_validation(self):
        for key, value, error in [('soft_target_upper_level_m', '146', 'target band mismatch'),
                                  ('reference_subphase_en', 'hold', 'subphase mismatch')]:
            def transform(row):
                if row['reservoir_id'] == 'THR' and row['date'] == '2006-09-07':
                    row[key] = value
                return row
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, error):
                self.load_mutated(SEASONAL_FILES[3], transform)

    def test_storage_bound_failure_identifies_date_and_boundary(self):
        def transform(row):
            if row['reservoir_id'] == 'XJB' and row['date'] in ('2022-10-24', '2022-10-25'):
                boundary = 'end' if row['date'] == '2022-10-24' else 'start'
                row['target_'+boundary+'_of_day_storage_m3'] = '4977000002'
                delta = float(row['target_end_of_day_storage_m3'])-float(row['target_start_of_day_storage_m3'])
                row['target_storage_change_m3'] = str(delta)
                row['target_net_storage_rate_m3s'] = str(delta/86400.)
            return row
        with self.assertRaisesRegex(ValueError, r'XJB 2022-10-25 start: storage=4977000002'):
            self.load_mutated(SEASONAL_FILES[3], transform)


if __name__ == '__main__':
    unittest.main()
