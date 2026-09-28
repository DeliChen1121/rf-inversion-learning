"""Check experiment isolation and output protection without loading ML libraries."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('runner', ROOT / 'simple-lab/run.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ExperimentPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.here = Path(self.temp.name)
        self.original_here = runner.HERE
        runner.HERE = self.here
        self.addCleanup(setattr, runner, 'HERE', self.original_here)
        self.config = (ROOT / 'simple-lab/config.py').read_text()
        self.write_config()

    def write_config(self, extra=''):
        (self.here / 'config.py').write_text(self.config + '\n' + extra)

    def test_eta_changes_only_eta_and_reuses_inversion(self):
        base, sweep, cases = runner.make_plan()
        self.assertEqual(sweep, 'eta')
        self.assertEqual([c['eta'] for c in cases], [0, .2, .4, .6, .8, .9, 1])
        self.assertEqual(len({runner.inversion_key(c) for c in cases}), 1)
        for c in cases:
            for key in runner.KEYS:
                if key not in ('name', 'eta'):
                    self.assertEqual(c[key], base[key])

    def test_gamma_requires_distinct_inversions(self):
        self.write_config('sweep = "gamma"\nvalues = [0, 0.5, 1]')
        _, _, cases = runner.make_plan()
        self.assertEqual(len({runner.inversion_key(c) for c in cases}), 3)

    def test_window_pairs_preserve_seven_updates(self):
        self.write_config('sweep = "window"\nvalues = [(0,7/28),(7/28,14/28),(14/28,21/28)]')
        _, _, cases = runner.make_plan()
        indices = [(int(c['start_timestep'] * 28), int(c['stop_timestep'] * 28)) for c in cases]
        self.assertEqual(indices, [(0, 7), (7, 14), (14, 21)])

    def test_invalid_cases_are_rejected_before_execution(self):
        for extra in ('values = [0, 1.2]', 'name = "../escape"',
                      'sweep = "window"\nvalues = [(0.8, 0.2)]',
                      'editing_strength = 0', 'values = []'):
            with self.subTest(extra=extra):
                self.write_config(extra)
                with self.assertRaises(ValueError):
                    runner.make_plan()

    def test_single_run_ignores_sweep_values(self):
        self.write_config('sweep = None\nvalues = []')
        base, sweep, cases = runner.make_plan()
        self.assertIsNone(sweep)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]['eta'], base['eta'])

    def test_existing_output_is_not_overwritten(self):
        output = self.here / 'results/01_eta_scan'
        output.mkdir(parents=True)
        marker = output / 'keep.txt'
        marker.write_text('previous result')
        with patch.object(runner, 'check_files', return_value='test-hash'), \
                patch('sys.argv', ['run.py']), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(FileExistsError):
                runner.main()
        self.assertEqual(marker.read_text(), 'previous result')


if __name__ == '__main__':
    unittest.main()
