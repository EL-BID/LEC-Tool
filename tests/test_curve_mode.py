import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_loader import load_config, load_input_data, ConfigError


class CurveModeTests(unittest.TestCase):
    def load(self, mode_lines):
        source = (ROOT / 'config.toml').read_text(encoding='utf-8')
        source = source.replace('mode = "complete"', mode_lines)
        # Resolve CSV paths so this temporary config has the same inputs.
        for key in ('event_loss_file', 'tail_curve_file', 'complete_curve_file', 'ppo_schedule_file'):
            lines = source.splitlines()
            for i, line in enumerate(lines):
                if line.startswith(key + ' ='):
                    value = line.split('=', 1)[1].strip().strip('"')
                    lines[i] = f'{key} = "{(ROOT / value).as_posix()}"'
            source = '\n'.join(lines)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'config.toml'
            path.write_text(source, encoding='utf-8')
            return load_config(path)

    def test_each_mode_selects_one_path(self):
        for mode in ('complete', 'empirical', 'hybrid'):
            with self.subTest(mode=mode):
                cfg = self.load(f'mode = "{mode}"')
                self.assertEqual(cfg.lec.mode, mode)
                self.assertEqual(cfg.lec.complete_curve, mode == 'complete')
                self.assertEqual(cfg.lec.hybrid_curve, mode == 'hybrid')
                data = load_input_data(cfg)
                self.assertEqual(data['complete_curve'] is not None, mode == 'complete')
                self.assertEqual(data['tail_loss'] is not None, mode == 'hybrid')

    def test_legacy_configuration(self):
        for flags, expected in [('', 'hybrid'), ('complete_curve = true\nhybrid_curve = true', 'complete'), ('hybrid_curve = false', 'empirical')]:
            with self.subTest(flags=flags):
                self.assertEqual(self.load(flags).lec.mode, expected)

    def test_invalid_and_mixed_modes_are_rejected(self):
        for value in ['mode = "typo"', 'mode = true', 'mode = "complete"\nhybrid_curve = true', 'mode = "hybrid"\ncomplete_curve = false']:
            with self.subTest(value=value), self.assertRaises(ConfigError):
                self.load(value)


if __name__ == '__main__':
    unittest.main()
