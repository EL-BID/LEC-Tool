import sys, unittest, tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from config_loader import load_config, load_input_data, ConfigError
from main import compute_lec

class CompleteCurveTests(unittest.TestCase):
    def config(self):
        return load_config(ROOT/'config.toml')

    def test_supplied_curve_is_used_without_fitting_or_scaling(self):
        cfg=self.config()
        data=load_input_data(cfg)
        expected=pd.read_csv(cfg.inputs.complete_curve_file)[['tail_loss','tail_aep']].to_numpy()
        expected=expected[np.argsort(expected[:,0],kind='stable')]
        with patch('main.compute_empirical_lec',side_effect=AssertionError('must not fit')), patch('main.build_hybrid_lec',side_effect=AssertionError('must not blend')):
            result=compute_lec(cfg,data)
        np.testing.assert_array_equal(result['lec_curve'],expected)
        self.assertIsNone(result['lec_result'])
        self.assertGreater(result['aal'],0)
        self.assertGreater(result['lec_curve'][:,1].max(),1)
        self.assertTrue(data['event_loss_df'].empty)

    def test_complete_curve_needs_no_historical_or_tail_file(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_text=(ROOT/'config.toml').read_text(encoding='utf-8')
            cfg_text='\n'.join(line for line in cfg_text.splitlines() if not line.startswith(('event_loss_file','tail_curve_file')))
            cfg_text=cfg_text.replace('"El Salvador_final_lec.csv"','"'+(ROOT/'El Salvador_final_lec.csv').as_posix()+'"')
            path=Path(td)/'config.toml';path.write_text(cfg_text,encoding='utf-8')
            self.assertTrue(load_config(path).lec.complete_curve)

    def test_invalid_curves(self):
        cfg=self.config()
        for text in ['tail_loss,tail_aep\n1,1\n2,2\n','tail_loss,tail_aep\n1,1\n2,-1\n','tail_loss,tail_aep\n1,1\n2,NaN\n','tail_loss,tail_aep\n1,1\n','wrong,tail_aep\n1,1\n2,0.1\n']:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as td:
                path=Path(td)/'curve.csv';path.write_text(text)
                cfg.inputs.complete_curve_file=path
                with self.assertRaises(ConfigError):load_input_data(cfg)

    def test_legacy_mode_still_fits(self):
        cfg=self.config();cfg.lec.complete_curve=False;cfg.lec.hybrid_curve=False;cfg.lec.bootstrap_samples=3
        result=compute_lec(cfg,load_input_data(cfg))
        self.assertIsNotNone(result['lec_result'])

if __name__=='__main__':unittest.main()
