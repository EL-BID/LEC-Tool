import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_loader import load_config, ConfigError
from risk_management import standard_insurance_payout, apply_strategy
from cba.config import InsuranceConfig, LECCBAConfig
from cba.engine import _build_instrument_config


class InsuranceValidationTests(unittest.TestCase):
    def test_valid_layer_payouts_and_percentage_boundaries(self):
        self.assertEqual(standard_insurance_payout([0,50,100,190,500],50,190,.5),
                         [0,0,25,70,70])
        self.assertEqual(standard_insurance_payout([100],50,190,0),[0])
        self.assertEqual(standard_insurance_payout([100],50,190,1),[50])

    def test_invalid_layers_rejected_by_payout_and_cost_config(self):
        cases=[(100,50,1),(50,50,1),(-1,50,1),(0,50,-.1),(0,50,1.1),
               (float('nan'),50,1),(0,float('inf'),1),(0,50,float('nan')),
               (True,50,1),('0',50,1)]
        for attachment,exhaustion,share in cases:
            with self.subTest(layer=(attachment,exhaustion,share)):
                with self.assertRaises(ValueError):
                    standard_insurance_payout([200],attachment,exhaustion,share)
                with self.assertRaises(ValueError):
                    InsuranceConfig(attachment_point=attachment,exhaustion_point=exhaustion,
                                    ceding_percentage=share)

    def test_cba_overrides_are_revalidated(self):
        for layer in [dict(attachment_point=100,exhaustion_point=50),
                      dict(attachment_point=50,exhaustion_point=190,ceding_percentage=1.1)]:
            with self.subTest(layer=layer),self.assertRaises(ValueError):
                _build_instrument_config(dict(type='insurance',**layer),LECCBAConfig())
        merged=_build_instrument_config(dict(type='insurance',attachment_point=50,
                                            exhaustion_point=190,ceding_percentage=.5,
                                            rate_on_line=.05),LECCBAConfig())
        self.assertEqual(merged.premium,3.5)

    def test_config_rejects_invalid_insurance_with_cba_on_and_off(self):
        source=(ROOT/'config.toml').read_text(encoding='utf-8')
        source=source.replace('"Country databases/','"'+ROOT.as_posix()+'/Country databases/')
        source=source.replace('"data/','"'+ROOT.as_posix()+'/data/')
        for cba_enabled in (False,True):
            for fields in ['attachment_point = 100\nexhaustion_point = 50\nceding_percentage = 1',
                           'attachment_point = 50\nexhaustion_point = 190\nceding_percentage = 1.1',
                           'attachment_point = nan\nexhaustion_point = 190\nceding_percentage = 1']:
                import re
                modified=re.sub(r'(\[cba\]\s*\nenabled\s*=\s*)(true|false)',
                                lambda m:m[1]+str(cba_enabled).lower(),source)
                block='[[strategy.instruments]]\nname = "Invalid insurance"\ntype = "insurance"\n'+fields+'\n\n'
                modified=modified.replace('[risk_reduction]',block+'[risk_reduction]')
                with self.subTest(cba=cba_enabled,fields=fields),tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
                    path=Path(folder)/'config.toml'
                    path.write_text(modified,encoding='utf-8')
                    with self.assertRaisesRegex(ConfigError,"Invalid insurance.*Insurance"):
                        load_config(path)


if __name__ == '__main__':
    unittest.main()
