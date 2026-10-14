import sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from risk_management import apply_strategy
from reporting import strategy_statistics
from test_credit import ddo
from config_loader import load_config

class FiscalTests(unittest.TestCase):
    def run_strategy(self, configs, losses, share=.3):
        return apply_strategy([dict(times=np.arange(len(losses))+.1,losses=np.array(losses))], configs,len(losses),first_year=2026,fiscal_share=share)

    def test_gross_trigger_fiscal_payment_and_balances(self):
        a=ddo(name='first',initial_balance=25,max_replenishments_per_period=0)
        b=ddo(name='second',initial_balance=100,max_replenishments_per_period=0)
        result=self.run_strategy([a,b],[200])
        self.assertEqual(result['payout_dfs'][0].iloc[0,0],25)
        self.assertEqual(result['payout_dfs'][1].iloc[0,0],35)
        self.assertEqual(result['balance_dfs']['second'].iloc[0,0],65)
        self.assertEqual(result['fiscal_loss'].iloc[0,0],60)
        self.assertEqual(result['fiscal_uncovered'].iloc[0,0],0)
        reverse=self.run_strategy([b,a],[200])
        self.assertEqual(reverse['payout_dfs'][0].iloc[0,0],60)
        self.assertEqual(reverse['balance_dfs']['first'].iloc[0,0],25)

    def test_zero_share_and_no_unnecessary_replenishment(self):
        cfg=ddo(initial_balance=100,contracted_amount=100,first_renewal_year=2029)
        result=self.run_strategy([cfg],[200,200])
        np.testing.assert_array_equal(result['payout_dfs'][0].values,[[60,40]])
        np.testing.assert_array_equal(result['balance_dfs']['DDO'].values,[[40,100]])
        zero=self.run_strategy([cfg],[200],0)
        self.assertEqual(zero['total_coverage'].iloc[0,0],0)
        self.assertEqual(zero['balance_dfs']['DDO'].iloc[0,0],100)

    def test_insurance_contract_uses_gross_loss(self):
        insurance=dict(name='insurance',type='insurance',attachment_point=100,exhaustion_point=300,ceding_percentage=.5)
        result=self.run_strategy([insurance],[200])
        self.assertEqual(result['total_coverage'].iloc[0,0],50)
        self.assertEqual(result['fiscal_uncovered'].iloc[0,0],10)

    def test_ccf_formula_uses_gross_loss_with_fiscal_cap(self):
        cfg=dict(name='CCF',type='ccf',ccf_threshold=100,ccf_person=1500,Pop_exposed=6.39e6,initial_balance=1000,contracted_amount=1000,first_renewal_year=2031,renewal_interval_years=5,max_replenishments_per_period=1)
        result=self.run_strategy([cfg],[200],.1)
        self.assertEqual(result['total_coverage'].iloc[0,0],20)
        self.assertEqual(result['balance_dfs']['CCF'].iloc[0,0],980)
        contractual=self.run_strategy([cfg],[200],1)
        expected=np.exp(0.001074*np.log(200*1000)**3.0883+7.9346)*1500/1e6
        self.assertAlmostEqual(contractual['total_coverage'].iloc[0,0],expected)

    def test_reporting_does_not_scale_payout_twice(self):
        row=strategy_statistics('example',pd.DataFrame([[200]]),pd.DataFrame([[50]]),.3,1)
        self.assertEqual(row['Total 1-year FISCAL uncovered median ($MM)'],10)
        self.assertNotIn('Total 1-year FISCAL financing gap median ($MM)', row)

    def test_invalid_factor_and_empty_strategy(self):
        for factor in [-.1,1.1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError): self.run_strategy([], [200],factor)
        result=self.run_strategy([], [200])
        self.assertEqual(result['fiscal_uncovered'].iloc[0,0],60)

    def test_config_priority(self):
        cfg=load_config(ROOT/'config.toml')
        self.assertEqual([x['type'] for x in cfg.instruments],['loan','ccf','ddo','ddo'])
        self.assertIn('BM',cfg.instruments[2]['name'])

if __name__=='__main__': unittest.main()
