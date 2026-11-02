import json
from pathlib import Path
import unittest
from scripts.run_development_budget import verify_control_settings
from train_spatial import SpatialConfig


class DevelopmentBudgetControlTests(unittest.TestCase):
    def test_archived_settings_without_lr_fields_and_conflicting_values(self):
        base=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/screen_m6_none.json').read_text())
        original=SpatialConfig(base)
        legacy={'seed':2046,'val_every':100,'budgets':[1000,5000],'test_evaluation':False}
        verify_control_settings(legacy,original,2046)
        for change in ({'backbone_lr':1e-5},{'head_lr':1e-3},{'sampling':'class_first'},
                       {'seed':2056},{'test_evaluation':True},{'val_every':50}):
            with self.subTest(change=change),self.assertRaises(AssertionError):
                verify_control_settings(legacy|change,original,2046)
