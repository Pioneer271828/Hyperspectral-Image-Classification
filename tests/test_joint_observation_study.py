import copy
import json
import tempfile
from pathlib import Path
import unittest
import torch
from joint_observation_study import load_study,evaluation_bank,condition_view
from joint_observation import freeze_settings,JointObservationRuntime
from joint_training import load_plan,options,schedule_for
from joint_baseline import load_config
from test_joint_observation import metadata
from test_calibrated_selection import items


class StudyTests(unittest.TestCase):
    def test_registered_budget_and_corrupted_plan(self):
        study=load_study();self.assertEqual(study['new_optimizer_updates'],6*1503)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'bad.json';study['updates_per_case']=5000;p.write_text(json.dumps(study))
            with self.assertRaises(ValueError):load_study(p)

    def test_full_budget_freezing_and_late_cursor(self):
        data=items();meta=metadata(data)
        for seed in (64,512,2048):
            base=options(load_plan('configs/experiments/base97_training.json'),'full',seed,
                         load_config('configs/experiments/base97_joint.json'))
            for preset in ('full','alternate75'):
                opts=freeze_settings(base,data,meta,preset=preset,sampling_seed=8128+seed,plan_hash='fixture')
                runtime=JointObservationRuntime(opts)
                self.assertEqual(len(opts['joint_observation']['records']),1503)
                for index in (0,4,5,1499,1502):
                    name=schedule_for(opts)[index]
                    view,_,record=runtime.view(index,name,data[name])
                    self.assertEqual(view['bands'],len(record['selected_positions']))
                    self.assertEqual(view['bands']==data[name]['bands'],preset=='full' or index%2==0)

    def test_eval_bank_is_fixed_and_views_preserve_coordinates_graph(self):
        study=load_study();data=items();meta=metadata(data)
        bank=evaluation_bank(study,meta,'fixture')
        self.assertEqual(bank,evaluation_bank(study,meta,'fixture'))
        for condition,records in bank.items():
            for name,record in records.items():
                view=condition_view(data[name],meta[name],record,'fixture')
                self.assertIs(view['A'],data[name]['A']);self.assertIs(view['Q'],data[name]['Q'])
                torch.testing.assert_close(view['net_input'][...,-2:],data[name]['net_input'][...,-2:],atol=0,rtol=0)
                if condition=='full':self.assertIs(view,data[name])
                else:self.assertLess(view['bands'],data[name]['bands'])
                with self.assertRaises(ValueError):condition_view(data[name],meta[name],record,'other')


if __name__=='__main__':unittest.main()
