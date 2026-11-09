import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from joint_supervision_study import load_study, case_settings, select_case
from joint_supervision import SupervisionRuntime, supervision_settings
from joint_training import load_plan, run_training, schedule_for
from joint_baseline import load_config
from scripts.train_joint_m4 import graph_binding
from evaluation.joint_finalization import finalize
from model.research_config import digest
from test_calibrated_selection import items
from test_joint_training import settings, PROVENANCE, hashes

DEVICE = os.environ.get('M4S2_DEVICE', 'cpu')


class SupervisionStudyTests(unittest.TestCase):
    def launch(self):
        study = load_study(); plan = load_plan('configs/experiments/base97_training.json')
        base = load_config('configs/experiments/base97_joint.json')
        cases = []
        for mode in study['modes']:
            for supervision in study['supervision']:
                for seed in study['training_seeds']:
                    opts = case_settings(study, plan, base, mode, supervision, seed)
                    self.assertEqual(len(schedule_for(opts)), 1503)
                    cases.append({'mode': mode, 'supervision': supervision, 'seed': seed,
                                  'out': f'/{mode}/{supervision}/{seed}', 'settings': opts})
        return {'code': {'commit': 'fixture'}, 'study': study, 'cases': cases}

    def test_full_factorial_budget_and_valid_selection(self):
        launch = self.launch()
        self.assertEqual(sum(c['settings']['optimizer_steps'] for c in launch['cases']), 18036)
        for case in launch['cases']:
            selected = select_case(launch, launch['code'], load_study(), case['mode'], case['supervision'], case['seed'])
            self.assertEqual(selected, case)
            SupervisionRuntime(case['mode'], case['supervision']).validate(case['settings'])

    def test_incomplete_duplicate_output_and_changed_source_rejected(self):
        launch = self.launch()
        bads = []
        bad = copy.deepcopy(launch); bad['cases'].pop(); bads.append(bad)
        bad = copy.deepcopy(launch); bad['cases'][-1] = bad['cases'][0]; bads.append(bad)
        bad = copy.deepcopy(launch); bad['cases'][-1]['out'] = bad['cases'][0]['out']; bads.append(bad)
        bad = copy.deepcopy(launch); bad['code']['commit'] = 'changed'; bads.append(bad)
        bad = copy.deepcopy(launch); bad['study']['updates_per_case'] = 5000; bads.append(bad)
        for bad in bads:
            with self.assertRaises(ValueError): select_case(bad, launch['code'], load_study(), 'none', 'primary_only', 64)

    def test_modified_budget_seed_and_policy_rejected(self):
        study = load_study()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bad.json'; bad = copy.deepcopy(study); bad['training_seeds'] = [1, 2, 3]
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): load_study(path)
        plan = load_plan('configs/experiments/base97_training.json'); base = load_config('configs/experiments/base97_joint.json')
        for seed in (65, True, 64.0):
            with self.assertRaises(ValueError): case_settings(study, plan, base, 'none', 'primary_only', seed)

    def test_primary_finalization_scoped_heads_no_update_and_immutable_resume(self):
        data = items(); opts = supervision_settings(settings(), 'x_residual', 'primary_only')
        runtime = SupervisionRuntime('x_residual', 'primary_only')
        files = {'fixture.py': 'fixture'}
        source = {'commit': 'a'*40, 'branch': 'fixture', 'files': files, 'source_hash': digest(files), 'dirty': False}
        prov = {**PROVENANCE, 'm4_graph_binding': graph_binding(data)}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/'case'
            run_training(opts, data, folder, prov, source, device=DEVICE, runtime_adapter=runtime)
            (folder/'result.json').unlink()
            before = {k: v for k, v in hashes(folder).items() if k.endswith(('.pt', '.pt.json'))}
            result = finalize(folder, opts, data, prov, source, source, runtime, device=DEVICE)
            self.assertEqual(result['updates_this_call'], 0)
            self.assertEqual(before, {k: v for k, v in hashes(folder).items() if k in before})
            for phases in result['result']['selected_validation']['metrics'].values():
                for metric in phases.values(): self.assertEqual(set(metric), {'fused'})
            receipt = json.loads((folder/'finalization-recovery.json').read_text())
            self.assertEqual({r['head'] for r in receipt['comparisons']}, {'fused'})
            before = hashes(folder)
            self.assertEqual(run_training(opts, data, folder, prov, source, device=DEVICE, resume=True,
                             runtime_adapter=runtime)['updates_this_call'], 0)
            self.assertEqual(before, hashes(folder))


if __name__ == '__main__': unittest.main()
