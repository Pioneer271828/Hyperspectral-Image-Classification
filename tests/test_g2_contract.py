import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from transfer.g2_contract import validate_contract, validate_lineage


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture():
    names = {'paviaU': ('ROSIS', 9), 'xuzhou': ('HySpex', 9), 'whu-honghu': ('Headwall', 22),
             'WHUL': ('Headwall', 9), 'whu-hanchuan': ('Headwall', 16)}
    catalog = {'policy': 'g2_family_inventory_v1', 'scenes': {
        n: {'family': family, 'class_count': count, 'evidence_level': 'verified_local_binding',
            'artifact_sha256': digest(n+'artifact'), 'family_evidence_sha256': digest(n+'family'),
            'class_table_sha256': digest(n+'table')} for n, (family, count) in names.items()}}
    sources = ['paviaU', 'xuzhou']
    contract = {'policy': 'g2_closed_set_v1', 'catalog_sha256': digest('catalog'),
                'target': 'whu-honghu', 'sources': sources, 'exclude': ['whu-honghu','WHUL','whu-hanchuan'],
                'semantic_classes': ['semantic_a','semantic_b'], 'semantic_evidence_sha256': digest('semantic'),
                'semantic_mapping': {n: {'1':'semantic_a','2':'semantic_b'} for n in sources+['whu-honghu']},
                'unmapped_policy': 'exclude_before_sampling_and_scoring',
                'source_releases': {n: {p: {'observation_sha256': digest(n+p+'obs'),
                                            'labels_sha256': digest(n+p+'labels')} for p in ('train','val')}
                                    for n in sources},
                'selection': 'source_validation_only'}
    checkpoint, parent = digest('checkpoint'), digest('parent')

    def event(scene, phase, partition):
        return dict(scene=scene, phase=phase, partition=partition, **contract['source_releases'][scene][partition])

    lineage = {'policy':'g2_source_lineage_v1','contract_sha256':digest('contract'),'selected':checkpoint,
               'records': {
                   parent: {'parents':[], 'access':[event('paviaU','source_train','train')], 'head':None},
                   checkpoint: {'parents':[parent], 'access':[event('xuzhou','source_train','train'),
                                                               event('xuzhou','source_select','val')],
                                'head':{'kind':'shared_semantic','ordered_classes':['semantic_a','semantic_b'],
                                        'state_sha256':digest('head')}}}}
    return catalog, contract, lineage, parent


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.catalog, self.contract, self.lineage, self.parent = fixture()

    def lineage_check(self):
        return validate_lineage(self.contract,self.lineage,digest('contract'))

    def test_valid_family_and_ancestor_closure(self):
        self.assertEqual(validate_contract(self.catalog,self.contract),{'WHUL','whu-hanchuan','whu-honghu'})
        self.assertEqual(self.lineage_check(),set(self.lineage['records']))

    def test_missing_sibling_and_false_alias_rejected(self):
        self.contract['exclude'].remove('WHUL')
        with self.assertRaisesRegex(ValueError,'whole target family'):
            validate_contract(self.catalog,self.contract)
        self.catalog['scenes']['WHUL']['family'] = 'other'
        with self.assertRaisesRegex(ValueError,'canonical family'):
            validate_contract(self.catalog,self.contract)

    def test_sibling_in_source_rejected(self):
        self.contract['sources'].append('WHUL')
        with self.assertRaisesRegex(ValueError,'whole target family'):
            validate_contract(self.catalog,self.contract)

    def test_nominal_release_and_unmapped_class_rejected(self):
        self.catalog['scenes']['xuzhou']['evidence_level'] = 'nominal_only'
        with self.assertRaisesRegex(ValueError,'Local family binding'):
            validate_contract(self.catalog,self.contract)
        self.catalog['scenes']['xuzhou']['evidence_level'] = 'verified_local_binding'
        self.contract['semantic_mapping']['xuzhou'] = {'1':'semantic_a','2':'semantic_a'}
        with self.assertRaisesRegex(ValueError,'incomplete or ambiguous'):
            validate_contract(self.catalog,self.contract)

    def test_source_train_and_val_releases_cannot_alias(self):
        releases = self.contract['source_releases']['xuzhou']
        releases['val'] = copy.deepcopy(releases['train'])
        with self.assertRaisesRegex(ValueError,'releases must differ'):
            validate_contract(self.catalog,self.contract)

    def test_target_family_in_ancestor_rejected(self):
        event = self.lineage['records'][self.parent]['access'][0]
        event['scene'] = 'WHUL'
        with self.assertRaisesRegex(ValueError,'non-source'):
            self.lineage_check()

    def test_missing_ancestor_and_cycle_rejected(self):
        self.lineage['records'][self.parent]['parents'] = [self.lineage['selected']]
        with self.assertRaisesRegex(ValueError,'Cyclic or incomplete'):
            self.lineage_check()
        self.lineage['records'].pop(self.parent)
        with self.assertRaisesRegex(ValueError,'Cyclic or incomplete'):
            self.lineage_check()

    def test_selection_on_train_or_with_target_label_hash_rejected(self):
        row = self.lineage['records'][self.lineage['selected']]['access'][-1]
        row['partition'] = 'train'
        with self.assertRaisesRegex(ValueError,'label access differs'):
            self.lineage_check()
        row['partition'] = 'val'
        row['labels_sha256'] = digest('different labels')
        with self.assertRaisesRegex(ValueError,'label access differs'):
            self.lineage_check()

    def test_independent_head_and_class_reordering_rejected(self):
        head = self.lineage['records'][self.lineage['selected']]['head']
        head['kind'] = 'dataset_specific'
        with self.assertRaisesRegex(ValueError,'shared head'):
            self.lineage_check()
        head['kind'] = 'shared_semantic'
        head['ordered_classes'].reverse()
        with self.assertRaisesRegex(ValueError,'semantic order'):
            self.lineage_check()

    def test_cli_byte_hashes_and_receipt(self):
        commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = [sys.executable,'-m','transfer.g2_contract','--expected-commit',commit]
            for name, value in (('catalog',self.catalog),('contract',self.contract),('lineage',self.lineage)):
                path = root/f'{name}.json'
                raw = json.dumps(value)
                path.write_text(raw,encoding='utf-8')
                checksum = digest(raw)
                args.extend(['--'+name,str(path),'--'+name+'-sha256',checksum])
                if name=='catalog':
                    self.contract['catalog_sha256']=checksum
                if name=='contract':
                    self.lineage['contract_sha256']=checksum
            out = root/'result'
            subprocess.run(args+['--out',str(out)],check=True,capture_output=True)
            receipt = json.loads((out/'receipt.json').read_text())
            self.assertEqual(receipt['status'],'passed_declared_contract')
            self.assertFalse(receipt['training_executable'])
            with (root/'catalog.json').open('a') as f:
                f.write('\n')
            rejected = root/'rejected'
            result = subprocess.run(args+['--out',str(rejected)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Pinned document changed',result.stderr)
            self.assertFalse(rejected.exists())


if __name__ == '__main__':
    unittest.main()
