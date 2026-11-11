"""Fixed operator mathematics and qualification identity for FP64 edge accumulation."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from transfer import p4_coverage_batch as batch, p4_cuda_qualification as qual, p4_runtime_device as device
from transfer import p4_full_budget_registration as registration


class StableSparseWeightTests(unittest.TestCase):
    def test_bounded_fp64_dots_against_fsum_and_finite_difference(self):
        rng = np.random.default_rng(14071)
        edges = device.WEIGHT_CHUNK + 11
        for channels in (64, 128):
            features = torch.from_numpy(rng.normal(size=(channels, 19)).astype(np.float32)).T
            sensitivity = torch.from_numpy(rng.normal(size=(channels, 13)).astype(np.float32)).T
            rows = torch.arange(edges) % 13
            cols = torch.arange(edges) % 19
            self.assertFalse(features.is_contiguous())
            self.assertFalse(sensitivity.is_contiguous())
            actual = device.sparse_weight_gradient64(sensitivity, features, rows, cols)
            sample = (0, 97, device.WEIGHT_CHUNK-1, device.WEIGHT_CHUNK, edges-1)
            for index in sample:
                a, b = sensitivity[rows[index]].numpy(), features[cols[index]].numpy()
                expected = math.fsum(float(x)*float(y) for x,y in zip(a,b,strict=True))
                self.assertAlmostEqual(actual[index].item(), expected, places=12)
                # Independent double forward objective of one sparse edge, at +/- epsilon.
                eps = 2**-12
                plus = np.dot((.5+eps)*b.astype(np.float64), a.astype(np.float64))
                minus = np.dot((.5-eps)*b.astype(np.float64), a.astype(np.float64))
                self.assertAlmostEqual(actual[index].item(), (plus-minus)/(2*eps), places=10)

    def test_noncontiguous_coo_duplicates_transpose_zero_and_unused_gradients(self):
        rng = np.random.default_rng(7345)
        base = np.array([[0, 0, 0, 2, 2, 3], [0, 1, 1, 3, 3, 4]])
        for indices, shape in ((base, (5,6)), (base[::-1].copy(), (6,5))):
            for weight_grad, feature_grad in ((True,True),(True,False),(False,True)):
                weights = torch.tensor(rng.uniform(.05,.5,len(indices[0])).astype(np.float32),requires_grad=weight_grad)
                x = torch.from_numpy(rng.normal(size=(64,shape[1])).astype(np.float32)).T.requires_grad_(feature_grad)
                dy = torch.from_numpy(rng.normal(size=(64,shape[0])).astype(np.float32)).T
                dy[0] = 0
                graph = torch.sparse_coo_tensor(torch.from_numpy(indices), weights, shape)
                y = device.sparse_row_product_weight64(graph, x)
                y.backward(dy)
                forward, edge, dense = qual.sparse_reference(indices,weights.detach().numpy(),x.detach().numpy(),dy.numpy())
                self.assertTrue(qual.compare(y.detach().numpy(),forward)['compatible'])
                if weight_grad:
                    np.testing.assert_array_equal(weights.grad.numpy(),edge.astype(np.float32))
                else:
                    self.assertIsNone(weights.grad)
                if feature_grad:
                    self.assertTrue(qual.compare(x.grad.numpy(),dense)['compatible'])
                else:
                    self.assertIsNone(x.grad)
                self.assertEqual(y.dtype,torch.float32)

    def test_unchanged_three_way_gates_and_fixed_geometry(self):
        config = {'tile_size': 28, 'model': {'learned_assignment_radius': 1, 'learned_graph_radius': 1}}
        shapes = ((8,23),(8,28),(28,18),(28,23),(28,28))
        allowed = {'train':[{'box':[0,h,0,w]} for h,w in shapes],'val':[]}
        contract = qual.geometry_contract({'shape':[92,107],'plan_hash':'fixture'}, allowed,config)
        # Pin the old 8x23 fixture and the newly observed 28x23 shape in one fixed stream.
        self.assertIn([8,23],contract['shapes'])
        self.assertIn([28,23],contract['shapes'])
        results, patterns = qual.run_fixtures(contract,'cpu',device.CUDA_FP64_WEIGHT)
        self.assertEqual(patterns,qual.expected_patterns(contract))
        self.assertTrue(all(qual.check_comparisons(name,row) for name,row in results.items()))

    def test_v4_receipt_binds_source_protocol_device_and_dtype(self):
        geometry = {'geometry_sha256':'test','shapes':[[8,23]],'token_stride':10,'assignment_radius':1,
                    'adjacency_radius':1,'pool_channels':64}
        results, patterns = qual.run_fixtures(geometry,'cpu',device.CUDA_FP64_WEIGHT)
        code = {'commit':'candidate','source_hash':'source'}
        gpu = 'GPU-11111111-1111-1111-1111-111111111111'
        row = {'schema':qual.WEIGHT64_SCHEMA,'status':'passed',**code,'parent_sha256':batch.PARENT_SHA,
               'geometry':geometry,'tolerances':{'rtol':qual.RTOL,'atol':qual.ATOL},
               'numeric':{'protocol':device.CUDA_FP64_WEIGHT,'device':'cuda:0','physical_uuid':gpu,
                          'deterministic_algorithms':True,'deterministic_warn_only':False,'tf32':False,
                          'float32_matmul_precision':'highest','graph_kernels':'adaptive_bin_mean_and_sparse_row_product_weight64_v4',
                          'sparse_weight_contract':device.FP64_WEIGHT_CONTRACT},
               'optimizer_updates':0,'query_accessed':False,'raw_gt_loaded':False,'scope_restored':True,
               'results':results,'coo_patterns':patterns,'fixture_seed':qual.FIXTURE_SEED,
               'interceptions':{'adaptive_avg_pool2d':1,'sparse_mm':6}}
        self.assertEqual(qual.check_receipt(row,device.CUDA_FP64_WEIGHT,code,geometry),gpu)
        for change in (lambda r:r.update(schema=qual.SCHEMA,status='failed'),
                       lambda r:r.update(source_hash='different'),
                       lambda r:r['numeric'].update(physical_uuid='unknown'),
                       lambda r:r['numeric']['sparse_weight_contract'].update(weight_gradient_accumulation_dtype='torch.float32')):
            wrong = deepcopy(row)
            change(wrong)
            with self.assertRaises(ValueError):
                qual.check_receipt(wrong,device.CUDA_FP64_WEIGHT,code,geometry)
        from research_runtime import write_json

        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'qualified.json'
            write_json(path,row)
            other = 'GPU-22222222-2222-2222-2222-222222222222'
            with self.assertRaisesRegex(ValueError,'unregistered GPU'):
                registration.qualification_bindings([path],[other],device.CUDA_FP64_WEIGHT,code,geometry)

    def test_v4_scope_restores_settings_on_exception(self):
        import os

        gpu = 'GPU-11111111-1111-1111-1111-111111111111'
        props = SimpleNamespace(name='mocked', major=8, minor=9)
        initial = qual.runtime_settings()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':gpu,'CUBLAS_WORKSPACE_CONFIG':device.WORKSPACE}), \
                patch.object(device,'gate',return_value={'driver':'test'}), \
                patch.multiple(torch.cuda,is_available=lambda:True,device_count=lambda:1,
                               set_device=lambda index:None,get_device_properties=lambda index:props):
            with self.assertRaisesRegex(RuntimeError,'leave the worker'):
                with device.arithmetic(device.CUDA_FP64_WEIGHT,gpu) as numeric:
                    self.assertTrue(torch.are_deterministic_algorithms_enabled())
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertEqual(numeric['sparse_weight_contract'],device.FP64_WEIGHT_CONTRACT)
                    raise RuntimeError('leave the worker')
        self.assertEqual(qual.runtime_settings(),initial)

    def test_admission_retains_all_class_counts_and_failed_recall(self):
        from transfer.p4_full_budget_campaign import admission_validation

        def audit(arm, matrix):
            recall = (np.diag(matrix)/5).tolist()
            score = {**dict.fromkeys(('oa','aa','kappa','macro_f1','nll'),.5),
                     'recall':recall,'confusion':matrix.tolist()}
            return {'case':{'arm':arm,'id':arm,'pair_sha256':'paired'},'selected_step':30,
                    'fit':{'selected':{'val':{'heads':{k:score for k in ('fused','graph','cnn')}}}}}
        baseline = np.eye(22,dtype=np.int64)*5
        candidate = baseline.copy()
        candidate[-1,-1],candidate[-1,0] = 0,5
        a,b = admission_validation([audit('none',baseline),audit('anchor',candidate)])
        self.assertEqual(a['heads']['fused']['class_support'],[5]*22)
        self.assertEqual(b['heads']['fused']['zero_recall_classes'],[22])
        self.assertEqual(b['heads']['fused']['recall_drop_over_2pp_classes'],[22])

    def test_new_real_scope_requires_recovery_and_single_card_scope_is_synthetic(self):
        gpu = 'GPU-11111111-1111-1111-1111-111111111111'
        other = 'GPU-22222222-2222-2222-2222-222222222222'
        registration.check_budget(registration.RECOVERY, device.CUDA_FP64_WEIGHT, [gpu], True, 60, 30)
        for gpus, synthetic in (([gpu],False),([gpu,other],True)):
            with self.assertRaises(ValueError):
                registration.check_budget(registration.RECOVERY,device.CUDA_FP64_WEIGHT,gpus,synthetic,60,30)
        with self.assertRaisesRegex(ValueError,'accepted physical recovery'):
            registration.bind_recovery(None,registration.ADMISSION,device.CUDA_FP64_WEIGHT,False,{})


@contextmanager
def synthetic_schema(classes):
    # Only this test process and its worker processes use synthetic class counts.
    from loadData.dataset_registry import DATASETS
    from transfer import p4_exposure, p4_guarded

    scene = p4_guarded.SCENE
    file, variable, _, sensor = DATASETS[scene]
    schedule = partial(p4_exposure.schedule, classes=classes)
    with patch.dict(DATASETS, {scene: (file, variable, classes, sensor)}), patch.object(p4_exposure,'schedule',schedule):
        yield


def bounded_recovery(out, code, gpu, qualification):
    from research_runtime import external_directory, hash_file, write_json
    from tests.test_p4_coverage import fixture
    from transfer import p4_full_budget_campaign as campaign

    out = external_directory(out)
    out.mkdir(exist_ok=False)
    protocol, gpus = (device.CUDA_FP64_WEIGHT,[gpu]) if gpu else (device.CPU,[])
    cases = []
    for classes in (9,22):
        with synthetic_schema(classes):
            inputs = out/f'classes-{classes}-inputs'
            inputs.mkdir()
            parent, _, _ = fixture(inputs, classes=classes)
            prior = inputs/'prior.json'
            write_json(prior, {'status':'passed','query_accessed':False,'protected_files':{str(parent):hash_file(parent)}})
            scope = registration.prepare(parent,hash_file(parent),prior,hash_file(prior),
                out/f'classes-{classes}-registration',code,protocol=protocol,gpus=gpus,
                synthetic=True,purpose=registration.RECOVERY,recovery_classes=classes,
                qualifications=(qualification,) if qualification else ())
            sha = hash_file(scope)
            execution = out/f'classes-{classes}-execution'

            def launcher(command, log_dir, case, operation, locks):
                wrapped = [*command[:2], 'tests.test_p4_stable_cuda', '--worker-classes', str(classes), *command[3:]]
                return campaign.invoke(wrapped,log_dir,case,operation,locks)

            state = campaign.run(scope,sha,execution,code,launcher=launcher)
            if state['status'] != 'completed':
                return {'status':state['status'],'classes':classes,'execution':str(execution),
                        'optimizer_updates':120*len(cases)+state['optimizer_updates_this_invocation']}
            acceptance = campaign.accept(scope,sha,execution,code)
            cases.append({'classes':classes,'scope':str(scope),'scope_sha256':sha,
                          'acceptance':str(acceptance),'acceptance_sha256':hash_file(acceptance)})
    receipt = out/'recovery.json'
    row = {'schema': registration.RECOVERY_SCHEMA, 'status':'passed','commit':code['commit'],
        'source_hash':code['source_hash'],'physical_uuid':gpu,'protocol':protocol,
        'optimizer_updates':240,'completed_resume_updates':0,'cases':cases}
    write_json(receipt,row)
    if gpu:
        batch.check_pins(registration.bind_recovery(receipt,registration.ADMISSION,protocol,False,code))
    return {**row,'path':str(receipt),'sha256':hash_file(receipt)}


if __name__ == '__main__':
    if sys.argv[1:2] == ['--worker-classes']:
        from transfer import p4_full_budget_campaign as campaign

        classes = int(sys.argv[2])
        device.require(classes in (9,22), 'Unknown recovery fixture class count')
        del sys.argv[1:3]
        with synthetic_schema(classes):
            campaign.main()
    else:
        from research_runtime import code_snapshot, require_frozen_code

        parser = argparse.ArgumentParser(description='Run the registered synthetic paired recovery with one physical GPU or CPU.')
        parser.add_argument('--expected-commit',required=True)
        parser.add_argument('--out',type=Path,required=True)
        parser.add_argument('--gpu')
        parser.add_argument('--qualification',type=Path)
        args = parser.parse_args()
        code = code_snapshot()
        require_frozen_code(code)
        device.require(code['commit']==args.expected_commit and bool(args.gpu)==bool(args.qualification),
                       'Exact source and physical qualification required')
        result = bounded_recovery(args.out,code,args.gpu,args.qualification)
        print(json.dumps(result),flush=True)
        sys.exit(int(result['status'] != 'passed'))
