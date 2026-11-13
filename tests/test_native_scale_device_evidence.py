"""Negative controls for read-only qualification, recovery and cost consumers."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from research_runtime import hash_file
from transfer import p4_cuda_qualification as qual, p4_runtime_device as p4
from transfer import native_scale_device_campaign as campaign, native_scale_device_evidence as evidence
from transfer import registered_development as transport


class DeviceEvidenceTests(TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.n = 0

    def save(self,body):
        self.n += 1
        return transport._json_new(self.root/f"{self.n}.json",body)

    def qualification(self):
        geo = self.save(dict(schema=campaign.SCHEMA,producer="original",shapes=[[1,1]]))
        result = dict(compatible=True,max_tolerance_ratio=.25)
        rows = {}
        for name,_ in campaign.operator_specs([[1,1]]):
            grads = qual.GRADIENTS["pool" if name.endswith("/adaptive_pool") else "sparse"]
            rows[name] = {key:{x:dict(result) for x in ("replacement_vs_math64","native_cpu32_vs_math64",
                     "replacement_vs_native_cpu32")} for key in ("forward",*grads)}
        case = next(name for name in rows if name.endswith("/assignment"))
        rows[case]["weights"]["replacement_vs_native_cpu32"] = dict(compatible=False,max_tolerance_ratio=1.2)
        num = dict(protocol=p4.CPU,device="cpu",physical_uuid=None,input_roster=None,
                   p4_arithmetic_sha256=hash_file(p4.__file__),deterministic_algorithms=False,
                   deterministic_warn_only=False)
        return dict(schema=campaign.SCHEMA,producer="original",geometry=geo,numeric=num,rows=rows,
                    scope_restored=True,fixture_count=len(rows),tolerances=dict(rtol=qual.RTOL,atol=qual.ATOL),
                    kernels=dict(adaptive_avg_pool2d=0,sparse_mm=0),failures=[case],oracle_failures=[],
                    updates=0,real_updates=0)

    def test_qualification_failures_cannot_be_hidden_or_rebound(self):
        value = self.qualification()
        pin = self.save(value)
        result = evidence.qualification(self.root,pin,"original",value["numeric"],value["geometry"])
        self.assertEqual((result["operator_math"],result["native_CPU_compatibility"]),("passed","failed"))
        hidden = dict(value,failures=[])
        with self.assertRaisesRegex(ValueError,"failure roster"):
            evidence.qualification(self.root,self.save(hidden),"original")
        fake = deepcopy(value)
        bad = fake["rows"][value["failures"][0]]["weights"]["replacement_vs_native_cpu32"]
        bad["compatible"] = True
        with self.assertRaisesRegex(ValueError,"failure bit"):
            evidence.qualification(self.root,self.save(fake),"original")
        with self.assertRaisesRegex(ValueError,"environment"):
            evidence.qualification(self.root,pin,"original",dict(value["numeric"],host="wrong"))
        with self.assertRaisesRegex(ValueError,"geometry"):
            evidence.qualification(self.root,pin,"original",geometry=self.save({}))
        Path(pin["path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError,"identity changed"):
            evidence.qualification(self.root,pin,"original")

    def test_partial_audit_cannot_appear_complete(self):
        value = self.qualification()
        q = self.save(value)
        numeric = dict(value["numeric"],input_roster="fixed")
        registered = dict(numeric=numeric,qualification=q,qualification_all_comparisons_passed=False)
        data = self.save(dict(schema=campaign.SCHEMA,producer="original",numeric=numeric,matrix=self.save({}),
                              real_updates=0,confirmation_reads=0,updates=209,endpoint_replays=0,cases={}))
        with patch.object(evidence,"matrix",return_value=registered):
            with self.assertRaisesRegex(ValueError,"all 48"):
                evidence.read_audit(self.root,data,"original")
            registered["qualification_all_comparisons_passed"] = True
            with self.assertRaisesRegex(ValueError,"hides qualification"):
                evidence.read_audit(self.root,data,"original")

    def test_cost_oracle_and_protocol_scope_guards(self):
        answer = evidence.summarize_cost([5,3,1,4,2],12)
        self.assertEqual(answer["p50_seconds"],3)
        self.assertAlmostEqual(answer["p95_seconds"],4.8)
        self.assertEqual(answer["native_pixels_per_second"],4)
        for sample in ([],[1,2,3],[1,2,3,4,float("nan")],[0]*5):
            with self.assertRaises(ValueError):
                evidence.summarize_cost(sample,12)
        cpu = dict(numeric=dict(protocol=p4.CPU),producer="original",geometry={},rows=[])
        with self.assertRaisesRegex(ValueError,"strict CUDA"):
            evidence.compare(self.root,cpu,cpu)
        gpu = dict(cpu,numeric=dict(protocol=p4.CUDA_FP64_WEIGHT),geometry={"wrong":True})
        with self.assertRaisesRegex(ValueError,"scope"):
            evidence.compare(self.root,cpu,gpu)
        row = dict(scene="xuzhou",tile=0,band="full",factor=1,arm="off",observation={},arrays={})
        cpu["rows"] = [row]
        gpu.update(geometry={},rows=[row])
        with patch.object(evidence,"_arrays",return_value={"fused":[],"cnn":[]}):
            with self.assertRaisesRegex(ValueError,"omit a head"):
                evidence.compare(self.root,cpu,gpu)

    def test_original_comparison_receipt_matches_recomputed_rows(self):
        cpu, gpu = self.save({}), self.save({})
        case = ("xuzhou",0,"full",1,"off")
        metrics = dict(fused=dict(compatible=False))
        parity = dict(failures=[case],rows=[dict(case=case,tolerance=metrics,argmax_changes={"fused":1},
                                                 sparse_structure_exact=True)])
        original = dict(schema=campaign.SCHEMA,producer="original",cpu=cpu,gpu=gpu,
                        tolerances=dict(rtol=qual.RTOL,atol=qual.ATOL),real_updates=0,query=0,
                        failures=[case],rows=[dict(identity=case,comparisons=metrics,class_changes={"fused":1},
                                                   sparse_structure_exact=True)])
        evidence._verify_comparison(self.root,self.save(original),"original",cpu,gpu,parity)
        with self.assertRaisesRegex(ValueError,"comparison receipt differs"):
            evidence._verify_comparison(self.root,self.save(dict(original,failures=[])),"original",cpu,gpu,parity)
