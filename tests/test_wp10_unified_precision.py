"""Check release- and target-specific precision decisions."""

import unittest
from pathlib import Path
from unittest.mock import patch

from inference.wp10_precision_deployment import _choice, _load, _passes


class WP10UnifiedPrecisionTests(unittest.TestCase):
    def test_published_choices_keep_default_and_match_recovery_seed(self):
        qat = {"kind":"wp10_frozen_onednn_pair_v1"}
        self.assertEqual(_choice(qat, None, "native", None, None),
                         ("fp32","fp32","requested_fp32"))
        self.assertEqual(_choice(qat, "qat-both", "native", None, None),
                         ("qat-both","fp32","original_precision_three_head_guard"))
        self.assertEqual(_choice(qat, "ternary", "native", None, None),
                         ("ternary","fp32","original_precision_three_head_guard"))
        low_rank = {"kind":"wp10-low-rank"}
        for seed in (64,512,2048):
            name = f"s{seed}-svd-r16"
            self.assertEqual(_choice(low_rank, name, "native", None, None),
                             (name,f"s{seed}-dense","original_rank_three_head_guard"))
        with self.assertRaisesRegex(ValueError,"recovery seed"):
            _choice(low_rank, "svd-r16", "native", None, None)

    def test_integer_backend_requires_qualified_target_and_all_three_heads(self):
        qat = {"kind":"wp10_frozen_onednn_pair_v1"}
        self.assertEqual(_choice(qat,"qat-first","native","gpu",None)[1:],
                         ("fp32","cpu_integer_backend_only"))
        self.assertEqual(_choice(qat,"qat-first","slic",None,None)[1:],
                         ("fp32","integer_partition_not_qualified"))
        self.assertEqual(_choice(qat,"qat-first","native",None,None)[1:],
                         ("fp32","backend_source_unconfigured"))
        self.assertFalse(_passes({"fused":{"passes_metric_guards":True},
                                  "graph":{"passes_metric_guards":True},
                                  "cnn":{"passes_metric_guards":False}}))
        with self.assertRaisesRegex(ValueError,"three|fused"):
            _passes({"fused":{"passes_metric_guards":True}})

    def test_retained_rank_control_uses_its_original_state_digest_schema(self):
        pin = {"path":"/old", "sha256":"a"*64}
        archive = {"kind":"wp10-low-rank", "cases":{
            f"s{s}-dense":{"package":{"weights":pin,"sidecar":pin,"format":"fp32"}}
            for s in (64,512,2048)
        }}
        with patch("inference.wp10_precision_deployment.load_rank_package") as rank_loader:
            with patch("inference.wp10_precision_deployment.load_package") as qat_loader:
                _, adjusted = _load(Path("/capsule"),archive,"s512-dense",None)
            qat_loader.assert_not_called()
            rank_loader.assert_called_once_with(adjusted)
            self.assertTrue(adjusted["weights"]["path"].endswith("s512-dense.pt"))
