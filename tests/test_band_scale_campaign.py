"""Verify shared native observations, fixed coefficients and strict restoration."""

from dataclasses import replace
from unittest import TestCase

import torch

from transfer.band_scale import BANDS, POLICIES, can_cross, condition, measure, validate_cross
from transfer.band_scale_fixture import open_fixture
from transfer.native_scale_training import Run, _exact


class BandScaleTests(TestCase):
    def test_actual_band_deletion_and_lattice_change_bind_selector_and_points(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        torch.set_num_threads(1)
        with TemporaryDirectory() as tmp:
            fixture,_ = open_fixture(Path(tmp),"spectral","row",64,"fixture")
            event = dict(tile=4,band="full",factor=1,condition="matched")
            full,_,targets = fixture.materialize("devval",event,"fixture")
            gap,_,members = fixture.materialize("devval",dict(event,band="central_gap",factor=2),"fixture")
            self.assertEqual(targets.points,members.points)
            self.assertEqual(len(full.view.bands.band_ids),8)
            self.assertEqual(list(gap.view.bands.band_ids),[103,110,145,152])
            self.assertEqual(gap.receipt["physical"]["nominal_gsd_m"],1.3)
            self.assertNotEqual(targets.summary()["mapping"],members.summary()["mapping"])
            self.assertNotEqual(full.receipt["source"]["retained_statistics"],gap.receipt["source"]["retained_statistics"])
            self.assertEqual(full.receipt["source"]["statistics"],gap.receipt["source"]["statistics"])
            descriptor = condition(full,"matched")
            with self.assertRaisesRegex(ValueError,"different native view"):
                descriptor.validate(gap,gap.receipt)
            with self.assertRaisesRegex(ValueError,"different native view"):
                replace(descriptor,gsd_m=99.).validate(full,full.receipt)
            model = fixture.model_factory(fixture.baseline(64),"joint",64).eval()
            self.assertTrue(can_cross(model))
            rows = {key:measure(model,gap,members,"fixture",policy=key,selector=condition(gap,"missing"))
                    for key in POLICIES}
            validate_cross(rows)
            self.assertEqual(rows["native"]["radii"],[1,2])
            self.assertEqual(rows["native"]["cnn_dilations"],[1,2,3])

    def test_train_resume_with_actual_exposure_rejects_other_layout(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        torch.set_num_threads(1)
        with TemporaryDirectory() as tmp, TemporaryDirectory() as foreign:
            fixture,_ = open_fixture(Path(tmp),"context","row",512,"fixture")
            separate,_ = open_fixture(Path(foreign),"context","column",512,"fixture")
            spec = fixture.contract(9,512,"joint",1,"fixture")
            run = Run(spec,"fixture",released=fixture)
            run.start()
            run.update(spec["training"][0])
            checkpoint = run.pack()
            same = Run(spec,"fixture",released=fixture)
            same.restore(checkpoint,1)
            run.update(spec["training"][1])
            same.update(spec["training"][1])
            self.assertTrue(_exact(run.pack(),same.pack()))
            with self.assertRaisesRegex(ValueError,"registration"):
                wrong = Run(separate.contract(9,512,"joint",1,"fixture"),"fixture",released=separate)
                wrong.restore(checkpoint,1)
