"""Independent arithmetic and native campaign admission regressions."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from model.task_losses import probability_loss_components
from transfer import native_scale_campaign as campaign, native_scale_fixture as fixture
from transfer import native_scale_observation as native, native_scale_supervision as supervision
from transfer import native_scale_training as training, registered_development as transport
from transfer.spatial_multiscale import initialize
from transfer.xuzhou_multiscale_audit import _exact


class NativeScaleTests(TestCase):
    def test_rectangle_scalar_centers_footprints_projection(self):
        for box in ([2,5,3,8],[1,2,2,7]):
            y0,y1,x0,x1 = box
            for factor in (1,2,4):
                grid = native.Lattice.make(box,[12,13],factor)
                raw = np.arange((y1-y0)*(x1-x0),dtype=np.float32).reshape((y1-y0,x1-x0,1))
                expected = []
                for a,b,c,d in grid.cells:
                    cell = [float(raw[y-y0,x-x0,0]) for y in range(a,b) for x in range(c,d)]
                    expected.append(sum(cell)/len(cell))
                np.testing.assert_array_equal(native.reduce_area(raw,grid).ravel(),expected)
                self.assertEqual(int(grid.areas().sum()),raw.size)
                self.assertEqual(grid.footprints(.5)[0][0],(y0-.5)*.5)
                self.assertEqual(grid.footprints(.5)[-1][-1],(x1-.5)*.5)
                self.assertEqual(grid.project(np.arange(len(grid.cells))[:,None])[-1,0],len(grid.cells)-1)
                np.testing.assert_allclose(grid.coordinates()[0,0],grid.centers()[0]/[11,12],rtol=1e-7)
            with self.assertRaises(ValueError):
                native.Lattice.make(box,[12,13],2,edge="padding")

    def test_hard_loss_legacy_and_soft_oracle_gradients(self):
        logits = [torch.tensor([[0.0,1.0],[-0.4,0.3]],requires_grad=True) for _ in range(3)]
        probs = tuple(value.softmax(1) for value in logits)
        pure = supervision.Targets(np.array([[1,0],[0,1]]),np.array([1,1]),True)
        hard,weight = pure.weights("high_purity")
        original = probability_loss_components(probs,torch.eye(2),torch.ones(2,2))
        self.assertTrue(_exact(supervision.loss(probs,hard,weight),original))
        target = supervision.Targets(np.array([[3,1],[0,2]]),np.array([4,2]),True)
        mixed,weight = target.weights("mixture")
        result = supervision.loss(probs,mixed,weight)["total"]
        expected = sum((-3*p[0,0].log()-p[0,1].log()-2*p[1,1].log())/6 for p in probs)
        torch.testing.assert_close(result,expected,rtol=1e-7,atol=1e-7)
        for a,b in zip(torch.autograd.grad(result,logits,retain_graph=True),torch.autograd.grad(expected,logits),strict=True):
            torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
        with self.assertRaises(ValueError):
            supervision.Targets(np.array([[2,2]]),np.array([4]),True).weights("high_purity")
        with self.assertRaises(ValueError):
            supervision.Targets(np.array([[1,0]]),np.array([4]),False).weights("mixture")

    def test_complete_fixture_maps_and_physical_provenance(self):
        f = fixture.Fixture(22,"devtrain")
        a,labels,target = f.materialize(3,4,"even_retained_positions","b","code")
        self.assertEqual(len(a.view.bands.band_ids),12)
        self.assertEqual(a.receipt["physical"]["nominal_gsd_m"],4.4)
        self.assertGreater(a.receipt["physical"]["partial_cells"],0)
        self.assertTrue((a.view.observation.global_ids<0).all())
        with self.assertRaises(ValueError):
            a.view.observation.spectral.setflags(write=True)
        copied = a.receipt
        copied["source"]["positions"][0] = 999
        self.assertEqual(a.verify("code","devtrain")["source"]["positions"][0],0)
        self.assertTrue((target.histogram.sum(0)>0).all())
        with self.assertRaises(ValueError):
            target.histogram.setflags(write=True)
        self.assertTrue((target.histogram.sum(1)==target.area).all())
        self.assertEqual(np.bincount(labels.ravel())[1:].tolist(),target.summary()["known_per_class"])
        with self.assertRaises(ValueError):
            a.verify("other","devtrain")
        with self.assertRaises(ValueError):
            a.verify("code","devval")
        with self.assertRaises(ValueError):
            f.materialize(81,2,"full","a","code")
        with self.assertRaises(ValueError):
            f.materialize(3,4,"full","c","code")

    def test_real_request_snapshots_and_rejects_before_decode(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        f = fixture.Fixture(9,"devtrain")
        original = {"mode":"none"}
        actual,_,_ = f.materialize(3,1,"full","a","code")
        info = actual.receipt
        row = dict(id=3,role="devtrain",box=info["box"],samples=[])
        scene = SimpleNamespace(dataset="xuzhou",classes=9)
        release = SimpleNamespace(profile="nested",shape=(*info["scene_shape"],11))
        opened = Mock(release=release)
        opened.role_row.return_value = row
        context = dict(tile=3,role="devtrain",box=row["box"],samples={"count":0,"sha256":native.digest([])})
        def take(*args):
            self.assertEqual(args[1],{"mode":"none"})
            return scene,None,actual.view,context
        opened.materialize.side_effect = take
        with self.assertRaises(ValueError):
            native.from_bank(opened,"devtrain",3,original,2,"code",edge="padded")
        opened.materialize.assert_not_called()
        def change_request(*args):
            original["mode"]="uniform_stride"
            return row
        opened.role_row.side_effect = change_request
        real,_ = native.from_bank(opened,"devtrain",3,original,2,"code")
        self.assertIsNone(real.receipt["physical"]["nominal_gsd_m"])
        self.assertFalse(supervision.sparse_targets(real,row).summary()["trainable"])
        with self.assertRaises(ValueError):
            fixture.condition_for(real,"matched")

    def test_conditions_frozen_same_graph_and_derangement(self):
        f = fixture.Fixture(9,"devtrain")
        originals, donors = [],[]
        a = None
        for key in fixture.PROFILES:
            a,_,_ = f.materialize(3,2,"full",key,"code")
            match = fixture.condition_for(a,"matched")
            shuffled = fixture.condition_for(a,"permuted")
            self.assertNotEqual(match.source_key,shuffled.source_key)
            originals.append(match.values())
            donors.append(shuffled.values())
        self.assertCountEqual(originals,donors)
        model = initialize(training.baseline(9,64),"joint",7321)
        conditions = [fixture.condition_for(a,name) for name in fixture.INTERVENTIONS]
        outputs = [native.forward(model,a,"code","devtrain",condition=c) for c in conditions]
        self.assertEqual(len({g["q"] for _,g in outputs}),1)
        self.assertNotEqual(outputs[0][1]["weights"],outputs[2][1]["weights"])
        with patch.object(model.backbone.learnable_graph.__class__,"build_graph",
                          side_effect=AssertionError("invalid condition consumed")):
            with self.assertRaises(ValueError):
                native.forward(model,a,"code","devtrain",condition=replace(conditions[0],unit="km/pixel"))
            with self.assertRaises(ValueError):
                native.forward(model,a,"code","devtrain",condition=replace(conditions[0],gsd_m=8.0))

    def test_train_freeze_and_confirmation_reject_early(self):
        spec = training.contract(9,64,"joint","code",mode="high_purity")
        run = training.Run(spec,"code")
        event = deepcopy(spec["training"][0])
        spec["training"][0]["factor"] = 4
        self.assertEqual(run.registered["training"][0],event)
        with self.assertRaises(ValueError):
            training.Run(spec,"code")
        run.start()
        prior = run.pack()
        empty = supervision.Targets(np.array([[2,2]]),np.array([4]),True)
        with patch.object(run,"_materialize",return_value=(None,None,empty)),patch.object(
                run.optimizer,"step",side_effect=AssertionError("empty mask stepped")):
            with self.assertRaises(ValueError):
                run.update(event)
        self.assertTrue(_exact(prior,run.pack()))
        with self.assertRaises(ValueError):
            run.restore(dict(prior,registration="wrong"),0)
        with self.assertRaises(ValueError):
            fixture.Fixture(9,"confirmation")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/"bad.json"
            dummy = {"path":str(path),"sha256":"0"*64}
            pin = transport._json_new(path,dict(producer="code",matrix=dummy))
            with patch.object(fixture.Fixture,"_payload",side_effect=AssertionError("early confirmation read")):
                with self.assertRaises(ValueError):
                    campaign.confirm(root,pin,root/"conf","code")
                with self.assertRaises(ValueError):
                    campaign.confirm(root,pin,root,"code")
