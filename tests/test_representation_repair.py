"""Analytic directional derivatives and the official residual repair contract."""

from dataclasses import replace
import math
from unittest import TestCase

import torch

from evaluation.representation_localization import Feature, LayerIdentity, local_response, retained_coordinates
from transfer.representation_bypass import ResidualBottleneckClassifier, observation
from transfer.representation_campaign import ReleasedFixture, Run


class RepairTests(TestCase):
    def test_linear_and_nonlinear_local_response(self):
        key = LayerIdentity("model","fixture","pixels","residual","synthetic_confirmation")
        base = Feature(key,"observed",torch.tensor([[.25]],dtype=torch.float64))
        donor = Feature(key,"injected",torch.tensor([[.75]],dtype=torch.float64))
        def models(nonlinear):
            def predict(x):
                p=(torch.sigmoid(x) if nonlinear else .25+.5*x)
                probabilities=torch.cat((p,1-p),dim=1)
                return (probabilities,)*3
            return predict
        for variant in (False,True):
            with self.subTest(nonlinear=variant):
                result=local_response(models(variant),base,donor,pixels=[0],truth=[1],competitors=[2])
                expect=.5
                if variant:
                    p=1/(1+math.exp(-.25))
                    expect=p*(1-p)
                row=result["slope"]["fused"]
                self.assertAlmostEqual(row["directional"][0],expect,places=11)
                self.assertLess(row["absolute_error"][0],1e-6)
        same=local_response(models(True),base,base,pixels=[0],truth=[1],competitors=[2])
        self.assertEqual(same["base"],same["donor"])

    def test_original_band_projection_and_foreign_coordinates_fail(self):
        site=LayerIdentity("state","scene","yx","sensor","devval")
        full=Feature(site,"full",torch.tensor([[1.,2.,3.]]))
        small=Feature(site,"gap",torch.tensor([[3.,1.]]))
        self.assertTrue(torch.equal(retained_coordinates(full,(101,103,109),small,(109,101)).tensor,small.tensor))
        for source,ids in ((replace(small,identity=replace(site,coordinates="other")),(109,101)),(small,(110,101))):
            with self.assertRaisesRegex(ValueError,"projection"):
                retained_coordinates(full,(101,103,109),source,ids)

    def test_realized_residual_and_observation_labels_are_orthogonal(self):
        torch.set_num_threads(1)
        window=observation("recoverable","vertical",64,"confirmation",0,"gap")
        model=ResidualBottleneckClassifier(64,"aligned").eval()
        packet=model.split(window)
        self.assertTrue(torch.allclose(packet.region_at_pixels,packet.region_at_pixels[0].expand_as(packet.region_at_pixels)))
        self.assertGreater(packet.residual.square().sum().item(),0)
        self.assertTrue(torch.allclose(packet.residual.square().sum(),packet.residual[window.shuffle].square().sum()))
        absent=observation("removed","vertical",64,"confirmation",0,"gap")
        self.assertEqual(torch.count_nonzero(model.split(absent).residual),0)
        bogus=replace(window,label=(window.label+1)%3)
        for a,b in zip(model(window),model(bogus),strict=True):
            self.assertTrue(torch.equal(a,b))

    def test_optimizer_resume_matches_and_rejects_foreign_view(self):
        torch.set_num_threads(1)
        fixture=ReleasedFixture("recoverable","horizontal",512)
        baseline=Run(fixture,"aligned","scope")
        baseline.validate(0)
        baseline.update()
        sealed=baseline.pack()
        replay=Run(fixture,"aligned","scope")
        replay.restore(sealed,1)
        baseline.update()
        replay.update()
        self.assertEqual(baseline.pack()["state"],replay.pack()["state"])
        with self.assertRaisesRegex(ValueError,"identity"):
            Run(ReleasedFixture("recoverable","vertical",512),"aligned","scope").restore(sealed,1)
