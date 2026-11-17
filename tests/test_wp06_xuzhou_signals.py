"""Contract and independent-native checks before the real full-block diagnostic."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

import research_runtime as rr
from evaluation.block_input_diagnostics import XUZHOU_SPECTRAL_CAP,input_gradients,intervene_spectrum
from evaluation.graph_edge_intervention import _selected
from evaluation.selected_signal_diagnostics import SelectedSignalPlan,review_selected_signals
from scripts.wp06_selected_signals import audit,runtime as rt
from tests.test_graph_edge_intervention import completed
from tests.test_selected_signal_diagnostics import spec


class XuzhouSignalWorkflowTests(unittest.TestCase):
    def test_full_population_center_far_tie_and_sorted_bands(self):
        tiles = [dict(id=i,partition="train",box=[0,27,0,27]) for i in range(78)]
        tiles.append(dict(id=78,partition="test",box=[0,27,0,27]))
        plan = dict(shape=[500,260],tiles=tiles)
        roles = [{**tile,"role":"devtrain" if i<73 else "devval" if i<78 else "test"}
                 for i,tile in enumerate(tiles)]
        rows = rt.request_population(plan,roles,list(range(1,10)),list(range(436)))
        row = rows[0]
        center,corner = row["anchors"]
        self.assertEqual(center["global_yx"],[13,13])
        self.assertEqual(corner["global_yx"],[0,0])
        self.assertEqual(center["mean_to_both_targets"],corner["mean_to_both_targets"])
        self.assertEqual(row["sources"],[[i,b] for i in row["target_ids"] for b in (0,217,435)])
        self.assertEqual(sum(len(row["requests"]) for row in rows),1404)
        for change in (roles[:-1],roles[:-2]+[{**roles[-2],"box":[1,28,0,27]},roles[-1]],
                       roles[:-1]+[{**roles[-1],"role":"devval"}]):
            with self.assertRaises(ValueError):
                rt.request_population(plan,change,list(range(1,10)),list(range(436)))

    def test_exact_explicit_large_spectral_cap_and_analytical_gradients(self):
        shape = (28,28,436)
        x = torch.zeros(shape,dtype=torch.float32)
        with self.assertRaises(ValueError):
            intervene_spectrum(x,{"kind":"identity"})
        self.assertTrue(torch.equal(intervene_spectrum(x,{"kind":"identity"},spectral_cap=XUZHOU_SPECTRAL_CAP),x))
        def model(value):
            half = .5 + value.reshape(-1,436)[:,217]/8
            p = torch.stack((half,1-half),dim=1)
            return p,p,p
        result = input_gradients(model,x,pixel_indices=[0,783],class_id=1,spectral_cap=XUZHOU_SPECTRAL_CAP)
        expected = torch.zeros_like(x)
        expected[0,0,217]=expected[-1,-1,217]=1/16
        self.assertTrue(all(torch.equal(grad,expected) for grad in result["gradients"]))
        for bad in (torch.zeros(29,27,436),torch.zeros(28,28,435),torch.zeros(shape,dtype=torch.float64)):
            with self.assertRaises(ValueError):
                intervene_spectrum(bad,{"kind":"identity"},spectral_cap=XUZHOU_SPECTRAL_CAP)

    def test_independent_native_observation_and_distinct_stage_graphs(self):
        run = completed("weight",False,3)
        request = spec(run)
        plan = SelectedSignalPlan.freeze(run,**request)
        measured = review_selected_signals(run,plan,expected_plan_hash=plan.plan_hash)
        tid = request["tile_id"]
        row = {k:request[k] for k in ("tile_id","target_ids","sources")}
        row["role"] = run.record["roles"][tid]["role"]
        row["requests"] = [{k:request[k] for k in ("class_id","epsilon")}]
        baseline,reference = audit.reference_tile(run,_selected(run),row,rr)
        result = {"tile_id":tid,"role":row["role"],"rows":[{"request":row["requests"][0],
                  "plan":plan.to_dict(),"result":measured}]}
        checked = audit.compare_block(row,result,baseline,reference)
        self.assertEqual(checked["directional_scores"],18)
        self.assertTrue(baseline["stage_graphs_distinct"])
        altered = deepcopy(result)
        altered["rows"][0]["result"]["rows"][0]["input_deltas"].reverse()
        with self.assertRaisesRegex(ValueError,"positive/negative"):
            audit.compare_block(row,altered,baseline,reference)
        with self.assertRaisesRegex(ValueError,"protected context"):
            audit.reference_tile(run,_selected(run),{**row,"role":"test"},rr)

    def test_pinned_driver_and_output_exclusive_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"receipt.json"
            pin = rt.write(path,{"result":"first"})
            with self.assertRaises(FileExistsError):
                rt.write(path,{"result":"overwrite"})
            with self.assertRaisesRegex(ValueError,"Changed pinned file"):
                rt.read({**pin,"sha256":"0"*64})
