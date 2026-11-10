import copy,json,unittest
from unittest.mock import patch
import numpy as np
import torch
from evaluation.selected_signal_diagnostics import SelectedSignalPlan,review_selected_signals
from evaluation.linear_probe import fingerprint
from evaluation.graph_edge_intervention import _selected
from evaluation.block_input_diagnostics import _probe
from loadData.spatial_tiles import observe_tile
from transfer.graph_route_context import route_state_hash
from tests.test_graph_edge_intervention import completed


def spec(run):
    tid=next(t['id'] for t in run.record['roles'] if t['role']=='devval')
    obs=observe_tile(run.scene.cube,run.scene.plan,run.scene.stats,run.scene.plan['tiles'][tid])
    ids=obs.global_ids.ravel().tolist()
    return dict(tile_id=tid,target_ids=sorted([ids[12],ids[24]]),class_id=1,
                sources=sorted([[ids[0],0],[ids[12],0],[ids[-1],1]]),epsilon=.01)


class SelectedSignalsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture_run=completed('weight',False,3)

    def test_native_three_heads_manual_signal_and_unchanged_original(self):
        run=self.fixture_run;args=spec(run);before=fingerprint({'state':run.model.state_dict(),'source':run.source.state_dict(),
            'optimizer':run.optimizer.state_dict(),'history':run.history,'grad':[p.grad for p in run.model.parameters()]})
        rng=torch.get_rng_state().clone();plan=SelectedSignalPlan.freeze(run,**args)
        with patch('torch.optim.Adam.step',side_effect=AssertionError('no update')):
            result=review_selected_signals(run,plan,expected_plan_hash=plan.plan_hash)
        self.assertEqual(before,fingerprint({'state':run.model.state_dict(),'source':run.source.state_dict(),
            'optimizer':run.optimizer.state_dict(),'history':run.history,'grad':[p.grad for p in run.model.parameters()]}))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(result['whole_model_forwards'],8)
        self.assertEqual(result['optimizer_updates'],0)
        restored=SelectedSignalPlan(json.loads(json.dumps(plan.to_dict())),expected_hash=plan.plan_hash)
        self.assertEqual(result,review_selected_signals(run,restored,expected_plan_hash=plan.plan_hash))
        # Independently inject one global source via original whole-model forward interception.
        model=_selected(run);obs=observe_tile(run.scene.cube,run.scene.plan,run.scene.stats,run.scene.plan['tiles'][args['tile_id']])
        ids=obs.global_ids.ravel().tolist();source=args['sources'][0];local=ids.index(source[0]);y,x=divmod(local,obs.spectral.shape[1])
        native=model.forward;calls=[]
        def forward(dataset,value,*a,**kw):
            value=value.clone();value[y,x,source[1]]+=args['epsilon'];calls.append(1);return native(dataset,value,*a,**kw)
        model.forward=forward
        try:
            with torch.no_grad():out=run._predict(model,run.scene.plan['tiles'][args['tile_id']],0)
        finally:del model.forward
        targets=[ids.index(i) for i in args['target_ids']]
        self.assertEqual(len(calls),1)
        self.assertEqual(result['rows'][0]['probability_scores'][0],[float(p[targets,0].mean()) for p in out[:3]])
        self.assertTrue(all(np.isfinite(z) for r in result['rows'] for z in r['finite_difference']))

    def test_wrong_role_sources_selection_and_partial_refused_before_forward(self):
        run=self.fixture_run;args=spec(run);plan=SelectedSignalPlan.freeze(run,**args)
        with patch.object(run,'_predict',side_effect=AssertionError('must reject before forward')):
            for change in (lambda v:v.update(sources=[[10**9,0]]),lambda v:v.update(sources=[[args['sources'][0][0],999]]),
                           lambda v:v.update(sources=[args['sources'][0],args['sources'][0]]),lambda v:v.update(epsilon=float('inf'))):
                bad=copy.deepcopy(args);change(bad)
                with self.assertRaises(ValueError):SelectedSignalPlan.freeze(run,**bad)
            query=next((t['id'] for t in run.record['roles'] if t['role'] not in ('devtrain','devval')),None)
            if query is not None:
                with self.assertRaises(ValueError):SelectedSignalPlan.freeze(run,**{**args,'tile_id':query})
            changed=copy.deepcopy(run);changed.selected_state=copy.deepcopy(run.selected_state)
            key=next(k for k,v in changed.selected_state.items() if v.is_floating_point());changed.selected_state[key].add_(1)
            with self.assertRaises(ValueError):review_selected_signals(changed,plan,expected_plan_hash=plan.plan_hash)
        with self.assertRaises(ValueError):review_selected_signals(run,plan,expected_plan_hash='bad')
        with patch('tests.test_graph_edge_intervention.GraphRouteRun.advance',return_value=None):partial=completed('weight',False,3)
        with patch.object(partial,'_predict',side_effect=AssertionError('no partial model forward')):
            with self.assertRaises(ValueError):SelectedSignalPlan.freeze(partial,**spec(partial))

    def test_perturbed_native_routing_is_reported_and_errors_cleanup(self):
        run=self.fixture_run;args=spec(run);plan=SelectedSignalPlan.freeze(run,**args)
        report=review_selected_signals(run,plan,expected_plan_hash=plan.plan_hash)
        baseline=report['baseline_support']
        for row in report['rows']:
            self.assertEqual(row['topology_changed'],[s['edges']!=baseline['edges'] for s in row['supports']])
            self.assertEqual(row['q_changed'],[s['q_hash']!=baseline['q_hash'] for s in row['supports']])
            self.assertEqual(row['graph_values_changed'],[s['final_a_hash']!=baseline['final_a_hash'] for s in row['supports']])
        model=_selected(run);obs=observe_tile(run.scene.cube,run.scene.plan,run.scene.stats,run.scene.plan['tiles'][args['tile_id']])
        with patch.object(model.backbone,'forward_stages',side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):_probe(run,model,args['tile_id'],torch.tensor(obs.spectral.copy()),plan.to_dict()['base']['normalized_spectrum_hash'])
        for owner,name in ((model,'forward'),(model.backbone.learnable_graph,'build_graph'),(model.backbone.graph_router,'forward')):
            self.assertNotIn(name,owner.__dict__)


if __name__=='__main__':unittest.main()
