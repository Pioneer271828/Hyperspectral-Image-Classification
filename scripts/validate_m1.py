"""Read-only real-data M1 validation; no labels enter sampling and no accuracy is evaluated."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    from datetime import datetime,timezone
    import numpy as np
    import torch
    from scipy.io import loadmat
    from loadData.dataset_registry import inspect_dataset,DATASETS
    from loadData.data_read import readWavelengths
    from loadData.spatial_tiles import make_tile_plan,fit_tile_statistics,observe_tile
    from loadData.observation_sampling import BandMetadata,sample,replay,subset_statistics
    from model.variable_channel_adapter import VariableChannelAdapter
    from research_runtime import code_snapshot,external_directory,write_json,require_frozen_code
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit',required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args();code=code_snapshot()
    if not Path('/.dockerenv').exists() or sys.platform!='linux' or sys.flags.optimize or code['dirty'] or code['commit']!=args.expected_commit or not torch.cuda.is_available():
        p.error('Require training container, CUDA, assertions and clean exact SHA')
    out=external_directory(args.out_dir);out.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','commit':code['commit'],'branch':code['branch'],'source_hash':code['source_hash'],'command':[sys.executable,*sys.argv],'started_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(out/'manifest.json',manifest);torch.set_num_threads(1);results=[]
    try:
        for name in ('paviaU','xuzhou','whu-honghu'):
            identity,_=inspect_dataset(args.data_root,name);key=DATASETS[name][1]
            cube=loadmat(args.data_root/identity['image_path'],variable_names=[key])[key]
            plan=make_tile_plan(tuple(cube.shape[:2]),32,2);stats=fit_tile_statistics(cube,plan)
            obs=observe_tile(cube,plan,stats,next(t for t in plan['tiles'] if t['partition']=='train'))
            wave,_=readWavelengths(name,cube.shape[-1])
            # IDs refer to the delivered cube columns, not inferred hardware raw-band numbers.
            bands=BandMetadata(identity['image_sha256'],tuple(range(len(wave))),tuple(wave),'nm','approximate',(True,)*len(wave))
            low,high=float(min(wave)),float(max(wave));span=high-low
            cases=[{'mode':'none'}]+[{'mode':'random_subset','keep_fraction':f} for f in (.75,.5,.25,1e-9)]+[
                {'mode':'uniform_stride','stride':2,'offset':0},
                {'mode':'contiguous_gap','lower':low+span/3,'upper':low+2*span/3,'unit':'nm'},
                {'mode':'range_crop','lower':low+span/4,'upper':low+3*span/4,'unit':'nm'}]
            adapter=VariableChannelAdapter(latent_channels=8,embed_dim=8,locality_window=None).cuda().eval()
            full=torch.tensor(np.array(obs.spectral,copy=True),device='cuda')
            for i,config in enumerate(cases):
                view,record=sample(obs,bands,config,np.random.default_rng(2026+i))
                again=replay(obs,bands,json.loads(json.dumps(record)))
                np.testing.assert_array_equal(view.observation.spectral,again.observation.spectral)
                assert view.observation.coordinates is obs.coordinates and view.observation.global_ids is obs.global_ids
                subset_statistics(stats,bands,record)
                selected=record['selected_positions'];mask=torch.zeros(len(wave),device='cuda',dtype=torch.bool);mask[selected]=True
                with torch.no_grad():
                    actual=adapter(torch.tensor(np.array(view.observation.spectral,copy=True),device='cuda'),torch.tensor(view.bands.wavelengths,device='cuda',dtype=torch.float32),wavelength_unit='nm')
                    expected=adapter(full,torch.tensor(bands.wavelengths,device='cuda',dtype=torch.float32),wavelength_unit='nm',band_mask=mask)
                torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
                item={'dataset':name,'source_bands':len(wave),'visible_bands':len(selected),'record':record,'max_latent_difference':float((actual-expected).abs().max())}
                results.append(item);print(json.dumps({k:v for k,v in item.items() if k!='record'}),flush=True)
            assert inspect_dataset(args.data_root,name)[0]==identity
            del cube,full,adapter
        require_frozen_code(code);write_json(out/'subsets.json',results)
        manifest.update(status='passed',case_count=len(results),datasets=3,optimizer_steps=0,accuracy_evaluated=False)
    except BaseException:
        import traceback
        manifest.update(status='failed',error=traceback.format_exc());raise
    finally:
        manifest['finished_at_utc']=datetime.now(timezone.utc).isoformat();write_json(out/'manifest.json',manifest)

if __name__=='__main__':main()
