"""CPU whole-tile cascade; conservative default and genuinely lazy full calls."""
import math
import torch
from inference.light_cnn import LightCNN,validate_source
from transfer.g3_training import state_hash


def cascade(full,light,requests,*,mode='full',criterion=None,threshold=None,min_coverage=1.0):
    """Thresholds here are explicit engineering inputs, not calibrated certificates.

    No labels, thresholds search, pixel deletion or cache shared between models.
    Independent light work and full recomputation both count when falling back.
    """
    validate_source(full)
    if (type(light) is not LightCNN or any(m.training for m in light.modules()) or light.architecture!=full.export_config()
            or light.source_state!=state_hash(full)):raise ValueError('Frozen light/full source identity mismatch')
    source=full.state_dict()
    prefixes={'adapter':'backbone.adapter','denoise':'backbone.CNN_denoise','cnn':'backbone.CNN_Branch','heads':'cnn_heads'}
    for key,value in light.state_dict().items():
        prefix,suffix=key.split('.',1)
        if not torch.equal(value,source[prefixes[prefix]+'.'+suffix]):raise ValueError('Light weights differ from frozen source')
    for v in light.state_dict().values():
        if v.device.type!='cpu' or not torch.isfinite(v).all() or (v.is_floating_point() and v.dtype!=torch.float32):raise ValueError('CPU FP32 light model required')
    if mode not in ('full','fast','adaptive'):raise ValueError('Unknown cascade mode')
    if type(min_coverage) not in (int,float) or not math.isfinite(min_coverage) or not 0<=min_coverage<=1:raise ValueError('Explicit coverage floor required')
    if mode=='adaptive':
        if criterion not in ('max_probability','margin','entropy') or type(threshold) not in (int,float) or not math.isfinite(threshold):raise ValueError('Explicit gate required')
        if threshold<0 or (criterion!='entropy' and threshold>1):raise ValueError('Invalid threshold')
    elif criterion is not None or threshold is not None:raise ValueError('Endpoint modes take no gate')
    if not isinstance(requests,(list,tuple)):raise ValueError('Materialized requests required')
    keys=set()
    for row in requests:
        if set(row)!={'id','dataset','x','wavelengths','scale','coverage_known','coverage_fraction'}:raise ValueError('Exact request fields required')
        if not isinstance(row['id'],str) or not row['id'] or row['id'] in keys:raise ValueError('Unique tile IDs required')
        keys.add(row['id']);x=row['x'];wave=row['wavelengths']
        if row['dataset'] not in light.heads or type(row['scale']) is not int or row['scale']<1:raise ValueError('Invalid task/graph scale')
        if mode=='adaptive' and criterion=='margin' and light.heads[row['dataset']].out_features<2:raise ValueError('Margin requires at least two classes')
        if type(row['coverage_known']) is not bool or type(row['coverage_fraction']) not in (int,float) or not math.isfinite(row['coverage_fraction']) or not 0<=row['coverage_fraction']<=1:raise ValueError('Invalid coverage')
        if (not torch.is_tensor(x) or x.device.type!='cpu' or x.dtype!=torch.float32 or x.requires_grad or x.ndim!=3
                or min(x.shape[:2])<1 or not torch.isfinite(x).all()):raise ValueError('Invalid tile tensor')
        if (not torch.is_tensor(wave) or wave.device.type!='cpu' or wave.dtype!=torch.float32 or wave.requires_grad or wave.ndim!=1
                or len(wave)<1 or not torch.isfinite(wave).all() or (wave<=0).any() or x.shape[-1]!=len(wave)+light.extra_channels):raise ValueError('Invalid tile wavelengths')
    result=[];counts=dict(light_calls=0,full_calls=0,accepted_tiles=0,total_tiles=len(requests))
    with torch.no_grad():
        for row in requests:
            accepted=False;score=None;known=row['coverage_known'] and row['coverage_fraction']>=min_coverage
            if mode!='full' and known:
                p=light(row['dataset'],row['x'],wavelengths=row['wavelengths']);counts['light_calls']+=1
                if mode=='fast':accepted=True
                else:
                    if criterion=='max_probability':score=float(p.max(1).values.min());accepted=score>=threshold
                    elif criterion=='margin':
                        top=p.topk(2,dim=1).values;score=float((top[:,0]-top[:,1]).min());accepted=score>=threshold
                    else:
                        logs=torch.zeros_like(p);nz=p>0;logs[nz]=p[nz].log();score=float((-(p*logs).sum(1)).max());accepted=score<=threshold
            if not accepted:
                p=full(row['dataset'],row['x'],wavelengths=row['wavelengths'],superpixel_scale=row['scale'])[0];counts['full_calls']+=1
            else:counts['accepted_tiles']+=1
            if not torch.isfinite(p).all():raise ValueError('Nonfinite cascade output')
            result.append(dict(id=row['id'],probabilities=p,route='fast' if accepted else 'full',score=score))
    return dict(outputs=result,counts=counts,mode=mode,calibrated=False,
                cost_scope='Independent light plus full recomputation on fallback; end-to-end measurement required.')

