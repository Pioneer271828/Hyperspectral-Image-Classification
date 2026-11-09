"""Observed same-class components: sparse annotations are not complete objects.

Pure array diagnostics; callers must enforce their frozen label-access protocol.
No prediction is accepted when defining geometry, area or inward-depth groups.
"""
from collections import deque
from copy import deepcopy
import hashlib
import numpy as np
from model.research_config import digest
from .classification_metrics import class_order,class_indices,integer_values,classification_metrics


def _hash(array):
    array=np.ascontiguousarray(array)
    return digest(dict(shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest()))


def _inputs(labels,class_ids,geometry_mask,score_mask):
    labels=np.asarray(labels);ids=class_order(class_ids)
    if labels.ndim!=2 or not all(labels.shape):raise ValueError('Nonempty two-dimensional label map required')
    geometry,score=np.asarray(geometry_mask),np.asarray(score_mask)
    if any(m.shape!=labels.shape or m.dtype!=np.bool_ for m in (geometry,score)) or (score & ~geometry).any():
        raise ValueError('Boolean scoring access must be contained in geometry access')
    safe=np.zeros(labels.shape,dtype=np.int64);safe[geometry]=integer_values(labels[geometry],'Geometry labels')
    class_indices(safe[geometry & (safe>0)],ids,'Geometry labels')
    return safe,ids,geometry,score


def component_geometry(labels,class_ids,*,geometry_mask,score_mask,small_area,thin_depth):
    """Four-neighbor components and L1 inward layers (boundary pixels have 1).

    Unknown/excluded/out-of-array neighbors terminate observed geometry and mark
    incompleteness. Different *labeled* classes terminate a component normally.
    Thin means max inward depth <= threshold for the whole observed component.
    """
    safe,ids,geometry,score=_inputs(labels,class_ids,geometry_mask,score_mask)
    if any(type(v) is not int or v<1 for v in (small_area,thin_depth)):
        raise ValueError('Predeclared positive integer area/depth thresholds required')
    height,width=safe.shape;components=np.zeros_like(safe);depth=np.zeros_like(safe);rows=[];frontier=deque()
    for y in range(height):
        for x in range(width):
            if safe[y,x]==0 or components[y,x]:continue
            cid=len(rows)+1;label=int(safe[y,x]);queue=deque([(y,x)]);components[y,x]=cid
            area=scored=0;ys=[];xs=[];touch=dict(array_border=False,excluded=False,unlabeled=False)
            while queue:
                a,b=queue.popleft();area+=1;scored+=bool(score[a,b]);ys.append(a);xs.append(b);boundary=False
                for u,v in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
                    if not (0<=u<height and 0<=v<width):touch['array_border']=True;boundary=True;continue
                    if safe[u,v]!=label:
                        boundary=True
                        if not geometry[u,v]:touch['excluded']=True
                        elif safe[u,v]==0:touch['unlabeled']=True
                    elif not components[u,v]:components[u,v]=cid;queue.append((u,v))
                if boundary:depth[a,b]=1;frontier.append((a,b))
            rows.append(dict(component_id=cid,class_id=label,observed_area=area,scored_count=int(scored),
                box=[min(ys),max(ys)+1,min(xs),max(xs)+1],touches=touch,incomplete_observation=any(touch.values()),small=area<=small_area))
    while frontier:
        y,x=frontier.popleft()
        for u,v in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=u<height and 0<=v<width and components[u,v]==components[y,x] and not depth[u,v]:
                depth[u,v]=depth[y,x]+1;frontier.append((u,v))
    maximum=np.zeros(len(rows)+1,dtype=np.int64)
    np.maximum.at(maximum,components.ravel(),depth.ravel())
    for row in rows:
        d=int(maximum[row['component_id']]);row.update(max_inward_depth=d,thin=d<=thin_depth)
    value=dict(version=1,class_ids=ids,shape=list(safe.shape),geometry_mask_hash=_hash(geometry),score_mask_hash=_hash(score),
        geometry_labels_hash=_hash(safe),component_map_hash=_hash(components),depth_map_hash=_hash(depth),
        small_area=small_area,thin_depth=thin_depth,components=rows,geometry_labeled_count=int((safe>0).sum()),
        score_unlabeled_count=int((score & (safe==0)).sum()),geometry_labeled_outside_scoring=int(((safe>0)&~score).sum()),
        policy='observed_same_class_4_connected; depth=1+distance_to_component_boundary; whole_component_max_depth',
        limits=['annotation fragments are not complete physical objects','unknown or crop boundaries bias area and thickness',
                'thin whole-component definition excludes thick objects with thin appendages','groups overlap; pixels/components are not independent statistical repeats'])
    value['plan_hash']=digest(value)
    return value,components,depth


class ComponentPlan:
    def __init__(self,value,*,expected_hash):
        if not isinstance(value,dict) or value.get('plan_hash')!=expected_hash or digest({k:v for k,v in value.items() if k!='plan_hash'})!=expected_hash:
            raise ValueError('Frozen component geometry plan identity mismatch')
        self._value=deepcopy(value);self.plan_hash=expected_hash
    def to_dict(self):return deepcopy(self._value)
    @classmethod
    def freeze(cls,labels,class_ids,**kwargs):
        value,_,_=component_geometry(labels,class_ids,**kwargs)
        return cls(value,expected_hash=value['plan_hash'])


def score_components(labels,predictions,class_ids,*,geometry_mask,score_mask,plan,expected_plan_hash):
    if not isinstance(plan,ComponentPlan):raise ValueError('Frozen ComponentPlan required before predictions')
    value=ComponentPlan(plan.to_dict(),expected_hash=expected_plan_hash).to_dict()
    fresh,components,_=component_geometry(labels,class_ids,geometry_mask=geometry_mask,score_mask=score_mask,
        small_area=value['small_area'],thin_depth=value['thin_depth'])
    if fresh!=value:raise ValueError('Geometry/access/classes changed after freeze')
    safe,ids,_,score=_inputs(labels,class_ids,geometry_mask,score_mask)
    pred=np.asarray(predictions)
    if pred.shape!=safe.shape:raise ValueError('Prediction map shape differs')
    selected=score & (safe>0);class_indices(pred[selected],ids,'Scored predictions')
    saved=np.zeros_like(safe);saved[selected]=integer_values(pred[selected],'Scored predictions')
    table=[dict(small=False,thin=False,incomplete_observation=False)]+value['components']
    small=np.array([r['small'] for r in table])[components];thin=np.array([r['thin'] for r in table])[components]
    incomplete=np.array([r['incomplete_observation'] for r in table])[components]
    masks=dict(all=selected,small=selected&small,large=selected&~small,thin=selected&thin,thick=selected&~thin,
        small_thin=selected&small&thin,observed_closed=selected&~incomplete,incomplete=selected&incomplete)
    groups={name:classification_metrics(safe,saved,ids,mask) for name,mask in masks.items()}
    counts=np.bincount(components[selected],minlength=len(table));correct=np.bincount(components[selected & (saved==safe)],minlength=len(table))
    rows=[dict(component_id=r['component_id'],class_id=r['class_id'],scored_count=int(counts[r['component_id']]),
               correct=int(correct[r['component_id']]),recall=float(correct[r['component_id']]/counts[r['component_id']]) if counts[r['component_id']] else None)
          for r in value['components']]
    return dict(plan_hash=expected_plan_hash,predictions_hash=_hash(saved),groups=groups,components=rows,
                counts_are_pixels=True,statistical_inference=False,original_ground_truth_file_read=False,
                limitation='Pure array API does not authorize label release; outer frozen protocol required. Observed closed is not certified physical completeness.')
