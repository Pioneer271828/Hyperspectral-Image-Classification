"""Independent frozen checkpoint publication through the paired native path."""

from collections import Counter
from itertools import product
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from evaluation.frozen_training_stages import HEADS, file_pin, json_pin, require
from evaluation.wp12_native_production import endpoints, selected_model
from evaluation.wp12_mechanism_factorial import prior
from model.research_config import digest
from research_runtime import require_frozen_code, tensor_hash
from scripts.run_frozen_band_region import _write
from transfer import native_paired_head as head, native_paired_head_campaign as campaign, native_paired_head_fixture as fixture
from transfer.g3_training import state_hash
from transfer.native_scale_observation import forward
from transfer.spatial_coverage_protocol import bank_view

RAW="wp12_paired_head_independent_points_v1"
TRACE="wp12_retained_ordinal_source_trace_v1"
METADATA=("declared","missing","misaligned")
REPEATS=2


@torch.inference_mode()
def trial(provider,model,bank,roles,tile,band,metadata,checkpoint,execution):
    result=None
    elapsed=[]
    for _ in range(REPEATS+1):
        execution.sync()
        if execution.device.type=="cuda":
            torch.cuda.reset_peak_memory_stats(execution.device)
        start=perf_counter()
        actual,_,targets=provider.confirmation(checkpoint,tile,band,metadata,bank=bank,allowed=roles)
        prep=perf_counter()-start
        outputs,graph,timing=provider.registered_forward(model,actual,"devval",execution)
        projected=[actual.lattice.project(head).cpu().numpy() for head in outputs[:3]]
        execution.sync()
        total=perf_counter()-start
        observed=actual.receipt
        rows={head:matrix[targets.cells.tolist()].tolist() for head,matrix in zip(HEADS,projected,strict=True)}
        evidence=dict(truth=[y for _,y in targets.points],points=targets.summary()["point_set"],heads=rows,
            band_ids=list(actual.view.bands.band_ids),spectral=observed["spectral"],geometry=observed["original_boxes"],
            fit=observed["source"]["retained_statistics"],wavelength_quality=actual.view.bands.wavelength_quality,
            mode=observed["source"]["spectral_metadata_mode"],order=actual.view.bands.fingerprint,
            graph=graph,shape=observed["grid_shape"])
        if result is None:
            result=evidence
        else:
            require(result==evidence,"Repeated frozen paired prediction or native graph differs")
            elapsed.append(dict(total=total,prep=prep,forward=timing["forward_seconds"],graph=sum(timing["graph_seconds"]),
                                peak_cuda_bytes=torch.cuda.max_memory_allocated(execution.device)
                                                if execution.device.type=="cuda" else None))
    return result|dict(times=elapsed)


def ledger(state,spec):
    initial=spec["initialization"]["inventory"]
    checkpoints={"selected":state["selected"]["state"],"last":state["model"]}
    counts={}
    for phase,value in checkpoints.items():
        changed=sorted(k for k,v in value.items() if tensor_hash(v)!=initial[k])
        require(set(changed).issubset(spec["trainable"]),"Non-head model tensor changed")
        if phase=="last":
            require(set(changed)==set(spec["trainable"]),"No effective trainable head update recorded")
        groups=Counter(k.split(".")[1] if k.startswith("backbone.") else k.split(".")[0] for k in value)
        counts[phase]=dict(changed=changed,frozen_groups=dict(groups),unchanged=len(value)-len(changed))
    return counts


def capture(root,program,folder,name,producer,code,execution):
    plan,link=campaign.load(root,program,producer)
    scope=plan["cases"][name]
    folder.mkdir(parents=True,exist_ok=True)
    where=folder/(name+".json")
    if where.exists():
        previous=json_pin(root,file_pin(where))
        require(previous["producer"]==producer and previous["consumer"]==code["commit"]
                and previous["device"]==execution.numeric,"Completed paired confirmation differs")
        return file_pin(where)
    data={}
    for arm in scope["policies"]:
        p,proof,receipt,spec,state,model=campaign.restored(root,program,plan,link,name,arm,execution)
        original=state_hash(model)
        banks={scope["kind"]:(p.bank,proof)}
        if scope["kind"]=="fallback":
            banks.update({kind:campaign.bank_for(program,plan,kind,scope["support"]) for kind in fixture.FINAL})
        rows=[]
        checkpoint=receipt["phases"]["full"]["checkpoint"]["sha256"]
        for label,(bank,meta) in banks.items():
            roles=meta["roles"]["evaluation"]
            require(not set(roles) & set(meta["roles"]["support"]+meta["roles"]["calibration"]),
                    "Held geometry crosses into training or selection")
            for tile,band,metadata in product(roles,fixture.VIEWS,METADATA if scope["kind"]=="fallback" else ("declared",)):
                evidence=trial(p,model,bank,roles,tile,band,metadata,checkpoint,execution)
                rows.append(dict(source=label,tile=tile,band=band,metadata=metadata,
                         layout=meta["blocks"][str(tile)]["layout"],mechanism=meta["blocks"][str(tile)]["mechanism"],
                         **evidence))
        require(state_hash(model)==original,"Independent confirmation mutated selected weights")
        exposure=Counter()
        effective_labels=np.zeros(spec["classes"],dtype=np.int64)
        for row in state["history"]:
            exposure[f'{row["band"]}/{row["metadata"]}']+=1
            effective_labels+=np.asarray(row["released"]["per_class"],dtype=np.int64)
        data[arm]=dict(checkpoint=checkpoint,state=original,selected=receipt["selected"],
            trainable=spec["trainable_parameters"],dimension=spec["recognition_dim"],class_axis=spec["classes_1_based"],
            source=spec["initialization"]["source"],initial=spec["initialization"]["state"],
            training_order=digest([r["tile"] for r in spec["training"]]),labels_seen=effective_labels.tolist(),
            exposure=dict(exposure),role=spec["roles"],updates=receipt["expected_updates"],layers=ledger(state,spec),
            graph_and_cnn_preserved=True,train_device=receipt["device"],recovery_exact=receipt["exact_reference"],
            training_times=receipt["phases"]["full"]["timing"],rows=rows,
            calibration=[dict(step=row["step"],nll=row["fused_nll"]) for row in state["validation"]])
    require_frozen_code(code)
    return _write(where,dict(schema=RAW,scope="independent_synthetic_confirmation",case=name,
                        producer=producer,consumer=code["commit"],registration=link["sha256"],device=execution.numeric,
                        arms=data,query_accessed=False,raw_gt_loaded=False,real_updates=0,independent_real_accuracy=None))


@torch.inference_mode()
def trace(root,execution):
    folder,plan,plan_pin,_=prior(root)
    name="s64-k211-native"
    p,receipt,_,state=endpoints(root,folder,plan,plan_pin,plan["producer"],name)
    model=selected_model(p,receipt,state,execution.device)
    adapter=model.backbone.adapter
    base=state_hash(model)
    tile=p.roles["evaluation"][0]
    declared_original,_,_=bank_view(p.bank,p.bank.role_row("devval",tile),p.producer,"central_gap")
    options={}
    for mode in ("declared","missing","ordinal","permuted_ordinal","misaligned"):
        actual,_,targets=bank_view(p.bank,p.bank.role_row("devval",tile),p.producer,"central_gap",metadata=mode)
        require(actual.receipt["spectral"]==declared_original.receipt["spectral"]
                and actual.receipt["source"]["retained_statistics"]==declared_original.receipt["source"]["retained_statistics"]
                and actual.view.bands.band_ids==declared_original.view.bands.band_ids,"Fallback changed actually retained columns or stats")
        invocations=[]
        original=adapter._adapt_flat

        def verify(flat_x,wavelengths=None,fwhm=None,return_attention=False,wavelength_unit="auto",**kwargs):
            w=adapter._to_1d_tensor(wavelengths,flat_x.shape[1],flat_x.device,flat_x.dtype,"wavelengths")
            w=adapter._normalize_wavelengths(w,wavelength_unit)
            value=original(flat_x,wavelengths=wavelengths,fwhm=fwhm,return_attention=return_attention,
                           wavelength_unit=wavelength_unit,**kwargs)
            positions=w.tolist()
            invocations.append(dict(positions=positions,width=flat_x.shape[1],wavelength_was_none=wavelengths is None,
                                    fwhm_was_none=fwhm is None,argument_unit=wavelength_unit,
                                    input_hash=tensor_hash(flat_x),latent_hash=tensor_hash(value[0] if isinstance(value,tuple) else value)))
            return value

        with patch.object(adapter,"_adapt_flat",new=verify):
            output,graph=forward(model,actual,p.producer,"devval",execution=execution)
        require(len(invocations)==1,"The normal VCA path did not consume the expected axis input")
        if mode=="declared":
            require(actual.receipt==declared_original.receipt,"Declared metadata changed legacy receipt identity")
            again,old_graph=forward(model,declared_original,p.producer,"devval",execution=execution)
            require(graph==old_graph and all(torch.equal(a,b) for a,b in zip(output[:3],again[:3],strict=True)),
                    "Existing declared checkpoints differ when ordinal support is installed")
        if mode in ("ordinal","permuted_ordinal"):
            require(actual.view.bands.wavelength_unit=="normalized" and actual.view.bands.wavelength_quality=="index_only"
                    and actual.view.bands.fwhm is None and not invocations[0]["wavelength_was_none"],
                    "Ordinal input was promoted to physical calibration")
        options[mode]=dict(source=actual.receipt["source"]["spectral_metadata_mode"],band_ids=list(actual.view.bands.band_ids),
             positions_before_mask=actual.receipt["source"]["positions"],axis=invocations[0],geometry=graph,
             head_hashes={name:tensor_hash(val) for name,val in zip(HEADS,output[:3],strict=True)},
             points=len(targets.points))
    require(len({v["axis"]["input_hash"] for v in options.values()})==1
            and options["missing"]["axis"]["positions"]!=options["ordinal"]["axis"]["positions"]
            and state_hash(model)==base,"Source trace did not isolate coordinate encoding")
    return dict(schema=TRACE,scope="frozen_synthetic_source_trace",case=name,checkpoint=receipt["full"]["sha256"],
                device=execution.numeric,metadata_paths=options,declared_backward_compatible=True,
                real_updates=0,raw_gt_loaded=False,query_accessed=False)
