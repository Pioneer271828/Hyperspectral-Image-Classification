"""Separate executable numeric development intake from shared-semantic claims."""
from inspect_bank import inspect


FAMILIES={'xuzhou':'HySpex VNIR-1600 + SWIR-384, premerged product',
          'whu-honghu':'Headwall Nano-Hyperspec'}


def pair_plan(source,target,*,mode='numeric_new_head',heldout_scenes=()):
    """Plan from authenticated original metadata, before opening any array.

    numeric_new_head preserves each scene's own labels; it is not a common
    taxonomy or a zero-shot classifier. This function grants no training budget.
    """
    if source.scene==target.scene:raise ValueError('Two independently acquired scenes required')
    if mode!='numeric_new_head':raise ValueError('Shared-head admission blocked: exact Xuzhou ordinal/product rights remain unresolved')
    if heldout_scenes:raise ValueError('No untouched third-acquisition/confirmation role has been released by C6')
    audits={b.scene:inspect(b) for b in (source,target)}
    return {'source':source.scene,'target':target.scene,'mode':mode,
            'device_families':{b.scene:FAMILIES[b.scene] for b in (source,target)},
            'source_fitting_role':'devtrain','source_selection_role':'devval',
            'target_adaptation_role':'devtrain','target_development_role':'devval',
            'source_native_classes':source.pin['classes'],'target_native_classes':target.pin['classes'],
            'class_heads_shared':False,'target_head_must_be_fresh':True,
            'original_role_audits':audits,'heldout_confirmation_admitted':False,
            'claim_scope':'Cross-scene/device few-shot classification development input readiness only. Device, scene, date, geography, resolution and class composition are confounded.',
            'causal_sensor_effect_identified':False,'native_SRF_authenticated':False,
            'training_authorized_updates':0,'real_neural_calls_authorized':0}


def require_role(bank,role,tile_id,*,purpose):
    roles={'source_fit':'devtrain','source_select':'devval',
           'target_adapt':'devtrain','target_development':'devval'}
    if purpose not in roles or roles[purpose]!=role:raise ValueError('Purpose cannot promote a development or held-out role')
    if tile_id not in bank.rows or bank.rows[tile_id]['role']!=role or not bank.samples[tile_id]:raise ValueError('Originally labeled tile/role required')
    return {'scene':bank.scene,'original_role':role,'purpose':purpose,'tile_id':tile_id}
