"""Independently check plotted values against restored aggregate counts, never labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pymupdf


def require(ok,reason):
    if not ok:raise ValueError(reason)


def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(Path(p).open()))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    frozen=a.run/'frozen';out=a.output
    original=read(frozen/'s2-results.json');expected={};supports={}
    require(original['optimizer_updates']==0,'Archived readout must be zero-update')
    for row in original['rows']:
        if row['precision']!='fp32':continue
        cm=np.asarray(row['scores']['fused']['confusion'],dtype=np.int64)
        require(cm.ndim==2 and cm.shape[0]==cm.shape[1] and np.all(cm>=0),'Invalid archived confusion')
        value=100*float(np.trace(cm)/cm.sum());key=(row['dataset'],row['mode'],row['supervision'],int(row['seed']))
        require(np.isclose(value,100*row['scores']['fused']['oa'],atol=1e-12,rtol=0),'Archived count/OA arithmetic differs')
        expected[key]=value;supports.setdefault(row['dataset'],set()).add(tuple(cm.sum(1)))
    for residual in ('none','x_residual'):
        for supervision in ('legacy_three_heads','primary_only'):
            for seed in (64,512,2048):
                expected[('macro',residual,supervision,seed)]=sum(expected[(scene,residual,supervision,seed)] for scene in supports)/3
    actual=rows(out/'data/S2-four-cells.csv');require(len(actual)==len(expected)==48,'Four-cell plot roster differs')
    for row in actual:
        key=(row['scene'],row['residual'],row['supervision'],int(row['seed']))
        require(abs(float(row['oa_percent'])-expected[key])<1e-11,'Figure R1 plotted value differs from archived counts')
    require(all(len(v)==1 for v in supports.values()),'Fixed per-scene label denominators changed across cells/seeds')
    cases=read(frozen/'wp04-cases.json')['cases'];correct={};total={}
    for key,row in cases.items():
        cm=np.asarray(row['metrics']['fused']['confusion']);require(cm.shape==(9,9) and np.array_equal(cm.sum(1),[5]*9),'WP04 fixed K1/V5 denominator differs')
        correct[key]=np.diag(cm).astype(float);total[key]=float(np.trace(cm))
    class_expected={}
    for seed in (64,512,2048):
        c=[correct[f'{seed}-{arm}'] for arm in ('M00','M10','M01','M11')]
        effects={'message':(c[1]+c[3]-c[0]-c[2])/2,'joint_edge':(c[2]+c[3]-c[0]-c[1])/2,
                 'interaction':c[3]-c[1]-c[2]+c[0]}
        for name,values in effects.items():
            for i,value in enumerate(values,1):class_expected[(name,seed,i)]=float(value)*20
    class_actual=rows(out/'data/WP04-class-protection.csv');require(len(class_actual)==len(class_expected)==81,'Selected fused class roster differs')
    for r in class_actual:
        require(int(r['points_per_class'])==5 and abs(float(r['delta_recall_pp'])-class_expected[(r['effect'],int(r['seed']),int(r['class_id']))])<1e-11,'Figure R2 differs from backed-up original four-cell counts')
    adoption={}
    for name in ('message','joint_edge'):
        means=[sum(class_expected[(name,s,c)] for s in (64,512,2048))/3 for c in range(1,10)]
        oa=[sum(class_expected[(name,s,c)] for c in range(1,10))/9 for s in (64,512,2048)]
        failed=[i for i,v in enumerate(means,1) if v < -2-1e-10]
        require(failed and sum(v>0 for v in oa)==2,'Original positive-seed/class-protection interpretation differs')
        adoption[name]={'mean_OA_change_pp':sum(oa)/3,'positive_seeds':sum(v>0 for v in oa),'failed_classes':failed,'adopted':False}
    role_rows=rows(out/'data/current-support-units.csv');require(len(role_rows)==4,'Role table roster differs')
    require(sum(int(r['released_points']) for r in role_rows)==40 and sum(int(r['C7_selected_blocks']) for r in role_rows)==17,'C7 unique support totals differ')
    require(all(int(r['released_points'])==10 and int(r['per_common_class'])==5 and int(r['original_acquisition_images'])==1
                and int(r['untouched_confirmation_units_released'])==0 and not r['validated_independent_parcels'] for r in role_rows),'Point/image/parcel/confirmation semantics differ')
    hh=rows(frozen/'honghu-class-parent-evidence.csv')
    for r in hh:
        role=next(x for x in role_rows if x['scene']=='whu-honghu' and x['original_role']==r['role'])
        require(int(r['points'])==5 and json.loads(role['per_class_parent_count'])[r['common_class']]==int(r['original_parents']),'HongHu per-class parent denominators differ')
    plan=read(out/'data/prospective-status.json');hyp=read(frozen/'C10-hypotheses.json')
    require(plan['performance_values'] is None and plan['actual_updates']==0 and plan['proposed_fits']==9
            and len(plan['evaluation_cells'])==9 and plan['proposed_source_steps']==64
            and plan['new_Hyperion_points_required_if_feasible']==40 and not hyp['current_C7_ontology_reused']
            and hyp['authorized_real_updates']==0 and hyp['authorized_new_Hyperion_points']==0,'Prospective plan was presented as a result')
    pdfs={}
    for path in sorted((out/'figures').glob('*.pdf')):
        width=int(path.stem.split('-')[-1].removesuffix('mm'));doc=pymupdf.open(path);page=doc[0];text=page.get_text()
        require(len(doc)==1 and abs(page.rect.width*25.4/72-width)<.02 and not page.get_images(),'Fixed-width/vector PDF audit failed')
        require(page.get_fonts() and all(doc.extract_font(f[0])[3] for f in page.get_fonts()),'Font not embedded')
        if path.name.startswith('D1'):require('No performance values available' in text and 'not run' in text,'Admission diagram lost unmeasured qualification')
        if path.name.startswith('D2'):require('confirmation units: 0' in text and 'not four acquisitions' in text,'Support table lost denominator qualification')
        pdfs[path.name]={'width_mm':width,'embedded_font_count':len(page.get_fonts()),'raster_image_objects':0}
    require(len(pdfs)==8,'Print-width variants missing')
    receipt={'status':'PASSED','R1_OA_values_checked':48,'R2_selected_fused_class_effects_checked':81,
             'S2_fixed_evaluation_points':{k:sum(next(iter(v))) for k,v in supports.items()},'S2_model_seeds':[64,512,2048],
             'WP04_unique_development_points':45,'WP04_points_per_class':5,'WP04_adoption':adoption,
             'C7_unique_points':40,'C7_original_images':{'xuzhou':1,'whu-honghu':1},'C7_confirmatory_units_released':0,
             'cross_figure_unique_point_total':None,'cross_figure_point_warning':'S2, WP04 and C7 are different releases. Their overlap is not resolved by these aggregates; do not sum their point counts.',
             'same_seed_warning':'The three numeric model seeds are reused across protocols; not nine independent seeds or devices.',
             'prospective_cells':9,'prospective_actual_updates':0,'prospective_performance_values':None,
             'PDF_audit':pdfs,'plotted_data_sha256':{p.name:sha(p) for p in (out/'data').iterdir()},
             'independent_of_plot_derivation':True,'new_predictions':0,'new_optimizer_updates':0,'query_arrays_read':0}
    with a.out.open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True)
    print(json.dumps({k:v for k,v in receipt.items() if k in ('status','R1_OA_values_checked','R2_selected_fused_class_effects_checked','WP04_adoption')}))


if __name__=='__main__':main()
