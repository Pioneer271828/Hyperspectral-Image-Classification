"""Draw only missing paper figures from frozen deidentified aggregates."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
import numpy as np
import pymupdf

from freeze import SEEDS, clean, csvread, read, require, sha, write

COLORS={'off':'#687681','on':'#007A87','ink':'#17324D','red':'#B24442','grey':'#697681'}
MARKERS=('o','s','^')
matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,
    'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,
    'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'C12-publication-v1','axes.spines.top':False,
    'axes.spines.right':False,'axes.edgecolor':'#87939B','text.color':COLORS['ink'],
    'axes.labelcolor':COLORS['ink'],'xtick.color':COLORS['ink'],'ytick.color':COLORS['ink']})


def csvwrite(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def derive(frozen,out):
    summary=read(frozen/'s2-summary.json');s2=[]
    for condition,v in summary['tables']['fp32'].items():
        mode,supervision=condition.split('-',1)
        groups={'macro':v['macro_oa'],**{k:d['oa'] for k,d in v['datasets'].items()}}
        for scene,s in groups.items():
            for seed,value in zip(SEEDS,s['values']):
                s2.append({'scene':scene,'residual':mode,'supervision':supervision,'seed':seed,'oa_percent':value})
    classes=[]
    for r in csvread(frozen/'a-matched-classes.snapshot.csv'):
        if r['endpoint']=='selected' and r['head']=='fused':
            classes.append({'effect':r['effect'],'seed':int(r['seed']),'class_id':int(r['class_id']),
                            'points_per_class':int(r['denominator']),'delta_recall_pp':100*float(r['delta_correct'])/int(r['denominator'])})
    roles=csvread(frozen/'spatial-unit-evidence.csv');hyp=read(frozen/'C10-hypotheses.json')
    csvwrite(out/'data/S2-four-cells.csv',s2);csvwrite(out/'data/WP04-class-protection.csv',classes)
    csvwrite(out/'data/current-support-units.csv',roles)
    write(out/'data/prospective-status.json',{'status':hyp['status'],'task_id':hyp['task_id'],
          'actual_updates':hyp['budget_if_separately_authorized']['actual_updates_now'],
          'proposed_fits':hyp['budget_if_separately_authorized']['source_linear_fits'],
          'proposed_source_steps':hyp['fitting']['steps'],'evaluation_cells':hyp['evaluation_cells'],
          'new_Hyperion_points_required_if_feasible':hyp['budget_if_separately_authorized']['new_Hyperion_role_points_required_if_feasible'],
          'performance_values':None})
    return s2,classes,roles,hyp


def r1(data,width):
    narrow=width==85
    fig,axes=plt.subplots(4 if narrow else 2,1 if narrow else 2,figsize=(width/25.4,(170 if narrow else 124)/25.4))
    fig.subplots_adjust(left=.24 if narrow else .12,right=.97,bottom=.09 if narrow else .13,top=.90 if narrow else .87,
                        hspace=.49 if narrow else .57,wspace=.35)
    for ax,scene,title in zip(np.asarray(axes).ravel(),('macro','paviaU','xuzhou','whu-honghu'),('Macro OA','PaviaU','Xuzhou','HongHu')):
        vals=[]
        for mode,color,ls,offset in [('none',COLORS['off'],'--',-.055),('x_residual',COLORS['on'],'-',.055)]:
            series=[]
            for i,seed in enumerate(SEEDS):
                y=[next(r['oa_percent'] for r in data if (r['scene'],r['residual'],r['supervision'],r['seed'])==(scene,mode,s,seed)) for s in ('legacy_three_heads','primary_only')]
                x=np.arange(2)+offset+(i-1)*.018
                ax.plot(x,y,color=color,ls=ls,lw=.7,alpha=.65,marker=MARKERS[i],ms=3.7,mfc='white',mew=.85)
                vals+=y;series.append(y)
            ax.plot(np.arange(2)+offset,np.mean(series,axis=0),color=color,ls=ls,lw=1.8,zorder=4)
        span=max(vals)-min(vals);ax.set_ylim(min(vals)-max(.13,span*.15),max(vals)+max(.13,span*.15))
        ax.set_xlim(-.2,1.2);ax.set_xticks([0,1],['Three-head','Fused-only']);ax.set_ylabel('Fused OA (%)');ax.set_title(title,loc='left',pad=6)
        ax.grid(axis='y',color='#E7EBEE',lw=.5);ax.set_axisbelow(True)
    handles=[Line2D([],[],color=COLORS['off'],ls='--',label='Residual off'),Line2D([],[],color=COLORS['on'],label='Residual on')]
    handles += [Line2D([],[],color='#56616A',lw=0,marker=m,mfc='white',ms=4,label=str(s)) for s,m in zip(SEEDS,MARKERS)]
    if narrow:
        fig.legend(handles=handles[:2],loc='upper center',bbox_to_anchor=(.5,1),ncol=2,frameon=False,fontsize=7.7)
        fig.legend(handles=handles[2:],loc='upper center',bbox_to_anchor=(.5,.965),ncol=3,frameon=False,fontsize=7.7)
    else:fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1),ncol=5,frameon=False,fontsize=7.7,columnspacing=.9,handlelength=1.6)
    fig.text(.5,.018,'S2: 12 models; 3 paired seeds; fixed scene splits.\nNUM-2 amended readout; y-axes are cropped.',ha='center',va='bottom',fontsize=7.5)
    return fig


def r2(data,width):
    narrow=width==85
    fig,axes=plt.subplots(2 if narrow else 1,1 if narrow else 2,figsize=(width/25.4,(153 if narrow else 95)/25.4))
    fig.subplots_adjust(left=.18 if narrow else .10,right=.97,top=.84 if narrow else .76,bottom=.17 if narrow else .22,hspace=.66,wspace=.30)
    for ax,effect,title in zip(np.asarray(axes).ravel(),('message','joint_edge'),('Message main effect','Joint-edge main effect')):
        rows=[r for r in data if r['effect']==effect];means=[]
        for i,seed in enumerate(SEEDS):
            y=[next(r['delta_recall_pp'] for r in rows if (r['seed'],r['class_id'])==(seed,c)) for c in range(1,10)]
            ax.scatter(np.arange(1,10)+(i-1)*.16,y,marker=MARKERS[i],s=16,facecolors='none',edgecolors='#6D7E88',lw=.8,zorder=3)
        for c in range(1,10):
            value=float(np.mean([r['delta_recall_pp'] for r in rows if r['class_id']==c]));means.append(value)
            ax.scatter(c,value,marker='D',s=22,color=COLORS['red'] if value < -2-1e-10 else COLORS['ink'],zorder=4)
        ys=[r['delta_recall_pp'] for r in rows];ax.set_ylim(min(-15,min(ys)-6),max(25,max(ys)+6))
        ax.axhspan(ax.get_ylim()[0],-2,color=COLORS['red'],alpha=.06);ax.axhline(-2,color=COLORS['red'],ls='--',lw=.9)
        ax.axhline(0,color='#AAB4BB',lw=.6);ax.set_xticks(range(1,10));ax.set_xlim(.5,9.5)
        ax.set_xlabel('Original numeric class ID');ax.set_ylabel('Recall change (pp)');ax.set_title(title,loc='left',pad=26)
        failed=', '.join(str(c) for c,v in enumerate(means,1) if v < -2-1e-10)
        ax.text(0,1.018,'Not admitted: class '+failed,transform=ax.transAxes,va='bottom',fontsize=7.3,color=COLORS['red'])
    handles=[Line2D([],[],color='#6D7E88',lw=0,marker=m,mfc='white',ms=4,label=str(s)) for s,m in zip(SEEDS,MARKERS)]
    handles += [Line2D([],[],color=COLORS['ink'],lw=0,marker='D',ms=4,label='Seed mean')]
    fig.legend(handles=handles,loc='upper center',ncol=4,frameon=False,fontsize=7.5,columnspacing=.8,handletextpad=.3)
    fig.text(.5,.025,'WP04: same 45 development points; 5 per class.\nK1/V5, 12 steps. Separate protocol from S2.\nDashed line: preregistered -2 pp class floor.',ha='center',va='bottom',fontsize=7.4)
    return fig


def box(ax,x,y,w,h,text,*,prospective=False):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',
         ec=COLORS['grey'] if prospective else COLORS['on'],fc='#F6F7F8' if prospective else '#EFF7F7',
         lw=.9,ls=(0,(4,2)) if prospective else '-')
    ax.add_patch(patch);t=ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8.2,linespacing=1.25)
    ax._c12_boxes=getattr(ax,'_c12_boxes',[])+[(patch,t)]


def d1(width,hyp):
    narrow=width==85;fig,ax=plt.subplots(figsize=(width/25.4,(185 if narrow else 126)/25.4));fig.subplots_adjust(left=.02,right=.98,bottom=.03,top=.99)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    current=['Reported device models\nXuzhou: HySpex\nHongHu: Nano-Hyperspec\nSerial/SRF calibration unverified',
             'Frozen ground/tree development roles\nK5 fit + V5 development per class\n40 unique points across both scenes\nBoth directions reuse these points',
             'Input admission established\nNo coarse-pair classifier outcomes\nNo untouched confirmation role\nSpatial independence not established']
    future=['HyRANK Dioni + Loukia\nSame Hyperion instrument family\nProspective water/woody task\nDifferent ontology from ground/tree',
            'Admission blockers remain\nOriginal delivered class codes\nPreprocessing and spatial role units\nNo new imagery or labels acquired',
            f"Conditional plan: {hyp['budget_if_separately_authorized']['source_linear_fits']} source fits x {hyp['fitting']['steps']} steps\n9 source-target/local cells\nActual fits and new role points: 0\nNo performance values available"]
    if narrow:
        groups=[(.07,.86,[.79,.645,.50],.105,current,False),(.07,.86,[.315,.18,.045],.105,future,True)]
        ax.text(.5,.955,'Established C7/C9 support',ha='center',fontsize=10,weight='bold')
        ax.text(.5,.445,'Prospective C10 - not run',ha='center',fontsize=10,weight='bold')
    else:
        groups=[(.045,.42,[.715,.435,.155],.205,current,False),(.535,.42,[.715,.435,.155],.205,future,True)]
        ax.text(.255,.963,'Established C7/C9 support',ha='center',fontsize=10,weight='bold')
        ax.text(.745,.963,'Prospective C10 - not run',ha='center',fontsize=10,weight='bold')
    for x,w,ys,h,texts,planned in groups:
        for y,text in zip(ys,texts):box(ax,x,y,w,h,text,prospective=planned)
        for high,low in zip(ys,ys[1:]):ax.annotate('',xy=(x+w/2,low+h+.012),xytext=(x+w/2,high-.012),arrowprops={'arrowstyle':'-|>','color':COLORS['grey'] if planned else COLORS['on'],'lw':.9})
    if not narrow:ax.text(.5,.035,'Solid: established input facts. Dashed: unadmitted plan. Neither panel claims new transfer accuracy.',ha='center',fontsize=7.4)
    return fig


def d2(roles,width):
    narrow=width==85;fig,ax=plt.subplots(figsize=(width/25.4,(145 if narrow else 92)/25.4));fig.subplots_adjust(left=.045,right=.955,bottom=.02,top=.98);ax.axis('off')
    rows=[]
    for r in roles:
        scene='Xuzhou' if r['scene']=='xuzhou' else 'HongHu';role='fit' if r['original_role']=='devtrain' else 'dev'
        parents=r['selected_parent_count_reported'];pc='Not verified' if r['scene']=='xuzhou' else ('2 / 3' if role=='fit' else '1 / 1')
        rows.append([scene+' '+role,'5 / 5',parents] if narrow else [scene+' '+role,'5 / 5',parents,pc,'1*'])
    labels=['Scene / role','Points G / T','Parents'] if narrow else ['Scene / role','Ground / tree\npoints','Selected\nparents','Parents per class\n(ground / tree)','Original\nimage*']
    table=ax.table(cellText=rows,colLabels=labels,cellLoc='center',loc='upper center',
                   colWidths=[.46,.29,.25] if narrow else [.24,.18,.15,.27,.16],bbox=[0,.57 if narrow else .42,1,.34 if narrow else .42])
    table.auto_set_font_size(False);table.set_fontsize(8.2)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#D6DEE3');cell.set_linewidth(.5)
        if row==0:cell.set_facecolor('#E7F0F3');cell.set_text_props(weight='bold')
        elif row%2==0:cell.set_facecolor('#F5F7F8')
    ax.text(.0,.97,'Established development support',ha='left',va='top',fontsize=10,weight='bold')
    note=('40 unique points total; two directions reuse them (80 presentations).\n'
          '* One original image per scene, reused across roles and analyses; not four acquisitions.\n'
          'HongHu parents by class: fit 2 / 3, dev 1 / 1. Xuzhou per-class parent counts unverified.\n'
          'Parent tiles are not validated independent parcels. Untouched confirmation units: 0.\n'
          'C10 Hyperion role points acquired: 0; 40 new points are conditional requirements, not released support.')
    note='\n'.join(textwrap.fill(line,43 if narrow else 107) for line in note.splitlines())
    ax.text(0,.515 if narrow else .345,note,ha='left',va='top',fontsize=8.0,linespacing=1.45)
    return fig


def save(fig,path,width):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
    clips=[]
    for t in fig.findobj(match=Text):
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        if b.x0<bounds.x0-1 or b.y0<bounds.y0-1 or b.x1>bounds.x1+1 or b.y1>bounds.y1+1:clips.append(t.get_text())
    for ax in fig.axes:
        for patch,t in getattr(ax,'_c12_boxes',[]):
            require(patch.get_window_extent(renderer).contains(*t.get_window_extent(renderer).get_points()[0])
                    and patch.get_window_extent(renderer).contains(*t.get_window_extent(renderer).get_points()[1]),'Flow text exceeds its box')
    require(not clips,'Text outside fixed print canvas: '+str(clips))
    for ext in ('pdf','svg'):fig.savefig(path.with_suffix('.'+ext),metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None})
    plt.close(fig)
    doc=pymupdf.open(path.with_suffix('.pdf'));page=doc[0]
    require(abs(page.rect.width/72*25.4-width)<.02 and not page.get_images(),'Vector PDF width/raster boundary differs')
    page.get_pixmap(matrix=pymupdf.Matrix(300/72,300/72),alpha=False).save(path.with_suffix('.png'))
    return {'width_mm':width,'height_mm':page.rect.height/72*25.4,'text_canvas_clips':0,'raster_image_objects':len(page.get_images()),'PDF_fonts':len(page.get_fonts())}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    clean(a.expected_commit);out=a.out.resolve();require(out.is_relative_to(a.run.resolve()),'Own external run output required');out.mkdir(mode=0o700)
    for name in ('figures','data','reused-D'): (out/name).mkdir()
    frozen=a.run/'frozen';s2,classes,roles,hyp=derive(frozen,out);checks={}
    for width in (175,85):
        for name,fig in [('R1-S2-four-cells',r1(s2,width)),('R2-WP04-class-protection',r2(classes,width)),('D1-admission-flow',d1(width,hyp)),('D2-support-units',d2(roles,width))]:
            stem=f'{name}-{width}mm';checks[stem]=save(fig,out/'figures'/stem,width)
    for name in ('scene-effects','class-effects'):
        for ext in ('pdf','svg'):
            src=a.run/'D-source/paper/submission-prep/figures'/f'{name}.{ext}'
            if src.is_file():shutil.copyfile(src,out/'reused-D'/src.name)
    write(out/'figure-receipt.json',{'producer':a.expected_commit,'frozen_receipt_sha256':sha(frozen/'frozen-receipt.json'),
          'figures':checks,'figure_files':{f.name:sha(f) for f in sorted((out/'figures').iterdir())},
          'new_neural_calls':0,'optimizer_updates':0,'query_arrays_read':0,'prospective_performance_values':None})
    print(json.dumps({'status':'FIGURES_RENDERED','fixed_width_figures':len(checks),'vector_files':len(checks)*2,'raster_images_in_PDFs':0}))


if __name__=='__main__':main()
