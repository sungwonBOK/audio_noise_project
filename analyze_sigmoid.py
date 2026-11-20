"""Analyze saved sigmoid scores only; never retrain or change operating thresholds."""
import argparse,ast,csv,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_fonts import korean_font_family

ROOT=Path(__file__).resolve().parent
CLASSES=['car_horn','siren','door_wood_knock','crying_baby','glass_breaking']
NAMES=['자동차 경적','사이렌','문 두드림','아기 울음','유리 깨짐']

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def export_csv(p,rows):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def point(y,s,t):
 pred=s>=t;tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum())
 precision=tp/(tp+fp) if tp+fp else None
 return dict(tp=tp,fp=fp,fn=fn,predicted_positive=tp+fp,precision=precision,
             recall=tp/(tp+fn),f1=2*tp/max(1,2*tp+fp+fn))
def describe(a):
 return {'n':len(a),'mean':float(a.mean()),'q10':float(np.quantile(a,.1)),
         'median':float(np.median(a)),'q90':float(np.quantile(a,.9)),'q95':float(np.quantile(a,.95)),'max':float(a.max())}

def analyze(run,out):
 out.mkdir(parents=True,exist_ok=True)
 data={};hashes={};identity={};dist=[];snr=[];sweeps=[];prrows=[];op=[];models=[]
 for count in [6,18,30,43]:
  for mode in ['clean','mixed']:
   base=run/f'noise_{count:02d}'/mode
   thpath=base/'validation/thresholds.json';hashes[str(thpath.relative_to(run))]=sha(thpath)
   th=load(thpath);assert th['classes']==CLASSES and th['selection_split']=='val'
   thresholds=np.asarray(th['thresholds'])
   model={'noise_classes':count,'mode':mode}
   for split,expected in [('val',650),('test',652)]:
    path=base/('validation' if split=='val' else 'test')/'predictions.csv'
    hashes[str(path.relative_to(run))]=sha(path)
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==expected and all(r['skipped']=='False' for r in rows)
    y=np.asarray([ast.literal_eval(r['labels']) for r in rows],dtype=bool)
    s=np.asarray([ast.literal_eval(r['scores']) for r in rows],dtype=np.float64)
    assert y.shape==s.shape==(expected,5) and np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
    assert (y.sum(1)<=1).all() and (y.sum(0)==56).all()
    sig=[tuple(r[k] for k in ['id','kind','noise_group','noise_class','snr_db','labels']) for r in rows]
    if split in identity:assert sig==identity[split]
    else:identity[split]=sig
    report=load(base/('validation' if split=='val' else 'test')/'report.json')
    assert report['checkpoint_sha256']==th['checkpoint_sha256']
    classes=[]
    for i,c in enumerate(CLASSES):
     masks={'present':y[:,i],'absent':~y[:,i],'background':~y.any(1),
            'other_target':y.any(1)&~y[:,i]}
     for group,mask in masks.items():
      desc=describe(s[mask,i]);dist.append(dict(noise_classes=count,mode=mode,split=split,category=c,group=group,**desc))
     ap=float(average_precision_score(y[:,i],s[:,i]));chosen=point(y[:,i],s[:,i],thresholds[i])
     assert chosen['tp']==report['ungated']['metrics']['tp'][i]
     assert chosen['fp']==report['ungated']['metrics']['fp'][i]
     assert chosen['fn']==report['ungated']['metrics']['fn'][i]
     op.append(dict(noise_classes=count,mode=mode,split=split,category=c,threshold=float(thresholds[i]),ap=ap,**chosen))
     classes.append(dict(category=c,ap=ap,present=describe(s[y[:,i],i]),
                         absent=describe(s[~y[:,i],i]),background=describe(s[~y.any(1),i]),
                         other_target=describe(s[y.any(1)&~y[:,i],i]),threshold=float(thresholds[i]),operating=chosen))
     if split=='val':
      last_recall=1.0
      for t in np.linspace(0,1,101):
       v=point(y[:,i],s[:,i],float(t));assert v['recall']<=last_recall+1e-12;last_recall=v['recall']
       sweeps.append(dict(noise_classes=count,mode=mode,category=c,threshold=float(t),**v))
      pr,rc,ts=precision_recall_curve(y[:,i],s[:,i])
      for j,(pv,rv) in enumerate(zip(pr,rc)):
       prrows.append(dict(noise_classes=count,mode=mode,category=c,precision=float(pv),recall=float(rv),
                          threshold=float(ts[j]) if j<len(ts) else None,terminal_convention=j==len(ts)))
     for group in ['all','candidate_pool','heldout_types']:
      for sn in [0,5,10]:
       mask=np.asarray([r['kind']=='mixed' and float(r['snr_db'] or -999)==sn and (group=='all' or r['noise_group']==group) for r in rows])&y[:,i]
       desc=describe(s[mask,i]);snr.append(dict(noise_classes=count,mode=mode,split=split,category=c,noise_group=group,snr_db=sn,**desc))
     mask=np.asarray([r['kind']=='clean' for r in rows])&y[:,i]
     snr.append(dict(noise_classes=count,mode=mode,split=split,category=c,noise_group='clean',snr_db=None,**describe(s[mask,i])))
    scores_by_snr={}
    for sn in [0,5,10]:
     mask=np.asarray([r['kind']=='mixed' and float(r['snr_db'] or -999)==sn for r in rows])
     assert int(mask.sum())==80
     scores_by_snr[str(sn)]=describe(s[mask][y[mask]])
    cleanmask=np.asarray([r['kind']=='clean' for r in rows]);assert int(cleanmask.sum())==40
    model[split]={'macro_ap':float(np.mean([c['ap'] for c in classes])),
                  'classes':classes,'target_score_by_snr':scores_by_snr,
                  'clean_target_score':describe(s[cleanmask][y[cleanmask]])}
    data[count,mode,split]=(rows,y,s,thresholds)
   models.append(model)
 for rel,digest in hashes.items():assert sha(run/rel)==digest
 summary={'models':models,'classes':CLASSES,'class_names_ko':NAMES,
          'analysis':'Saved sigmoid scores, no sigmoid reapplied, no retraining, no new threshold selection.',
          'scope':'All eight fixed models; validation distributions, SNR response and threshold sweeps. Test summaries and existing operating-point verification only.',
          'validation_positive_windows_per_class':56,'validation_absent_windows_per_class':594,
          'validation_background_windows':370,'validation_other_target_windows_per_class':224,
          'source_hashes':hashes,'source_unchanged':True,'existing_metrics_reproduced':True,
          'threshold_sweep':'Validation only, 0..1 step 0.01 for display; undefined precision stored blank/null.',
          'ap_definition':'Non-interpolated average precision, scikit-learn average_precision_score; not trapezoidal PR area.'}
 save(out/'summary.json',summary)
 export_csv(out/'score_distributions.csv',dist);export_csv(out/'snr_scores.csv',snr)
 export_csv(out/'validation_threshold_sweep.csv',sweeps);export_csv(out/'validation_pr_curve_points.csv',prrows)
 export_csv(out/'existing_operating_points.csv',op)
 export_csv(out/'model_summary.csv',[dict(noise_classes=m['noise_classes'],mode=m['mode'],val_macro_ap=m['val']['macro_ap'],test_macro_ap=m['test']['macro_ap'],
         val_snr0=m['val']['target_score_by_snr']['0']['mean'],val_snr5=m['val']['target_score_by_snr']['5']['mean'],
         val_snr10=m['val']['target_score_by_snr']['10']['mean'],val_clean=m['val']['clean_target_score']['mean']) for m in models])
 return summary,data,sweeps

def plots(out,summary,data,sweeps):
 plt.rcParams.update({'font.family':korean_font_family(),'font.size':9,'axes.unicode_minus':False,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'lines.linewidth':1.5})
 fg=out/'figures';fg.mkdir(exist_ok=True)
 colors={'clean':'#246BA0','mixed':'#D46224'};labels={'clean':'깨끗한 학습','mixed':'혼합 학습'}
 def finish(fig,name):fig.savefig(fg/name,dpi=220);plt.close(fig)
 def panels(height=4.6):return plt.subplots(3,2,figsize=(7,height),layout='constrained')
 def tidy(ax,i,xlabel,ylabel):
  ax.set_title(NAMES[i]);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(alpha=.18)
 for count in [30,43]:
  fig,axs=panels(4.8)
  for i,ax in enumerate(axs.flat[:5]):
   for mode in ['clean','mixed']:
    _,y,s,th=data[count,mode,'val']
    for present,ls,word in [(True,'-','있음'),(False,'--','없음')]:
     ax.hist(s[y[:,i] if present else ~y[:,i],i],bins=np.linspace(0,1,21),density=True,histtype='step',linestyle=ls,color=colors[mode],label=labels[mode]+' 해당 소리 '+word)
    ax.axvline(th[i],color=colors[mode],ls=':',lw=1)
   tidy(ax,i,'Sigmoid 점수','밀도');ax.set_xlim(0,1);ax.set_ylim(0,21);ax.set_xticks([0,.5,1])
  axs.flat[5].axis('off');ha,la=axs.flat[0].get_legend_handles_labels();axs.flat[5].legend(ha,la,loc='center',frameon=False,fontsize=8)
  finish(fig,f'sigmoid_distribution_{count}.png')
 fig,axs=plt.subplots(2,2,figsize=(7,3.35),layout='constrained',sharex=True,sharey=True)
 for count,ax in zip([6,18,30,43],axs.flat):
  for mode in ['clean','mixed']:
   m=next(m for m in summary['models'] if m['noise_classes']==count and m['mode']==mode)
   ax.plot([0,5,10],[m['val']['target_score_by_snr'][str(sn)]['mean'] for sn in [0,5,10]],marker='o',ms=3,color=colors[mode],label=labels[mode])
   ax.axhline(m['val']['clean_target_score']['mean'],color=colors[mode],ls=':',lw=1)
  ax.set_title(f'{count}종');ax.set_xticks([0,5,10]);ax.set_ylim(0,1.02);ax.grid(alpha=.18)
 for ax in axs[:,0]:ax.set_ylabel('정답 소리 평균 점수')
 for ax in axs[1,:]:ax.set_xlabel('혼합 SNR dB')
 ha,la=axs.flat[0].get_legend_handles_labels();fig.legend(ha,la,loc='outside upper center',ncol=2,frameon=False)
 finish(fig,'sigmoid_snr_all.png')
 fig,axs=panels(3.8)
 for i,ax in enumerate(axs.flat[:5]):
  for mode in ['clean','mixed']:
   rows,y,s,th=data[30,mode,'val'];means=[]
   for sn in [0,5,10]:
    mask=np.asarray([r['kind']=='mixed' and float(r['snr_db'] or -999)==sn for r in rows])&y[:,i];means.append(float(s[mask,i].mean()))
   ax.plot([0,5,10],means,marker='o',ms=3,color=colors[mode],label=labels[mode]);mask=np.asarray([r['kind']=='clean' for r in rows])&y[:,i]
   ax.scatter([15],[s[mask,i].mean()],color=colors[mode],s=16)
  tidy(ax,i,'SNR dB 또는 단독','정답 소리 평균 점수');ax.set_ylim(0,1.02);ax.set_xticks([0,5,10,15],['0','5','10','단독'])
 axs.flat[5].axis('off');axs.flat[5].legend(*axs.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
 finish(fig,'sigmoid_snr_classes_30.png')
 for metric,klabel in [('precision','정밀도'),('recall','재현율')]:
  fig,axs=panels(4.4)
  for i,ax in enumerate(axs.flat[:5]):
   for mode in ['clean','mixed']:
    curve=[r for r in sweeps if r['noise_classes']==30 and r['mode']==mode and r['category']==CLASSES[i]]
    ax.plot([r['threshold'] for r in curve],[np.nan if r[metric] is None else r[metric] for r in curve],color=colors[mode],label=labels[mode])
    _,y,s,th=data[30,mode,'val'];v=point(y[:,i],s[:,i],th[i]);ax.scatter([th[i]],[v[metric]],color=colors[mode],s=24,edgecolor='black',lw=.4,zorder=3)
   tidy(ax,i,'검출 기준',klabel);ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.set_xticks([0,.5,1])
  axs.flat[5].axis('off');axs.flat[5].legend(*axs.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
  finish(fig,f'sigmoid_threshold_{metric}_30.png')
 fig,axs=panels(4.5)
 for i,ax in enumerate(axs.flat[:5]):
  for mode in ['clean','mixed']:
   _,y,s,th=data[30,mode,'val'];pr,rc,_=precision_recall_curve(y[:,i],s[:,i]);ax.step(rc,pr,where='post',color=colors[mode],label=labels[mode]);v=point(y[:,i],s[:,i],th[i]);ax.scatter([v['recall']],[v['precision']],color=colors[mode],s=24,edgecolor='black',lw=.4,zorder=3)
  tidy(ax,i,'재현율','정밀도');ax.set_xlim(0,1.02);ax.set_ylim(0,1.02)
 axs.flat[5].axis('off');axs.flat[5].legend(*axs.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
 finish(fig,'sigmoid_pr_30.png')
 print('Saved',len(list(fg.glob('*.png'))),'figures')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',default=str(ROOT/'runs/noise_expansion_20261115'));p.add_argument('--out',default=None);a=p.parse_args()
 run=Path(a.experiment).resolve();out=Path(a.out).resolve() if a.out else run/'sigmoid_analysis'
 summary,data,sweeps=analyze(run,out);plots(out,summary,data,sweeps)
 for m in summary['models']:
  print(m['noise_classes'],m['mode'],'val AP',round(m['val']['macro_ap'],6),'SNR scores',[round(m['val']['target_score_by_snr'][str(sn)]['mean'],4) for sn in [0,5,10]])
 print('Source predictions and thresholds unchanged; original TP/FP/FN reproduced.')

if __name__=='__main__':main()
