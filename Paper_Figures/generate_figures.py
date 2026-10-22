"""Plot existing FedContextTrust outputs. Never trains, changes thresholds, or invents results."""
from pathlib import Path
import argparse, glob, hashlib, html, json, platform, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                            average_precision_score, confusion_matrix)
from make_diagrams import generate as diagrams

COLORS=['#2166ac','#d6604d','#218768','#825aa3','#be6a16']
STREAMS=['movement','timing','context']
DECISIONS=['Trusted','Uncertain','Suspicious']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
 'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42,'svg.fonttype':'none'})

def ratio(a,b): return float(a/b) if b else np.nan
def binary_metrics(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return dict(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),accuracy=ratio(tp+tn,len(y)),
                precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),fpr=ratio(fp,fp+tn),
                f1=ratio(2*tp,2*tp+fp+fn))

def numeric(df,cols):
    for c in cols:
        df[c]=pd.to_numeric(df[c],errors='raise')
        if not np.isfinite(df[c]).all(): raise ValueError(f'{c} contains missing or non-finite values.')

class Report:
    def __init__(self,out): self.out=out;self.items=[];self.warnings=[];self.inputs=[]
    def source(self,path):
        path=Path(path);h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
        self.inputs.append(dict(path=str(path.resolve()),sha256=h.hexdigest()))
    def skip(self,msg): self.warnings.append(msg);print('NOTE:',msg)
    def save(self,fig,name,caption,interpretation):
        fig.tight_layout()
        for ext in ['pdf','svg','png']: fig.savefig(self.out/f'{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig);self.items.append(dict(name=name,caption=caption,interpretation=interpretation))
    def finish(self,summary):
        (self.out/'figure_manifest.json').write_text(json.dumps(dict(inputs=self.inputs,summary=summary,
            figures=self.items,notes=self.warnings,python=platform.python_version(),
            numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),indent=2,allow_nan=False),encoding='utf8')
        intro=('These plots use saved predictions and logs. They do not retrain the models. '
               'Test windows can overlap and share sessions; these are window-level descriptive results, '
               'not independent statistical trials. Warm-up messages without a model score are outside these '
               'prediction metrics; inspect the original coverage report for the full message denominator. '
               'No confidence intervals are fabricated.')
        md=['# Measured results and interpretation','',intro,'',f'Processed summary: `{json.dumps(summary)}`','']
        body=['<h1>Measured results and interpretation</h1>',f'<p>{html.escape(intro)}</p>']
        for it in self.items:
            md += [f'## {it["name"]}',it['caption'],'',f'![{it["name"]}]({it["name"]}.png)','',it['interpretation'],'']
            body += [f'<section><h2>{html.escape(it["name"])}</h2><p>{html.escape(it["caption"])}</p>',
                     f'<img src="{it["name"]}.png"><p>{html.escape(it["interpretation"])}</p></section>']
        md+=['## Missing data / limits']+[f'- {x}' for x in self.warnings]
        body+=['<h2>Missing data / limits</h2>']+[f'<p>{html.escape(x)}</p>' for x in self.warnings]
        (self.out/'INTERPRETATION.md').write_text('\n'.join(md),encoding='utf8')
        (self.out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>FedContextTrust results</title><style>body{font:16px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:20px;color:#243447}img{max-width:100%}section{border-top:1px solid #ddd;margin-top:32px}</style>'+''.join(body),encoding='utf8')

def curve_plots(df,r,thresholds):
    y=df.label.to_numpy(int);s=df.score.to_numpy(float);both=len(np.unique(y))==2
    n=len(y); rows=[]
    for policy,pred in [('strict',df.decision.eq('Suspicious')),('alert',df.decision.ne('Trusted'))]:
        m=binary_metrics(y,pred.astype(int));rows.append(dict(policy=policy,**m))
        cm=confusion_matrix(y,pred,labels=[0,1]);fig,axs=plt.subplots(1,2,figsize=(8,3.6))
        for ax,norm in zip(axs,[False,True]):
            den=cm.sum(1,keepdims=True)
            a=np.divide(cm,den,out=np.full_like(cm,np.nan,dtype=float),where=den!=0) if norm else cm
            ax.imshow(a,cmap='Blues',vmin=0,vmax=1 if norm else None)
            for i in range(2):
                for j in range(2):
                    val=a[i,j];t='N/A' if np.isnan(val) else (f'{val:.1%}' if norm else str(int(val)))
                    ax.text(j,i,t,ha='center',va='center',color='white' if val>(.5 if norm else max(cm.max()/2,0)) else 'black')
            ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Negative','Positive'],yticklabels=['Benign','Attack'],xlabel='Predicted',ylabel='Actual',title='Row proportions' if norm else 'Window counts');ax.grid(False)
        r.save(fig,f'R_confusion_{policy}',f'{policy.capitalize()} policy: '+('only Suspicious is positive.' if policy=='strict' else 'Uncertain and Suspicious are positive.'),
               f'TP={m["tp"]}, FP={m["fp"]}, FN={m["fn"]}, TN={m["tn"]}. Recall={m["recall"]:.4f}; FPR={m["fpr"]:.4f}. Lower off-diagonal counts are desirable. Undefined rates appear as NaN/N/A, not zero.')
    met=pd.DataFrame(rows);met.to_csv(r.out/'recomputed_policy_metrics.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4));cols=['accuracy','precision','recall','f1','fpr'];x=np.arange(len(cols))
    for i,row in met.iterrows(): ax.bar(x+(i-.5)*.35,row[cols].astype(float),.35,label=row.policy,color=COLORS[i])
    ax.set(xticks=x,xticklabels=cols,ylim=(0,1.08),ylabel='Metric value');ax.legend()
    r.save(fig,'R_policy_metrics','Strict and alert policy metrics on the same held-out windows.',
           'Recall, precision and F1 should be read together. Lower FPR is better; higher values of the other displayed metrics are better. Accuracy can be misleading when one class dominates. NaN bars are undefined.')
    if both:
        fpr,tpr,_=roc_curve(y,s);auc=roc_auc_score(y,s)
        p,rec,_=precision_recall_curve(y,s);ap=average_precision_score(y,s)
        fig,ax=plt.subplots(figsize=(5,4));ax.plot(fpr,tpr,label=f'ROC-AUC = {auc:.4f}',color=COLORS[0]);ax.plot([0,1],[0,1],'--',color='gray',label='Chance ranking')
        ax.set(xlabel='False-positive rate',ylabel='True-positive rate (recall)',xlim=(0,1),ylim=(0,1.02));ax.legend()
        r.save(fig,'R_ROC','ROC curve from continuous fused anomaly scores.',f'ROC-AUC={auc:.4f}. Curves toward the top-left indicate better score ordering. This does not imply equally good performance at the deployed thresholds.')
        fig,ax=plt.subplots(figsize=(5,4));ax.step(rec,p,where='post',label=f'AP = {ap:.4f}',color=COLORS[0]);ax.axhline(y.mean(),ls='--',color='gray',label=f'Attack prevalence = {y.mean():.3f}')
        ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02));ax.legend()
        r.save(fig,'R_precision_recall','Precision-recall curve; AP is average precision, not trapezoidal PR area.',f'AP={ap:.4f}; attack prevalence={y.mean():.4f}. Higher precision at useful recall is desirable. Compare AP only on comparable test populations.')
        fig,ax=plt.subplots(figsize=(5,4));ax.plot(fpr,1-tpr,color=COLORS[0]);ax.set(xlabel='False-positive rate',ylabel='False-negative rate',xlim=(0,1),ylim=(0,1))
        r.save(fig,'R_error_tradeoff','False-positive versus false-negative tradeoff as the decision threshold changes.',
               'Closer to the lower-left is better. Lowering the threshold generally catches more attacks but flags more benign windows. This is a descriptive test-set sweep, not permission to tune thresholds on test data.')
        # Exact score thresholds for predicted positive iff score > threshold, including ties.
        u,inv=np.unique(s,return_inverse=True);pos=np.bincount(inv,weights=y);neg=np.bincount(inv,weights=1-y)
        tp=y.sum()-np.cumsum(pos);fp=(n-y.sum())-np.cumsum(neg);fn=y.sum()-tp
        sweep=pd.DataFrame({'threshold':u,'recall':tp/y.sum(),'fpr':fp/(n-y.sum()),
          'precision':np.divide(tp,tp+fp,out=np.full_like(tp,np.nan),where=(tp+fp)>0),
          'f1':np.divide(2*tp,2*tp+fp+fn,out=np.full_like(tp,np.nan),where=(2*tp+fp+fn)>0)})
        sweep.to_csv(r.out/'threshold_sweep_descriptive_only.csv',index=False)
        fig,ax=plt.subplots(figsize=(7,4))
        for i,k in enumerate(['precision','recall','f1','fpr']): ax.plot(sweep.threshold,sweep[k],label=k,color=COLORS[i])
        for name,v in thresholds.items(): ax.axvline(v,ls='--',color='gray');ax.text(v,1.01,name,ha='center')
        ax.set(xlabel='Threshold (positive iff score > threshold)',ylabel='Metric value',ylim=(0,1.1));ax.legend()
        r.save(fig,'R_threshold_sensitivity','Descriptive sensitivity to threshold; vertical lines are saved calibration thresholds, when supplied.',
               'Use this to explain the recall/false-alarm tradeoff. Do not select the best test F1 threshold and report it as an untouched test result. Operational thresholds must come from the independent calibration procedure.')
    else: r.skip('ROC, precision-recall, error tradeoff and threshold sweep skipped: both benign and attack labels are required.')
    fig,ax=plt.subplots(figsize=(7,4))
    for cls,name,col in [(0,'Benign',COLORS[0]),(1,'Attack',COLORS[1])]:
        a=s[y==cls]
        if len(a): ax.hist(a,bins=np.linspace(0,1,41),weights=np.ones(len(a))/len(a),alpha=.55,label=f'{name} (n={len(a)})',color=col)
    for name,v in thresholds.items(): ax.axvline(v,ls='--',label=f'{name} = {v:.4f}',color='black' if name=='high' else 'gray')
    ax.set(xlabel='Fused anomaly score',ylabel='Fraction within each class');ax.legend()
    r.save(fig,'R_score_distribution','Class-normalized distributions of fused scores with saved thresholds.',
           'Separation between low-scoring benign windows and high-scoring attacks helps detection. Overlap explains false alarms and missed attacks. A score of 0.9 is not a 90% attack probability.')
    fig,ax=plt.subplots(figsize=(6,4))
    for cls,name,col in [(0,'Benign',COLORS[0]),(1,'Attack',COLORS[1])]:
        a=np.sort(s[y==cls])
        if len(a): ax.step(a,np.arange(1,len(a)+1)/len(a),where='post',label=name,color=col)
    ax.set(xlabel='Fused anomaly score',ylabel='Cumulative proportion',ylim=(0,1));ax.legend()
    r.save(fig,'R_score_ECDF','Empirical cumulative distribution of test scores, by actual class.',
           'At threshold t, the attack curve gives the fraction of attacks with score ≤ t; the complement of the benign curve gives FPR for score > t.')
    tab=pd.crosstab(df.label,df.decision).reindex(index=[0,1],columns=DECISIONS,fill_value=0)
    fig,ax=plt.subplots(figsize=(6,4));bottom=np.zeros(2)
    for i,k in enumerate(DECISIONS):
        a=tab[k].to_numpy();tot=tab.sum(1).to_numpy();v=np.divide(a,tot,out=np.zeros(2),where=tot>0);ax.bar(['Benign','Attack'],v,bottom=bottom,label=k,color=[COLORS[2],COLORS[4],COLORS[1]][i]);bottom+=v
    ax.set(ylabel='Fraction within actual class',ylim=(0,1.05));ax.legend(bbox_to_anchor=(1.02,1),loc='upper left')
    r.save(fig,'R_three_decisions','Trusted, Uncertain and Suspicious proportions for the two actual classes.',
           f'Uncertain fraction={df.decision.eq("Uncertain").mean():.4f}. Uncertain is a decision state, not a third ground-truth attack class. Empty actual classes have no bar.')
    return met

def stream_plots(df,r):
    for suffix,title in [('mse','Error'),('rank','Benign-reference rank')]:
        cols=[f'{s}_{suffix}' for s in STREAMS]
        if not all(c in df for c in cols): r.skip(f'Stream {suffix} plot skipped: expected columns {cols}.');continue
        numeric(df,cols)
        if suffix=='mse' and (df[cols]<0).any().any(): raise ValueError('MSE values cannot be negative.')
        if suffix=='rank' and ((df[cols]<0)|(df[cols]>1)).any().any(): raise ValueError('Stream ranks must lie in [0,1].')
        fig,axs=plt.subplots(1,3,figsize=(10,3.5))
        for ax,col in zip(axs,cols):
            groups=[df.loc[df.label==k,col].to_numpy() for k in [0,1]]
            for i,g in enumerate(groups):
                if len(g): ax.boxplot(g,positions=[i],widths=.55,showfliers=False)
            ax.set(xticks=[0,1],xticklabels=['Benign','Attack'],title=col.replace('_',' '),ylabel=title)
            if suffix=='mse': ax.set_yscale('symlog',linthresh=1e-5)
            else: ax.set_ylim(0,1.02)
        r.save(fig,f'R_stream_{suffix}',f'Three stream {title.lower()} distributions; boxes show median and quartiles, whiskers 1.5 IQR; outliers hidden.',
               'Better separation can indicate useful evidence. Raw MSE scales differ, so compare ranks for cross-stream interpretation. A weak stream may reveal a dataset limitation; it must not be hidden.')
    cols=[f'{s}_rank' for s in STREAMS]
    if all(c in df for c in cols):
        a=df[cols].corr(method='spearman').to_numpy();fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(a,vmin=-1,vmax=1,cmap='coolwarm')
        for i in range(3):
            for j in range(3): ax.text(j,i,'N/A' if np.isnan(a[i,j]) else f'{a[i,j]:.2f}',ha='center',va='center')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=STREAMS,yticklabels=STREAMS);ax.grid(False);fig.colorbar(im,ax=ax,label='Spearman correlation')
        r.save(fig,'R_stream_correlation','Rank correlation across the three evidence scores.',
               'Strong correlation indicates similar ordering on this dataset; weak correlation suggests differing evidence, but does not by itself prove improved detection. Constant streams have undefined correlation.')

def groups_and_timeline(df,r,thresholds):
    group='variant' if 'variant' in df else ('scenario' if 'scenario' in df else None)
    if group:
        rows=[]
        for name,g in df[df.label==1].groupby(group,dropna=False):
            rows.append(dict(group=str(name),n=len(g),strict_recall=g.decision.eq('Suspicious').mean(),alert_recall=g.decision.ne('Trusted').mean(),uncertain=g.decision.eq('Uncertain').mean()))
        a=pd.DataFrame(rows)
        if not a.empty:
            a.to_csv(r.out/'attack_group_metrics.csv',index=False);fig,ax=plt.subplots(figsize=(8,max(3,len(a)*.45)))
            x=np.arange(len(a));ax.barh(x-.18,a.strict_recall,.36,label='Strict',color=COLORS[0]);ax.barh(x+.18,a.alert_recall,.36,label='Alert',color=COLORS[1]);ax.set(yticks=x,yticklabels=[f'{v} (n={n})' for v,n in zip(a.group,a.n)],xlabel='Attack-window recall',xlim=(0,1.05));ax.legend()
            r.save(fig,'R_attack_groups',f'Attack recall grouped by {group}; sample counts shown.',
                   'Low recall identifies attack groups needing improvement. These are attack-only denominators: precision and FPR cannot be inferred from this chart. Groups are taken from saved metadata, not guessed attack names.')
    if all(c in df for c in ['session','receiver','pseudonym','rx_time']):
        # Choose the largest sender history containing attack windows; otherwise largest history.
        keys=['session','receiver','pseudonym'];g=df.groupby(keys,dropna=False).agg(n=('score','size'),attacks=('label','sum')).reset_index()
        candidates=g[g.attacks>0] if (g.attacks>0).any() else g;row=candidates.sort_values('n',ascending=False,kind='stable').iloc[0]
        mask=np.ones(len(df),bool)
        for k in keys: mask&=df[k].isna() if pd.isna(row[k]) else df[k].eq(row[k])
        a=df.loc[mask].sort_values('rx_time',kind='stable');t=pd.to_numeric(a.rx_time);t=t-t.iloc[0]
        fig,ax=plt.subplots(figsize=(8,4));ax.plot(t,a.score,color=COLORS[0],lw=1,label='Score')
        for cls,col,lab in [(0,COLORS[2],'Actual benign'),(1,COLORS[1],'Actual attack')]:
            take=a.label==cls;ax.scatter(t[take],a.loc[take,'score'],s=12,color=col,label=lab)
        for name,v in thresholds.items():ax.axhline(v,ls='--',label=name,color='gray')
        ax.set(xlabel='Seconds since first scored window',ylabel='Fused anomaly score',ylim=(0,1.03));ax.legend(fontsize=8)
        a.to_csv(r.out/'selected_timeline.csv',index=False)
        r.save(fig,'R_sender_timeline','One sender within one receiver/session; largest scored history containing attacks, if any.',
               'This descriptive case study shows score changes and threshold crossings. Selection uses ground truth to find an attack-containing example; it is not a random example or evidence of average performance. The time origin is the first scored window, not attack onset.')
    else:r.skip('Sender timeline skipped: session, receiver, pseudonym and rx_time are needed.')

def subset_metrics(path,r,recomputed):
    if not path.exists():r.skip('No metrics.csv: stream-subset comparison skipped.');return
    r.source(path);df=pd.read_csv(path)
    if not {'subset','policy','f1','recall','fpr'}.issubset(df):r.skip('metrics.csv needs subset, policy, f1, recall, fpr for stream-subset panels.');return
    if df.duplicated(['subset','policy']).any():r.skip('Multiple rows per subset/policy: select one experiment before comparing subsets.');return
    full=df[df['subset'].eq('movement+timing+context')].set_index('policy')
    for _,row in recomputed.iterrows():
        if row.policy in full.index:
            for metric in ['f1','recall','fpr']:
                if not np.isclose(float(full.loc[row.policy,metric]),float(row[metric]),atol=1e-6,equal_nan=True):
                    r.skip('Stream-subset graph skipped: metrics.csv full-stream metrics disagree with predictions. Check that both files come from the same run and metric definitions.');return
    fig,axs=plt.subplots(1,3,figsize=(12,4.5));sub=df['subset'].drop_duplicates().tolist();x=np.arange(len(sub))
    for ax,metric in zip(axs,['f1','recall','fpr']):
        for i,policy in enumerate(['strict','alert']):
            a=df[df.policy==policy].set_index('subset').reindex(sub)
            ax.bar(x+(i-.5)*.35,pd.to_numeric(a[metric]),.35,label=policy,color=COLORS[i])
        ax.set(xticks=x,xticklabels=[s.replace('movement','M').replace('timing','T').replace('context','C') for s in sub],ylim=(0,1.05),ylabel=metric);ax.tick_params(axis='x',rotation=45)
    axs[0].legend()
    r.save(fig,'R_stream_subsets','Post-training stream-subset evaluation: M=movement, T=timing, C=context.',
           'Compare full fusion against single and paired streams. These are subsets of one trained system with subset-specific calibration; they are not separately retrained ablations or comparisons with published systems. Lower FPR is better.')

def training(paths,r):
    frames=[]
    for path in paths:
        r.source(path);a=pd.read_csv(path);need=['round','samples']+[s+'_loss' for s in STREAMS]
        if not set(need).issubset(a):r.skip(f'{path}: missing training columns {need}.');continue
        numeric(a,need)
        if a['round'].duplicated().any() or (a.samples<=0).any():raise ValueError(f'Duplicate rounds or invalid sample count in {path}')
        a['client']=Path(path).parent.name;frames.append(a)
    if not frames:r.skip('No usable RSU logs: local training and client payload figures skipped.');return
    a=pd.concat(frames,ignore_index=True)
    if a.duplicated(['client','round']).any():raise ValueError('Client logs were supplied twice.')
    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    for ax,stream in zip(axs,STREAMS):
        for name,g in a.groupby('client'):
            g=g.sort_values('round');ax.plot(g['round'],g[stream+'_loss'],marker='.',label=name)
        ax.set(xlabel='Federated round',ylabel='Local training MSE',title=stream.capitalize());ax.set_yscale('symlog',linthresh=1e-5)
    axs[-1].legend(fontsize=7)
    r.save(fig,'T_local_losses','Per-RSU local training loss, averaged over the local optimization steps in each round.',
           'An overall decrease indicates better fitting of local benign data. Small rises can follow FedAvg and shuffled mini-batches. These are not global validation losses or detection accuracy.')
    counts=a.groupby('round').client.nunique();complete=counts[counts==a.client.nunique()].index
    means=[]
    for rd,g in a[a['round'].isin(complete)].groupby('round'):
        means.append({'round':rd,**{s:np.average(g[s+'_loss'],weights=g.samples) for s in STREAMS}})
    if means:
        b=pd.DataFrame(means);b.to_csv(r.out/'weighted_local_losses.csv',index=False);fig,axs=plt.subplots(1,3,figsize=(11,3.5))
        for ax,s in zip(axs,STREAMS):ax.plot(b['round'],b[s],marker='o',ms=3,color=COLORS[0]);ax.set(xlabel='Federated round',ylabel='Window-weighted local MSE',title=s.capitalize())
        r.save(fig,'T_weighted_local_losses','Sample-weighted summary of logged client losses; only rounds present in all supplied client logs.',
               'This summarizes local optimization, weighted by training-window count. It is not loss measured after averaging the weights. Confirm that logs for every registered RSU were supplied.')
    b=a.sort_values('round').groupby('client').first();fig,ax=plt.subplots(figsize=(7,4));ax.bar(b.index,b.samples,color=COLORS[0]);ax.set(ylabel='Benign training windows',xlabel='RSU');ax.tick_params(axis='x',rotation=25)
    r.save(fig,'T_client_sizes','Training-window count per supplied RSU at its first recorded round.',
           'These counts determine FedAvg weights. Unequal counts affect contributions; unequal counts alone do not prove that feature distributions are non-IID.')
    payload=['download_body_bytes','upload_body_bytes']
    if set(payload).issubset(a):
        numeric(a,payload);b=a.groupby('round')[payload].sum().sort_index()/1024**2
        for cumulative,name in [(False,'F_payload_per_round'),(True,'F_payload_cumulative')]:
            v=b.cumsum() if cumulative else b;fig,ax=plt.subplots(figsize=(7,4))
            for k,col in zip(payload,COLORS):ax.plot(v.index,v[k],marker='.',label=k.replace('_body_bytes',''),color=col)
            ax.set(xlabel='Federated round',ylabel=('Cumulative ' if cumulative else '')+'model-body MiB');ax.legend()
            r.save(fig,name,'Model-body bytes reported in the supplied client training logs.',
                   'Includes recorded round downloads/uploads only. Excludes HTTP/TLS/TCP headers, polling, retry overhead not present in the logs and post-training final downloads. This does not establish bandwidth savings against a centralized baseline.')
    if 'local_train_and_upload_seconds' in a:
        numeric(a,['local_train_and_upload_seconds']);fig,ax=plt.subplots(figsize=(7,4))
        for name,g in a.groupby('client'):
            g=g.sort_values('round');ax.plot(g['round'],g.local_train_and_upload_seconds,label=name)
        ax.set(xlabel='Federated round',ylabel='Local training + upload seconds');ax.legend(fontsize=8)
        r.save(fig,'F_client_duration','Logged training-plus-upload duration per RSU and round.',
               'This combines computation and upload; it is not pure network latency. Concurrent processes on one laptop share CPU resources and do not represent geographically deployed RSUs.')

def communication(path,r):
    if not path or not path.exists():r.skip('Cloud communication.csv absent: round-duration figure skipped.');return
    r.source(path);a=pd.read_csv(path)
    if not {'round','round_seconds'}.issubset(a):r.skip('Cloud log lacks round and round_seconds.');return
    numeric(a,['round','round_seconds']);a=a.sort_values('round');fig,ax=plt.subplots(figsize=(7,4));ax.plot(a['round'],a.round_seconds,marker='o',ms=3,color=COLORS[0]);ax.set(xlabel='Federated round',ylabel='Cloud round duration (seconds)')
    r.save(fig,'F_round_duration','Cloud-observed round duration.',
           f'Median recorded round duration={a.round_seconds.median():.3f} s. Synchronous federation waits for every RSU, so slow clients and process startup may lengthen rounds. This is training time, not per-message detection latency.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--calibration',type=Path)
    p.add_argument('--clients',nargs='*',default=[]);p.add_argument('--cloud',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise SystemExit('Output folder exists. Use a new --out folder to preserve old figures.')
    prediction=args.evaluation/'predictions.csv.gz'
    if not prediction.exists():prediction=args.evaluation/'predictions.csv'
    if not prediction.exists():raise SystemExit(f'No predictions.csv.gz or predictions.csv in {args.evaluation}. Run finalize first, or correct --evaluation.')
    df=pd.read_csv(prediction,dtype={c:str for c in ['session','receiver','pseudonym','message_id','record_id']})
    if not {'label','score','decision'}.issubset(df):raise SystemExit('Predictions must include label, score, decision. Inspect the CSV header.')
    if df.empty:raise SystemExit('No predictions in file.')
    numeric(df,['label','score'])
    if not df.label.isin([0,1]).all():raise SystemExit('Only verified ground-truth labels 0=benign and 1=attack are accepted.')
    if not df.score.between(0,1).all():raise SystemExit('Expected empirical-rank fused scores in [0,1].')
    if not df.decision.isin(DECISIONS).all():raise SystemExit('Expected decisions Trusted, Uncertain, Suspicious; warm-up/non-scored rows must be evaluated separately.')
    if 'partition' in df and not df.partition.eq('test').all():raise SystemExit('This file includes non-test rows; use the held-out evaluation predictions.')
    if 'subset' in df:
        df=df[df['subset'].eq('movement+timing+context')].copy()
        if df.empty:raise SystemExit('No full three-stream predictions in this file.')
    if 'record_id' in df and df.record_id.duplicated().any():raise SystemExit('Duplicate record_id: avoid mixing experiments or duplicate prediction exports.')
    thresholds={}
    if args.calibration:
        cal=json.loads(args.calibration.read_text(encoding='utf-8-sig'))
        vals=cal.get('thresholds',{}).get('movement+timing+context')
        if not isinstance(vals,list) or len(vals)!=2:raise SystemExit('Expected calibration.thresholds["movement+timing+context"] = [low, high].')
        lo,hi=map(float,vals)
        if not np.isfinite([lo,hi]).all() or not 0<=lo<=hi<=1:raise SystemExit('Invalid thresholds.')
        expected=np.where(df.score>hi,'Suspicious',np.where(df.score>lo,'Uncertain','Trusted'))
        if not np.array_equal(expected,df.decision.to_numpy()):raise SystemExit('Calibration thresholds do not reproduce saved decisions. Use the calibration.json from the same finalized run.')
        thresholds={'low':lo,'high':hi}
    paths=[]
    for pattern in args.clients:paths.extend(sorted(glob.glob(pattern)))
    paths=list(dict.fromkeys(paths))
    args.out.mkdir(parents=True);r=Report(args.out);r.source(prediction)
    if args.calibration:r.source(args.calibration)
    else:r.skip('Calibration not supplied: threshold lines omitted; saved decisions still evaluated.')
    diagrams(args.out/'architecture')
    recomputed=curve_plots(df,r,thresholds);stream_plots(df,r);groups_and_timeline(df,r,thresholds)
    subset_metrics(args.evaluation/'metrics.csv',r,recomputed);training(paths,r);communication(args.cloud,r)
    summary=dict(test_windows=len(df),benign=int((df.label==0).sum()),attack=int((df.label==1).sum()),
                 decision_counts={str(k):int(v) for k,v in df.decision.value_counts().items()},thresholds=thresholds,
                 result_figures=len(r.items),architecture_diagrams=6)
    r.skip('No accuracy-versus-round curve: network logs do not contain held-out per-round predictions or retain every round checkpoint.')
    r.skip('No claim of unseen-attack generalization, privacy guarantee, road validation, or physical OBU latency follows from these plots.')
    r.finish(summary);print(json.dumps(summary,indent=2));print('Open:',args.out/'report.html')

if __name__=='__main__':main()
