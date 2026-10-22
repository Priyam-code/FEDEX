"""Exact, editable vector architecture diagrams; no empirical results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
BLUE='#2166ac'; GREEN='#218768'; ORANGE='#be6a16'; INK='#243447'

def canvas():
    fig, ax=plt.subplots(figsize=(10,7))
    ax.set(xlim=(0,10),ylim=(0,7)); ax.axis('off')
    return fig,ax

def box(ax,x,y,w,h,text,color=BLUE,fs=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.09',fc=color+'10',ec=color,lw=1.4))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=INK,linespacing=1.5)

def arrow(ax,a,b,text='',rad=0):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.25,connectionstyle=f'arc3,rad={rad}'))
    if text: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.13,text,ha='center',fontsize=8,color=INK,bbox=dict(fc='white',ec='none',pad=1))

def save(fig,out,name):
    for ext in ('pdf','svg','png'): fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def generate(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    fig,ax=canvas()
    for x,t in [(0.2,'Benign JSON files\nTrain / Validation / Test'),(3.55,'Attack JSON files\nTest evaluation only'),(6.9,'Additional sessions\nSeparate receiver histories')]: box(ax,x,5.95,2.85,.75,t,fs=9)
    box(ax,2.4,4.7,5.2,.8,'Validate schema and labels; convert time units\nSort within receiver session; form causal windows')
    for x in (1.6,5,8.3): arrow(ax,(x,5.9),(5,5.5))
    box(ax,.4,3.05,2.7,1,'Training windows\nBenign RSU shards',GREEN)
    box(ax,3.65,3.05,2.7,1,'Validation sessions\nSeparate reference + calibration',ORANGE,9)
    box(ax,6.9,3.05,2.7,1,'Held-out test windows\nBenign + attack',BLUE)
    for x in (1.75,5,8.25): arrow(ax,(5,4.65),(x,4.1))
    box(ax,.4,1.4,2.7,.95,'Local model training\nCloud FedAvg',GREEN)
    box(ax,3.65,1.4,2.7,.95,'Freeze final weights\nFit ranks and thresholds',ORANGE,9)
    box(ax,6.9,1.4,2.7,.95,'Evaluate frozen detector\nReport test metrics',BLUE)
    for x in (1.75,5,8.25): arrow(ax,(x,3),(x,2.4))
    arrow(ax,(3.15,1.87),(3.6,1.87)); arrow(ax,(6.4,1.87),(6.85,1.87))
    ax.text(5,.55,'Official splits stay separate. Test labels are used only to measure performance.',ha='center',fontsize=10)
    save(fig,out,'D1_data_and_experiment_flow')

    fig,ax=canvas()
    box(ax,2.7,5.9,4.6,.7,'One RSU: scaled benign training windows',GREEN)
    xs=[.25,3.6,6.95]
    inputs=['Movement history\n10 messages × 6 features','Timing sequence\n10 messages × 7 features','Current neighbour context\n10-slot vector; road slots zero']
    models=['GRU: 6 → 32\nLast hidden state\nLinear: 32 → 16 → 6','GRU encoder: 7 → 24\nRepeat latent for 10 steps\nGRU decoder: 24 → 24\nLinear: 24 → 7','MLP encoder: 10 → 16 → 8 → 4\nMLP decoder: 4 → 8 → 16 → 10\nReLU hidden activations']
    outputs=['Next movement prediction\nOutput: 6 values','Reconstructed timing\nOutput: 10 × 7','Reconstructed context\nOutput: 10 values']
    for i,x in enumerate(xs):
        box(ax,x,4.7,2.8,.7,inputs[i],fs=9); arrow(ax,(5,5.85),(x+1.4,5.45))
        box(ax,x,2.7,2.8,1.5,models[i],fs=8.5); arrow(ax,(x+1.4,4.65),(x+1.4,4.25))
        box(ax,x,1.45,2.8,.8,outputs[i],fs=9); arrow(ax,(x+1.4,2.65),(x+1.4,2.3))
        arrow(ax,(x+1.4,1.4),(5,.95))
    box(ax,1.7,.15,6.6,.75,'Loss = mean(MSE movement + MSE timing + MSE context)\nAdam update; gradient clipping; upload locally learned weights',GREEN,9)
    save(fig,out,'D2_one_RSU_models')

    fig,ax=canvas()
    box(ax,.6,4.45,8.8,1.95,'Cloud coordinator: wait for every registered RSU\n'+r'$\theta^{r+1}=\sum_k \frac{n_k}{\sum_j n_j}\theta_k^{r}$'+'\nMatch tensors within each model; save the next global weights',BLUE,12)
    for x,name in [(.3,'RSU A'),(3.65,'RSU B'),(7,'RSU K')]:
        box(ax,x,1.65,2.65,1.1,name+'\nLocal benign windows\nDownload → Adam → upload',GREEN,9)
        arrow(ax,(x+.65,2.8),(x+.65,4.4))
        arrow(ax,(x+2,4.4),(x+2,2.8))
        ax.text(x+.65,3.55,'Upload',ha='center',rotation=90,fontsize=9,bbox=dict(fc='white',ec='none'))
        ax.text(x+2,3.55,'Fetch next',ha='center',rotation=90,fontsize=9,bbox=dict(fc='white',ec='none'))
    ax.text(5,.85,'Upward arrows: updated weights. Downward arrows: averaged global weights.',ha='center',fontsize=9)
    ax.text(5,.4,'Synchronous rounds; weights travel over HTTP/HTTPS. Raw training windows stay at RSUs.',ha='center',fontsize=9)
    save(fig,out,'D3_federated_learning')

    fig,ax=canvas()
    box(ax,2.3,5.9,5.4,.75,'New message at receiver / OBU\nUpdate sender and neighbour histories',fs=10)
    box(ax,.4,4.3,2.6,.85,'Movement prediction\nCompare with new message',fs=9)
    box(ax,3.7,4.3,2.6,.85,'Timing reconstruction\nCompare with latest window',fs=9)
    box(ax,7,4.3,2.6,.85,'Context reconstruction\nCompare with current vector',fs=9)
    for x in (1.7,5,8.3): arrow(ax,(5,5.85),(x,5.2)); arrow(ax,(x,4.25),(5,3.6))
    box(ax,2.2,2.75,5.6,.8,'Three MSE values → benign-reference ranks\nEqual-weight average → anomaly score',ORANGE)
    for x,t,c in [(.4,'Trusted\nscore ≤ low',GREEN),(3.7,'Uncertain\nlow < score ≤ high',ORANGE),(7,'Suspicious\nscore > high','#b33145')]:
        box(ax,x,1.1,2.6,.9,t,c); arrow(ax,(5,2.7),(x+1.3,2.05))
    ax.text(5,.4,'Insufficient history: Uncertain. Inference changes history, not model weights.',ha='center',fontsize=10)
    save(fig,out,'D4_OBU_inference')

    fig,ax=canvas()
    box(ax,2.6,5.85,4.8,.8,'Final global weights are frozen',GREEN)
    box(ax,.5,4.2,4,.95,'Benign reference windows\nRun all three models\nSort each stream\'s MSE values',fs=9)
    box(ax,5.5,4.2,4,.95,'Separate benign calibration windows\nRun the same frozen models\nGet three MSE values per window',fs=9)
    arrow(ax,(5,5.8),(2.5,5.2)); arrow(ax,(5,5.8),(7.5,5.2))
    box(ax,2.0,2.55,6,1,'Rank each error against the reference distribution\nrank = count(reference error ≤ current error) / (N + 1)\nFused score = mean of the three ranks',ORANGE,10)
    arrow(ax,(2.5,4.15),(4,3.6)); arrow(ax,(7.5,4.15),(6,3.6))
    box(ax,2,.95,6,1,'low = 95th percentile; high = 99th percentile\nUse the higher-order-statistic quantile rule\nSave calibration.json and keep thresholds fixed for testing',GREEN,10)
    arrow(ax,(5,2.5),(5,2))
    ax.text(5,.3,'Scores are anomaly ranks, not probabilities of an attack. Quantiles are configurable.',ha='center',fontsize=9)
    save(fig,out,'D5_offline_calibration')

    fig,ax=canvas()
    box(ax,2.0,5.85,6,.75,'Sender message sequence at one receiver\nMessages 1, 2, ... , 10, 11 in reception order')
    for y,left,right in [(4.25,'Movement input: messages 1–10','Target: movement of message 11'),(2.85,'Timing input: messages 2–11','Reconstruct the same timing window'),(1.45,'Context input: at message 11','Reconstruct current neighbour vector')]:
        box(ax,.35,y,4.2,.85,left,fs=10);box(ax,5.45,y,4.2,.85,right,GREEN,10);arrow(ax,(4.6,y+.42),(5.4,y+.42))
    ax.text(5,.5,'Other senders arriving earlier update neighbour context; later messages are never used.',ha='center',fontsize=9)
    save(fig,out,'D6_window_and_target')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',default='architecture');a=p.parse_args();generate(a.out)
