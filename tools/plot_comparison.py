#!/usr/bin/env python3
"""All historical baseline lines, plus supplied fresh PPO/SFT curves."""
import argparse,csv,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--curve',action='append',default=[])
    parser.add_argument('--metric',choices=['base_macro','config_mean'],default='base_macro')
    parser.add_argument('--out',default='runs/portable_comparison/all_methods');args=parser.parse_args()
    folder=ROOT/'runs/sft_teacher64_bheft_original_20261102/all_baselines_val862'
    with (folder/'baseline_values.csv').open() as stream:baselines=list(csv.DictReader(stream))
    with (folder/'sft_curve.csv').open() as stream:history=list(csv.DictReader(stream))
    fig,axes=plt.subplots(1,2,figsize=(16,6),layout='constrained');colors=plt.get_cmap('tab20')
    for ax in axes:
        for i,row in enumerate(baselines):
            ax.axhline(float(row[args.metric]),label=f"{row['method']} ({float(row[args.metric]):.4f})",
                       color=colors(i%20),linestyle=['--',':','-.'][i//20],lw=.9,alpha=.8)
        ax.plot([int(r['step']) for r in history],[float(r[args.metric]) for r in history],color='black',lw=2,label='Historical SFT / teacher64v2')
        for path in args.curve:
            records=json.loads(Path(path).read_text());xs=[];ys=[]
            for record in records:
                xs.append(record.get('actor_updates',record.get('step')))
                ys.append(record.get('C_HEFT_'+args.metric,record.get(args.metric,
                    record.get('C_over',{}).get('HEFT') if args.metric=='base_macro' else None)))
            assert all(v is not None for v in xs+ys),'Curve has no requested metric'
            ax.plot(xs,ys,lw=2,label=str(Path(path).parent.name))
        ax.set(xlabel='Optimizer updates (each curve uses its own SFT/PPO update count)',ylabel='C / HEFT (lower is better)')
        ax.grid(alpha=.15)
    axes[0].set_title('All methods / full range');axes[1].set(title='Same values / zoom',ylim=(.79,1.12))
    axes[1].legend(fontsize=6,loc='upper left',bbox_to_anchor=(1.01,1))
    out=ROOT/args.out;out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(str(out)+'.'+ext,dpi=220,bbox_inches='tight')
    print(str(out)+'.png/pdf')

if __name__=='__main__':main()
