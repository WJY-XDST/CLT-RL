"""Draw the measured physical linkage and its wheel-level five-bar equivalent."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from five_bar import FiveBar

root=Path(__file__).resolve().parents[1]
f=FiveBar("right")
p={key:value*1000 for key,value in f.points([0,0]).items()}
model=json.loads((root/'source/user_model.json').read_text())
joints={j['name']:j for j in model['joints']}
p['K']=np.array(joints['RS_joint']['origin']['xyz'][1:])
p['Q']=np.array(joints['RL3_joint']['origin']['xyz'][1:])
p['D']=np.array(json.loads((root/'config/closures.json').read_text())['joints'][0]['anchor_cad_mm'][1:])
fig,axes=plt.subplots(1,2,figsize=(11,7),layout='constrained')

def line(ax, points, keys, color, label, style='-'):
    xy=np.array([points[k] for k in keys])
    ax.plot(xy[:,1],xy[:,0],style+'o',color=color,lw=2.5,ms=5,label=label)

line(axes[0],p,['O','A','P'],'#dc6b20','RL + RL1')
line(axes[0],p,['P','C','Q','P'],'#8b5cc7','RL2 rigid lever')
line(axes[0],p,['O','C','K'],'#237fb0','RB')
line(axes[0],p,['K','D'],'#32a277','RS lever arm')
line(axes[0],p,['K','W'],'#32a277','RS to wheel')
line(axes[0],p,['Q','D'],'#e15162','RL3')
for key,(y,z) in p.items():axes[0].annotate(key,(z,y),xytext=(6,5),textcoords='offset points',weight='bold')
axes[0].set_title('Physical mechanism\nFive-bar loop O-A-P-C-O + parallelogram C-K-D-Q')
equiv={key:p['O']+f.scale*(p[key]-p['O']) for key in ('O','A','P','C')}
line(axes[1],equiv,['O','A','P'],'#dc6b20','RL virtual branch: 210 + 250 mm')
line(axes[1],equiv,['O','C','P'],'#237fb0','RB branch: 210 + 250 mm')
line(axes[1],p,['O','P'],'#777777','Original internal endpoint P',':')
for key,label in [('O','O: coaxial motors'),('A',"A'"),('C','K'),('P','W: wheel')]:
 y,z=equiv[key];axes[1].annotate(label,(z,y),xytext=(6,5),textcoords='offset points',weight='bold')
axes[1].set_title('Equivalent five-bar at the wheel\nW = O + (20/9)(P - O)')
for ax in axes:
 ax.set_aspect('equal');ax.set_xlabel('CAD Z (mm)');ax.set_ylabel('CAD Y (mm)')
 ax.margins(0.16)
 ax.grid(alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,fontsize=8)
fig.savefig(root/'docs/figures/five_bar_mapping.png',dpi=180)
