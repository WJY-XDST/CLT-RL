"""Plot measured actuator integration contrast for the illustrated tutorial."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/figures';OUT.mkdir(exist_ok=True)
fig,axes=plt.subplots(2,1,figsize=(11,5.5),gridspec_kw={'height_ratios':[1.4,1.]},layout='constrained')
for run,label,color in [('explicit_hold_audit_20261004','Explicit joint PD','#b43b39'),('joint_hold_audit_20261004','Implicit joint Drive','#246b8b')]:
    with (ROOT/'results'/run/'trace.csv').open() as stream:
        rows=[r for r in csv.DictReader(stream) if r['case']=='joint_hold_10Nm']
    axes[0].plot([float(r['time_s']) for r in rows],[float(r['height_m']) for r in rows],label=label,color=color,lw=2)
axes[0].axhline(.294187737,color='#777777',ls='--',label='CAD assembly height')
axes[0].set(xlabel='Time (s)',ylabel='Root height (m)',title='Same CAD q=0 target, 10 Nm limit, Kp=1000 / Kd=10; vertical guide')
axes[0].grid(alpha=.2);axes[0].legend(loc='center right')
ax=axes[1];ax.axis('off');ax.set(xlim=(0,10),ylim=(0,3))
labels=[('Policy output',1),('Virtual leg\nL_ref, theta_ref',3.5),('Five-bar inverse\nq_B_ref, q_L_ref',6),('Implicit joint Drive\n+ loop constraints',8.8)]
for text,x in labels:
    ax.text(x,1.6,text,ha='center',va='center',fontsize=10,bbox={'boxstyle':'round,pad=.6','fc':'#eef4f7','ec':'#345d72'})
for a,b in zip(labels,labels[1:]):
    ax.annotate('',xy=(b[1]-.95,1.6),xytext=(a[1]+.95,1.6),arrowprops={'arrowstyle':'->','color':'#345d72'})
ax.text(5,.25,'FK and velocity observations remain relative to base_link; actuator execution contract changes.',ha='center',fontsize=10)
fig.savefig(OUT/'implicit_actuator_diagnosis.png',dpi=180)
fig.savefig(OUT/'implicit_actuator_diagnosis.svg')
plt.close(fig)
