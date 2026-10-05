"""Plot measured same-weight initialization/control ablations for the guide."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1]
experiment=root/'results/speed_response_initialization_2199_20261004'
reports=[
    root/'results/auto_training_implicit_20pct_20261004/round_006_optimization/replay/assessment.json',
    experiment/'replay/assessment.json',
    experiment/'angle_0p25/damping_2p0_height_reference/replay/assessment.json',
]
labels=['Original policy','Speed-column initialization','Angle initialization + D2 + height reference']
colors=['#b3473d','#d69c43','#28658e']
data=[json.loads(p.read_text()) for p in reports]
cases=['forward_0p3','forward_0p6','reverse_0p3','reverse_0p6']
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
x=np.arange(len(cases));width=.24
for index,(report,label,color) in enumerate(zip(data,labels,colors)):
    conditions={c['name']:c for c in report['conditions']}
    for ax,key in zip(axes,['speed_mae_m_s','body_p95_deg']):
        values=[max(t['metrics'][key] for t in conditions[c]['trials'] if t['metrics']) for c in cases]
        ax.bar(x+(index-1)*width,values,width,label=label,color=color)
axes[0].plot(x,[.06,.12,.06,.12],'k--',lw=1,label='Acceptance limit')
axes[1].axhline(5,color='k',ls='--',lw=1,label='Acceptance limit')
for ax in axes:
    ax.set_xticks(x,['Fwd 0.3','Fwd 0.6','Rev 0.3','Rev 0.6'])
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set(title='Worst-seed steady speed error',ylabel='MAE (m/s)')
axes[1].set(title='Worst-seed upright posture',ylabel='Pitch / roll p95 (degrees)')
axes[1].legend(fontsize=7,loc='upper right')
path=root/'docs/figures/policy_initialization_comparison.png'
fig.savefig(path,dpi=180);plt.close(fig)
print(path)
