"""Measured wheel chatter and the separate action-rate penalty adjustment."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
a=np.genfromtxt(root/'results/auto_training_implicit_20pct_20261004/round_012_optimization/replay/trace.csv',delimiter=',',names=True)
d=a[(a['case_id']==0)&(a['seed']==43)&(a['time_s']>=9)]
fig,axes=plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
window=d[d['time_s']<=9.4]
for side,color in [('left','#28658e'),('right','#b3473d')]:
    axes[0].plot(window['time_s']-9,window[f'{side}_wheel_ref_rad_s'],'.-',lw=.8,color=color,label=side)
    x=d[f'{side}_wheel_ref_rad_s'];freq=np.fft.rfftfreq(len(x),.02)
    axes[1].plot(freq,abs(np.fft.rfft(x-x.mean()))/len(x),lw=.9,color=color,label=side)
axes[0].set(title='Measured stand commands, iteration 3000',xlabel='Time after 9 s (s)',ylabel='Wheel reference (rad/s)')
axes[1].set(title='Last 3 s of the same replay',xlabel='Frequency (Hz)',ylabel='DFT magnitude (rad/s)')
delta=np.linspace(0,2,200)
axes[2].plot(delta,-.01*2*delta**2,label='Old weight -0.01',color='#b3473d')
axes[2].plot(delta,-.5*2*delta**2,label='New weight -0.5',color='#28658e')
axes[2].set(title='Penalty formula, both wheels',xlabel='Normalized change per wheel',ylabel='Action-rate reward/s')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
path=root/'docs/figures/wheel_flutter.png';fig.savefig(path,dpi=180);plt.close(fig);print(path)
