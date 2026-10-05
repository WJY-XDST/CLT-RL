"""Draw the actual old and revised penalty formulas, not simulated trajectories."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

output=Path(__file__).resolve().parents[1]/'docs/figures/reward_clipping.png'
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
speed=np.linspace(-.8,.8,500)
raw=30*(speed-.3)**2
axes[0].plot(speed,-np.minimum(raw,1),label='Old cap: 1 reward/s',color='#b3473d')
axes[0].plot(speed,-np.minimum(raw,43.2),label='Revised cap: 43.2',color='#28658e')
axes[0].axvline(.3,color='gray',ls='--',lw=.8)
axes[0].set(title='Command +0.3 m/s',xlabel='Measured forward speed (m/s)',ylabel='Squared-error penalty (reward/s)')
error=np.linspace(0,3,300)
raw=1.52587890625*2*error**2
axes[1].plot(error,-np.minimum(raw,1),label='Old cap: 1 reward/s',color='#b3473d')
axes[1].plot(error,-np.minimum(raw,12.20703125),label='Revised cap: 12.207',color='#28658e')
axes[1].set(title='Same tracking error on both wheels',xlabel='Wheel reference tracking error (rad/s)',ylabel='Standing-wheel penalty (reward/s)')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(output,dpi=180)
plt.close(fig)
print(output)
