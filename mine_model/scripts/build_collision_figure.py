"""Matched-weight tire geometry and measured acceptance comparison."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[1]
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
before=json.loads((ROOT/'results/wheel_head_precision_5399_5600_20261004/early_replay_5500/assessment.json').read_text())
after=json.loads((ROOT/'results/smooth_tire_collision_5500_20261004/replay/assessment.json').read_text())
if before['checkpoint_sha256']!=after['checkpoint_sha256']:raise ValueError('Comparison weights differ')
keys=['speed_mae_m_s','yaw_mae_rad_s','height_mae_m','leg_length_mae_m','leg_angle_mae_rad','wheel_speed_mae_rad_s','body_p95_deg']
def ratios(report):
    return [max(t['metrics'][k]/t['limits'][k] for c in report['conditions'] for t in c['trials']) for k in keys]
fig=plt.figure(figsize=(10,6.4),layout='constrained');grid=fig.add_gridspec(2,2,height_ratios=[1.5,1.])
for i,n in enumerate((18,96)):
    ax=fig.add_subplot(grid[0,i]);a=np.linspace(0,2*np.pi,n+1);c=np.linspace(0,2*np.pi,500)
    ax.plot(60*np.cos(c),60*np.sin(c),'--',color='#777777',lw=1,label='半径 60 mm 圆轮')
    ax.fill(60*np.cos(a),60*np.sin(a),color=('#f2c894' if n==18 else '#b9d6e8'),alpha=.5)
    ax.plot(60*np.cos(a),60*np.sin(a),color=('#a66018' if n==18 else '#28658e'),lw=1.5,label=f'{n} 边碰撞外轮廓')
    ax.set_aspect('equal');ax.set_xlim(-67,67);ax.set_ylim(-67,67);ax.set_xticks([-60,0,60]);ax.set_yticks([-60,0,60]);ax.grid(alpha=.2)
    ax.set_title(f'{n} 边轮胎碰撞  平面中点径向差 {60*(1-np.cos(np.pi/n)):.3f} mm',fontproperties=font,fontsize=12)
    ax.set_xlabel('轮轴局部 X (mm)',fontproperties=font);ax.set_ylabel('轮轴局部 Y (mm)',fontproperties=font)
    ax.legend(prop=font,fontsize=9,loc='upper right')
ax=fig.add_subplot(grid[1,:]);x=np.arange(len(keys));ax.bar(x-.18,ratios(before),.36,label='18 边 原碰撞',color='#d8974c');ax.bar(x+.18,ratios(after),.36,label='96 边 新变体',color='#28658e')
ax.axhline(1,color='#777777',ls='--',lw=1);ax.set_ylim(0,1.5);ax.set_xticks(x,['速度','偏航','高度','腿长','腿角','轮速','姿态'],fontproperties=font)
ax.set_ylabel('最差误差 / 验收上限',fontproperties=font);ax.set_title('相同 5500 权重  相同指令及种子  27 次完整回放',fontproperties=font,fontsize=12)
ax.legend(prop=font,ncol=2,loc='upper right');ax.grid(axis='y',alpha=.2)
out=ROOT/'docs/figures/tire_collision_comparison.png';fig.savefig(out,dpi=200);plt.close(fig);print(out)
