"""Document diagram for exact policy diagnosis and training-only precision mode."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[1]
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
ax.set_xlim(0,10);ax.set_ylim(0,5.2);ax.axis('off')

def box(x,y,w,h,text,color='#e8eef3',size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',
                              facecolor=color,edgecolor='#486171',linewidth=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontproperties=font,fontsize=size)

def arrow(a,b):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#486171',lw=1.4))

ax.text(.1,4.98,'精确诊断流程',fontproperties=font,fontsize=13,weight='bold')
box(.1,3.85,2.15,.7,'记录实际 27 维输入\n以及策略原始均值')
box(2.8,3.85,2.65,.7,'重算 actor 并验证一致性\n最大差约 8.35e-7')
box(6.0,3.85,3.75,.7,'自动微分测量局部响应\n固定权重对照后再做完整回放')
arrow((2.33,4.2),(2.7,4.2));arrow((5.53,4.2),(5.9,4.2))
ax.text(.1,3.48,'局部响应提供排查方向   整个物理闭环仍以实际回放为准',fontproperties=font,fontsize=10,color='#486171')
ax.text(.1,3.0,'轮速输出精度训练对照',fontproperties=font,fontsize=13,weight='bold')
box(.1,1.4,1.6,.8,'标准观测\n27 维')
box(2.25,1.65,3.1,.7,'actor 共享特征保持固定\n256 → 128 → 64')
box(6.15,2.05,3.45,.65,'腿角和腿长输出行 0 1 3 4\n权重与对应 std 保持固定',size=10)
box(6.15,1.05,3.45,.65,'轮速输出行 2 5 及 wheel std\n继续 PPO 更新','#fff0d8',10)
box(2.25,.25,3.1,.7,'critic 和优化器继续更新\n价值估计参与 PPO 学习','#e2f0e8')
arrow((1.8,1.8),(2.15,2.0));arrow((5.45,2.0),(6.05,2.35));arrow((5.45,1.9),(6.05,1.38))
arrow((.9,1.32),(2.15,.6))
ax.text(6.15,.45,'仅改变训练梯度\n推理仍为普通 27 输入 6 输出 MLP',fontproperties=font,fontsize=10,color='#486171')
out=ROOT/'docs/figures/policy_feedback_precision.png';out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,dpi=200);plt.close(fig);print(out)
