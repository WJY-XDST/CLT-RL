"""Technical schematics for the open-chain to closed-chain implementation guide."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT=Path(__file__).resolve().parents[1]/"docs/figures"
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})
fig,axs=plt.subplots(2,1,figsize=(12,6.3),layout="constrained")
points={"base_link":(0,1),"RB":(1.8,1.8),"RS":(4.0,1.8),"RW":(6.3,1.8),"RL":(1.8,.3),"RL1":(3.3,.3),"RL2":(4.8,.3),"RL3":(6.3,.3)}
edges=[("base_link","RB"),("RB","RS"),("RS","RW"),("base_link","RL"),("RL","RL1"),("RL1","RL2"),("RL2","RL3")]
for idx,ax in enumerate(axs):
    for a,b in edges:ax.plot([points[a][0],points[b][0]],[points[a][1],points[b][1]],color="#78909c",lw=2,zorder=1)
    for name,(x,y) in points.items():
        ax.scatter(x,y,s=650 if name!="base_link" else 1400,color="#e8f1f7",edgecolors="#28566f",zorder=2)
        ax.text(x,y,name,ha="center",va="center",fontsize=10,zorder=3)
    if idx:
        for a,b,rad,xy,label in [("RL2","RB",.34,(3.2,1.12),"RL2 middle hole - RB middle hole"),("RL3","RS",-.15,(6.0,1.08),"RL3 end hole\nRS end hole")]:
            ax.add_patch(FancyArrowPatch(points[a],points[b],connectionstyle=f"arc3,rad={rad}",arrowstyle="-",linestyle="--",lw=2.5,color="#c0392b",shrinkA=18,shrinkB=18))
            ax.text(*xy,label,color="#a93226",fontsize=10,ha="center",va="center",bbox=dict(facecolor="white",edgecolor="none",pad=1))
    ax.set_title("1  URDF spanning tree" if not idx else "2  USD tree plus two passive revolute closures",loc="left",fontweight="bold")
    ax.set_xlim(-.65,7.3);ax.set_ylim(-.2,2.3);ax.axis("off")
fig.suptitle("Right leg topology  |  Left leg is mirrored  |  Node names omit _link",fontsize=13)
fig.savefig(OUT/"topology.png",dpi=200);plt.close(fig)

fig,ax=plt.subplots(figsize=(11,3.2),layout="constrained")
for center,title,names in [((1.2,1.0),"CAD frame",["+Z forward","+X left","+Y up"]),((6.8,1.0),"Project base_link",["+X forward","+Y left","+Z up"])]:
    x,y=center
    for (dx,dy),color,label in zip([(1.4,0),(-.7,-.7),(0,1.35)],["#cb4335","#229954","#2874a6"],names):
        ax.annotate("",xy=(x+dx,y+dy),xytext=(x,y),arrowprops=dict(arrowstyle="->",lw=3,color=color))
        ax.text(x+dx*1.1,y+dy*1.1,label,ha="center",va="bottom" if dy>=0 else "top",color=color,fontsize=11)
    ax.text(x,y-1.08,title,ha="center",fontweight="bold")
ax.text(4.1,1.35,"new XYZ = old ZXY",ha="center",fontsize=14)
ax.annotate("",xy=(5.5,1.),xytext=(2.9,1.),arrowprops=dict(arrowstyle="->",lw=2,color="#555555"))
ax.set_xlim(-.4,9.2);ax.set_ylim(-.5,3.0);ax.axis("off")
fig.savefig(OUT/"coordinate_frames.png",dpi=200);plt.close(fig)

fig,ax=plt.subplots(figsize=(12,2.5),layout="constrained")
items=[("URDF + meshes","Spanning tree"),("Normalized URDF","Base frame + names"),("Closed-chain USD","4 revolute closures"),("Isaac Lab config","Mass, limits, contacts"),("Five-bar VMC","Jacobian + raw torque")]
for i,(title,sub) in enumerate(items):
    x=i*2.3
    ax.text(x,.6,title,ha="center",fontsize=11,fontweight="bold")
    ax.text(x,.13,sub,ha="center",fontsize=9,color="#4b5563")
    if i<4:ax.annotate("",xy=(x+1.65,.45),xytext=(x+.8,.45),arrowprops=dict(arrowstyle="->",color="#2874a6",lw=2))
ax.set_xlim(-1.25,10.5);ax.set_ylim(-.5,1.2);ax.axis("off")
fig.savefig(OUT/"integration_pipeline.png",dpi=200);plt.close(fig)
