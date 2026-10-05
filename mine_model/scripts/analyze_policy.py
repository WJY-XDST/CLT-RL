"""Summarize pre-reset telemetry and plot actual transfer errors, including failures."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory",type=Path)
args=parser.parse_args()
root=args.directory
data={}
summary={}
labels=["Stand", "Forward 0.3 m/s", "Forward 0.6 m/s", "Reverse 0.3 m/s", "Reverse 0.6 m/s", "Spin +0.3 rad/s", "Spin -0.3 rad/s", "Low stance", "High stance"]
for robot in ("mine","legacy"):
    path=root/robot
    assert json.loads((path/"completed.json").read_text())["completed"]
    a=np.genfromtxt(path/"trace.csv",delimiter=",",names=True)
    data[robot]=a
    manifest=json.loads((path/"manifest.json").read_text())
    summary[robot]={"checkpoint_sha256":manifest["checkpoint_sha256"],"conditions":[]}
    for cid,case in enumerate(manifest["cases"]):
        d=a[a["case_id"]==cid]
        first=d[d["episode"]==0]
        failures=first[first["terminated"]>0]
        # Steady-state metrics only exist when the FIRST episode survives.
        stable=first[first["time_s"]>=manifest["seconds_per_case"]-2]
        record=dict(case)
        record.update(resets=int(d["terminated"].sum()),survivors=int(3-len(failures)),
            first_failure_s=failures["time_s"].tolist(),
            all_sample_speed_mae_m_s=float(np.mean(abs(d["speed_m_s"]-d["speed_cmd_m_s"]))),
            all_sample_height_mae_mm=float(np.mean(abs(d["height_m"]-d["height_cmd_m"]))*1000),
            peak_body_angle_deg=float(np.max(np.abs(np.stack((d["pitch_rad"],d["roll_rad"]))))*180/np.pi),
            mean_torque_saturation_fraction=float(d["torque_saturation_fraction"].mean()),
            mean_action_clipping_fraction=float(d["action_clipping_fraction"].mean()),
            max_applied_speed_command_m_s=float(np.max(abs(d["speed_cmd_m_s"]))),
            first_episode_speed_mae_m_s=float(np.mean(abs(first["speed_m_s"]-first["speed_cmd_m_s"]))),
            first_episode_height_mae_mm=float(np.mean(abs(first["height_m"]-first["height_cmd_m"]))*1000),
            steady_speed_mae_m_s=float(np.mean(abs(stable["speed_m_s"]-stable["speed_cmd_m_s"]))) if len(stable) else None,
            steady_height_mae_mm=float(np.mean(abs(stable["height_m"]-stable["height_cmd_m"]))*1000) if len(stable) else None,
            reasons={k[5:]:int(d[k].sum()) for k in a.dtype.names if k.startswith("fail_")})
        summary[robot]["conditions"].append(record)
    summary[robot]["total_resets"]=int(a["terminated"].sum())
    summary[robot]["survivors"]=sum(r["survivors"] for r in summary[robot]["conditions"])
assert summary["mine"]["checkpoint_sha256"]==summary["legacy"]["checkpoint_sha256"]
summary["interpretation"]="Unchanged model_7738 actor; robot-specific VMC. Heights 0.28/0.30/0.32 m on mine, 0.16/0.18/0.20 m on legacy. All-sample MAE includes repeated failed episodes; it is not steady-state tracking quality. Capture is before reset."
(root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
with (root/"summary.csv").open("w",newline="") as f:
    keys=["robot","name","speed","yaw","height","resets","survivors","first_failure_s","all_sample_speed_mae_m_s","all_sample_height_mae_mm","peak_body_angle_deg","max_applied_speed_command_m_s","steady_speed_mae_m_s","steady_height_mae_mm"]
    w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore");w.writeheader()
    for robot in ("mine","legacy"):
        for row in summary[robot]["conditions"]:w.writerow(dict(robot=robot,**row))

plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
def sample(a,cid,seed=43,first=False):
    return a[(a["case_id"]==cid)&(a["seed"]==seed)&((a["episode"]==0) if first else True)]
def broken(d,values):
    # Insert a NaN between episodes but retain the physical failure sample.
    t=[];v=[]
    for i,(x,y) in enumerate(zip(d["time_s"],values)):
        if i and d["episode"][i]!=d["episode"][i-1]:t.append(np.nan);v.append(np.nan)
        t.append(x);v.append(y)
    return t,v

with PdfPages(root/"error_curves.pdf") as pdf:
    for group,cids in [("straight",[0,1,3]),("fast_and_turn",[2,4,5]),("turn_and_height",[6,7,8])]:
        fig,axs=plt.subplots(4,3,figsize=(14,11),sharex=True,layout="constrained")
        for col,cid in enumerate(cids):
            for robot,color in [("legacy","#2471a3"),("mine","#c0392b")]:
                d=sample(data[robot],cid)
                series=[d["speed_m_s"]-d["speed_cmd_m_s"],1000*(d["height_m"]-d["height_cmd_m"]),
                    (d["yaw_rate_rad_s"]-d["yaw_cmd_rad_s"]),np.maximum(abs(d["pitch_rad"]),abs(d["roll_rad"]))*180/np.pi]
                for row,y in enumerate(series):axs[row,col].plot(*broken(d,y),color=color,lw=1.,label=robot)
                if robot=="mine":
                    for when in d["time_s"][d["terminated"]>0]:
                        for ax in axs[:,col]:ax.axvline(when,color=color,alpha=.20,lw=.6)
            axs[0,col].set_title(labels[cid])
            axs[-1,col].set_xlabel("Simulation time (s)")
        for row,label in enumerate(["Speed error (m/s)","Height error (mm)","Yaw rate error (rad/s)","Max |pitch, roll| (deg)"]):
            axs[row,0].set_ylabel(label)
            for ax in axs[row]:ax.grid(alpha=.18);ax.axhline(0,color="black",lw=.5);ax.set_xlim(0,8)
        axs[0,0].legend()
        fig.suptitle("Unchanged model_7738 on flat ground | seed 43\nRed: new closed-chain robot; blue: original robot. Vertical red lines: resets. Error = actual - applied reference.",fontsize=12)
        fig.savefig(root/f"errors_{group}.png",dpi=160);pdf.savefig(fig);plt.close(fig)

fig,axs=plt.subplots(3,3,figsize=(13,8),sharex=True,layout="constrained")
for col,cid in enumerate([0,1,3]):
    for seed,alpha in [(43,1.),(44,.55),(45,.35)]:
        d=sample(data["mine"],cid,seed,True)
        for row,y in enumerate([d["speed_m_s"]-d["speed_cmd_m_s"],1000*(d["height_m"]-d["height_cmd_m"]),np.maximum(abs(d["pitch_rad"]),abs(d["roll_rad"]))*180/np.pi]):
            axs[row,col].plot(d["time_s"],y,color="#c0392b",alpha=alpha,label=f"seed {seed}")
            if d["terminated"][-1]:axs[row,col].plot(d["time_s"][-1],y[-1],"x",color="#922b21")
    axs[0,col].set_title(labels[cid]);axs[2,col].set_xlabel("Time before first reset (s)")
for row,label in enumerate(["Speed error (m/s)","Height error (mm)","Max |pitch, roll| (deg)"]):
    axs[row,0].set_ylabel(label)
    for ax in axs[row]:ax.grid(alpha=.2);ax.axhline(0,color="black",lw=.5)
axs[0,0].legend(fontsize=8)
fig.suptitle("New robot startup: all three seeds | crosses mark first termination",fontsize=13)
fig.savefig(root/"startup_errors.png",dpi=180);plt.close(fig)

fig,axs=plt.subplots(3,1,figsize=(12,8),sharex=True,layout="constrained")
d=sample(data["mine"],0,43)
for side,color in [("left","#2471a3"),("right","#b9770e")]:
    axs[0].plot(*broken(d,(d[f"{side}_length_m"]-d[f"{side}_length_ref_m"])*1000),color=color,label=side)
    axs[1].plot(*broken(d,(d[f"{side}_angle_rad"]-d[f"{side}_angle_ref_rad"])*180/np.pi),color=color,label=side)
axs[2].plot(d["time_s"],d["torque_saturation_fraction"]*100,label="Actuators at torque limit")
axs[2].plot(d["time_s"],d["action_clipping_fraction"]*100,label="Clipped policy actions",alpha=.7)
for ax,label in zip(axs,["Leg length error (mm)","Leg angle error (deg)","Fraction (%)"]):
    ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(loc="upper right")
    for when in d["time_s"][d["terminated"]>0]:ax.axvline(when,color="red",lw=.5,alpha=.3)
axs[-1].set_xlabel("Simulation time (s)")
fig.suptitle("New robot standing command | model_7738 | seed 43 | pre-reset telemetry")
fig.savefig(root/"leg_and_saturation_errors.png",dpi=160);plt.close(fig)
print(json.dumps({k:{"survivors":summary[k]["survivors"],"resets":summary[k]["total_resets"]} for k in data},indent=2))
