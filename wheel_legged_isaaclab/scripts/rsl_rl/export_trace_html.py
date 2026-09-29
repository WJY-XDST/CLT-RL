# Copyright (c) 2026, Wheel-Legged-Gym Isaac Lab Migration
# SPDX-License-Identifier: BSD-3-Clause

'''Export a replay diagnostic CSV produced by play.py to self-contained HTML.'''

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def _number(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', type=Path, help='Trace CSV created by play.py --trace_csv.')
    parser.add_argument('--output', type=Path, default=None, help='Output HTML path.')
    args = parser.parse_args()

    with args.csv_path.expanduser().open(newline='', encoding='utf-8') as stream:
        rows = [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError(f'Trace contains no rows: {args.csv_path}')

    output = (args.output or args.csv_path.with_suffix('.html')).expanduser()
    reward_keys = [key for key in rows[0] if key.startswith('reward_')]
    stats = []
    for key in reward_keys:
        values = [float(row[key]) for row in rows]
        stats.append((key, sum(values) / len(values), min(values), max(values)))
    stat_rows = ''.join(
        f'<tr><td>{html.escape(key)}</td><td>{mean:.6g}</td><td>{low:.6g}</td><td>{high:.6g}</td></tr>'
        for key, mean, low, high in stats
    )

    template = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>CLT-RL Replay Trace</title>
<style>
body{margin:0;background:#0b1017;color:#e5edf5;font-family:Arial,sans-serif} header{padding:14px 20px;background:#111927;position:sticky;top:0} main{padding:18px} canvas{width:100%;height:980px;background:#070b10;border:1px solid #2a3545} table{border-collapse:collapse;width:100%;margin-top:20px} th,td{border:1px solid #2a3545;padding:6px 9px;text-align:right} th:first-child,td:first-child{text-align:left} th{background:#162033}
</style></head><body><header><b>CLT-RL 回放诊断</b> <span id="meta"></span></header><main>
<canvas id="plot"></canvas><h2>Reward 分项统计（每控制步）</h2>
<table><thead><tr><th>分项</th><th>平均</th><th>最小</th><th>最大</th></tr></thead><tbody>__STATS__</tbody></table>
</main><script>
const rows=__ROWS__;
const groups=[
 ["水平速度",["cmd_x","vel_x_heading","vel_y_heading"]],
 ["Yaw",["cmd_yaw","yaw_rate_body"]],
 ["姿态估计",["pitch_est_rad","roll_est_rad"]],
 ["高度",["height_cmd","base_height"]],
 ["轮速",["wheel_vel_left","wheel_vel_right"]]
];
const colors=["#59a5ff","#ff7b72","#7ee787","#d2a8ff","#f2cc60"];
const canvas=document.getElementById("plot"),ctx=canvas.getContext("2d");
function resize(){canvas.width=canvas.clientWidth*devicePixelRatio;canvas.height=canvas.clientHeight*devicePixelRatio;draw()}
function draw(){const w=canvas.width,h=canvas.height,pad=55*devicePixelRatio,gap=28*devicePixelRatio;ctx.clearRect(0,0,w,h);const ch=(h-pad*2-gap*(groups.length-1))/groups.length;groups.forEach((g,i)=>chart(g[0],g[1],pad,pad+i*(ch+gap),w-pad*2,ch));}
function chart(title,keys,x,y,w,h){let vals=[];for(const r of rows)for(const k of keys)if(Number.isFinite(r[k]))vals.push(r[k]);if(!vals.length)return;let lo=Math.min(...vals),hi=Math.max(...vals);if(hi-lo<1e-9){lo-=1;hi+=1}const margin=.08*(hi-lo);lo-=margin;hi+=margin;const sx=i=>x+i/Math.max(rows.length-1,1)*w,sy=v=>y+(hi-v)/(hi-lo)*h;ctx.strokeStyle="#344154";ctx.strokeRect(x,y,w,h);ctx.fillStyle="#dce5ef";ctx.font=`${13*devicePixelRatio}px Arial`;ctx.fillText(`${title}  [${lo.toFixed(3)}, ${hi.toFixed(3)}]`,x+8*devicePixelRatio,y+17*devicePixelRatio);keys.forEach((k,j)=>{ctx.strokeStyle=colors[j%colors.length];ctx.beginPath();rows.forEach((r,n)=>{const px=sx(n),py=sy(r[k]);n?ctx.lineTo(px,py):ctx.moveTo(px,py)});ctx.stroke();ctx.fillStyle=colors[j%colors.length];ctx.fillText(k,x+(150*j+8)*devicePixelRatio,y+36*devicePixelRatio)});}
document.getElementById("meta").textContent=`rows=${rows.length}, duration=${rows[rows.length-1].sim_time_s.toFixed(2)}s`;
window.addEventListener("resize",resize);resize();
</script></body></html>'''
    rendered = template.replace('__ROWS__', json.dumps(rows, ensure_ascii=False)).replace('__STATS__', stat_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding='utf-8')
    print(output.resolve())


if __name__ == '__main__':
    main()
