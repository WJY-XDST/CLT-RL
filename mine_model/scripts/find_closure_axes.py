"""Recover the four user-specified hinge axes from circular STL cap boundaries.

Coordinates are original CAD millimetres. Faces normal to X contain closed hole
rings; fit each ring in YZ, then match coaxial holes between the specified parts.
"""
import numpy as np,json,collections
from pathlib import Path
root=Path(__file__).resolve().parents[1]
(root/'config').mkdir(exist_ok=True)
parts={p['part_id']:p for p in json.loads((root/'source/parts.json').read_text())['parts']}
results={}
for pid in ['part_159','part_156','part_160','part_154','part_370','part_410','part_369','part_411']:
 p=parts[pid]; raw=(root/'source'/p['mesh_file']).read_bytes(); n=int.from_bytes(raw[80:84],'little'); f=np.frombuffer(raw, dtype=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')]),count=n,offset=84)
 vertices=np.round(f['v'].astype(float),5);groups=collections.defaultdict(list)
 for tri in vertices:
  if np.ptp(tri[:,0])<1e-4:groups[round(float(tri[:,0].mean()),4)].append(tri)
 circles=[]
 for x,tris in groups.items():
  edges=collections.Counter()
  for tri in tris:
   vs=[tuple(v[1:]) for v in tri]
   for i in range(3):edges[tuple(sorted([vs[i],vs[(i+1)%3]]))]+=1
  adj=collections.defaultdict(set)
  for (a,b),num in edges.items():
   if num==1:adj[a].add(b);adj[b].add(a)
  while adj:
   pending=[next(iter(adj))];component=set()
   while pending:
    v=pending.pop()
    if v in component:continue
    component.add(v);pending.extend(adj.pop(v,[]))
   if len(component)<8:continue
   pts=np.array(list(component));A=np.column_stack([2*pts,np.ones(len(pts))]);u=np.linalg.lstsq(A,(pts*pts).sum(axis=1),rcond=None)[0];center=u[:2];r=np.sqrt(u[2]+(center*center).sum());err=np.max(np.abs(np.linalg.norm(pts-center,axis=1)-r))
   if err<.015 and 1<r<60:
    circles.append({'x':x,'y':round(center[0],6),'z':round(center[1],6),'r':round(r,6),'fit_max_error_mm':round(err,8),'vertices':len(pts)})
 results[pid]=circles
(root/'config/hole_candidates.json').write_text(json.dumps(results,indent=2)+'\n')

connections = [
 ('RL3_RS_closure', 'RL3_link', 'RS_link', 'part_159', 'part_156', -1),
 ('RL2_RB_closure', 'RL2_link', 'RB_link', 'part_160', 'part_154', -1),
 ('LL3_LS_closure', 'LL3_link', 'LS_link', 'part_370', 'part_410', 1),
 ('LL2_LB_closure', 'LL2_link', 'LB_link', 'part_369', 'part_411', 1),
]
closures=[]
for name, body0, body1, part0, part1, direction in connections:
 matches=[]
 for a in results[part0]:
  for b in results[part1]:
   if np.linalg.norm(np.array([a['y']-b['y'],a['z']-b['z']]))<0.001:
    matches.append((a,b))
 if not matches:
  raise RuntimeError(f'No matching hole axis: {name}')
 centers=np.array([[(a['y']+b['y'])/2,(a['z']+b['z'])/2] for a,b in matches])
 if np.ptp(centers,axis=0).max()>0.001:
  raise RuntimeError(f'Ambiguous matching axes: {name}')
 # The axial coordinate can be anywhere along a hinge line. Choose the midpoint
 # of the nearest plate faces so the inspector marker lies near their interface.
 bounds0=[parts[part0]['bbox'][k][0] for k in ('min','max')]
 bounds1=[parts[part1]['bbox'][k][0] for k in ('min','max')]
 x0,x1=min(((a,b) for a in bounds0 for b in bounds1),key=lambda v:abs(v[0]-v[1]))
 center=centers.mean(axis=0)
 mismatch=max(np.linalg.norm(np.array([a['y']-b['y'],a['z']-b['z']])) for a,b in matches)
 joint={'name':name,'body0':body0,'body1':body1,
        'anchor_cad_mm':[round((x0+x1)/2,8),round(float(center[0]),8),round(float(center[1]),8)],
        'axis_cad':[direction,0,0],
        'source_parts':[part0,part1], 'max_hole_axis_mismatch_mm':float(mismatch)}
 closures.append(joint)
 print(name, joint['anchor_cad_mm'], 'coaxial mismatch mm:', mismatch)
(root/'config/closures.json').write_text(json.dumps({'joints':closures},indent=2)+'\n')
