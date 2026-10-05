"""Physical MuJoCo limit-block sweep using the unchanged converted CAD shapes."""
import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from core import FIVE_BAR, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-config', required=True, type=Path)
    parser.add_argument('--mjcf', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cfg = load_config(args.env_config)
    args.output.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(args.mjcf)
    root = tree.getroot()
    base = root.find("worldbody/body[@name='base_link']")
    base.remove(base.find('freejoint'))
    base.set('pos', '0 0 .6')
    root.find('option').set('gravity', '0 0 0')
    root.find('option').set('timestep', '.0001')
    test_model = args.output / 'fixed_base_stop_test.xml'
    tree.write(test_model, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(test_model))
    data = mujoco.MjData(model)
    rows, poses = [], []
    body_names = ['LB_link', 'LS_link', 'RB_link', 'RS_link']
    phases = [.12, .18, .36, .42]
    dt = .02
    torch.set_num_threads(1)
    for phase, target in enumerate(phases):
        mujoco.mj_resetData(model, data)
        for step in range(600):
            # Match the existing Isaac stop bench: 2 s initial hold, 5 s ramp,
            # .052219456 rad assembly angle, then 5 s held overstroke.
            ramp = min(max((step * dt - 2.) / 5., 0.), 1.)
            reference = .27 + ramp * (target - .27)
            length = torch.full((1, 2), reference)
            b, l, valid = FIVE_BAR.five_bar_inverse(length, torch.full((1, 2), .052219456), **cfg['five_bar_geometry'])
            if not valid.all():
                raise ValueError('Invalid inverse test reference')
            data.ctrl[:] = [b[0, 0], l[0, 0], 0, b[0, 1], l[0, 1], 0]
            for _ in range(200):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            gaps = [np.linalg.norm(data.site_xpos[model.eq_obj1id[i]] - data.site_xpos[model.eq_obj2id[i]]) for i in range(model.neq)]
            stop_contacts = [c for c in data.contact if {model.geom_contype[c.geom1], model.geom_contype[c.geom2]} == {4, 8}]
            measured = []
            for hip, wheel in [('LB_link', 'LW_link'), ('RB_link', 'RW_link')]:
                offset = data.xpos[model.body(wheel).id] - data.xpos[model.body(hip).id]
                measured.append(np.linalg.norm(offset[[0, 2]]))
            row = {'time_s': (step + 1) * dt, 'phase': phase, 'target_m': target, 'reference_m': reference,
                   'left_length_m': measured[0], 'right_length_m': measured[1],
                   'stop_contact_count': len(stop_contacts),
                   'stop_penetration_max_mm': 1000 * max(0., max((-c.dist for c in stop_contacts), default=0.)),
                   'closure_gap_max_mm': 1000 * max(gaps), 'peak_torque_nm': max(abs(data.actuator_force))}
            rows.append(row)
            poses.append([[*data.xpos[model.body(name).id], *data.xquat[model.body(name).id][[1, 2, 3, 0]]] for name in body_names])
    np.savez_compressed(args.output / 'protected_cad_poses.npz', poses=np.asarray(poses)[:, None], body_names=body_names)
    with (args.output / 'trace.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    phases_report = []
    for phase, target in enumerate(phases):
        tail = rows[phase * 600 + 450:(phase + 1) * 600]
        phases_report.append({'target_m': target,
            'left_mean_m': float(np.mean([r['left_length_m'] for r in tail])),
            'right_mean_m': float(np.mean([r['right_length_m'] for r in tail])),
            'max_stop_contact_count': max(r['stop_contact_count'] for r in tail),
            'max_stop_penetration_mm': max(r['stop_penetration_max_mm'] for r in tail)})
    report = {'asset': cfg['robot']['spawn']['usd_path'], 'mjcf': str(args.mjcf.resolve()),
              'fixed_base': True, 'gravity': 0, 'leg_stiffness': cfg['leg_joint_stiffness'],
              'leg_damping': cfg['leg_joint_damping'], 'effort_limit_nm': cfg['leg_effort_limit'],
              'phases': phases_report,
              'max_closure_gap_mm': max(r['closure_gap_max_mm'] for r in rows),
              'max_stop_penetration_mm': max(r['stop_penetration_max_mm'] for r in rows),
              'limitation': 'MuJoCo cooked convex contact separation; not an exact CAD boolean intersection test.'}
    lower, upper = phases_report[0], phases_report[3]
    report['physical_stops_block_overtravel'] = all(lower[side + '_mean_m'] > .15 and upper[side + '_mean_m'] < .39 for side in ('left', 'right')) and lower['max_stop_contact_count'] > 0 and upper['max_stop_contact_count'] > 0
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['physical_stops_block_overtravel'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
