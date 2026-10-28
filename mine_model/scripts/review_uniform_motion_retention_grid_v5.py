"""Retain an explicitly recorded older525 physical grid across later curricula.

The source policy's training amplitude and a regression grid's fixed commands
are distinct metadata. Only the qualifier's expected command-axis amplitude is
bound to the verified retained grid. All525 cells, numerical scoring, physical
contracts and initial27 pairing remain unchanged from v4.
"""
import argparse
import ast
import copy
import inspect
import math
from pathlib import Path

import review_uniform_motion_retention_grid_v4 as retained


SCRIPT = Path(__file__).resolve()


def qualify(args):
    import joint_pd_uniform_motion_envelope_runtime_v4 as runtime
    from joint_pd_continuation import sha
    runtime.configure(args.recipe)
    source, _, cfg = runtime.baseline()
    proof = retained.original.read(args.retained_grid_proof)
    assert proof['complete'] and proof['candidate_checkpoint_sha256'] == sha(source)
    assert all(sha(path) == digest for path, digest in proof['source_pins'].items())
    assert proof['paired525_summary']['trials'] == 525
    assert {Path(row['candidate_directory']).resolve() for row in proof['comparisons']} == {
        path.resolve() for path in args.source}
    physical_scale = proof['retained_source_motion_scale']
    assert 0. < physical_scale <= cfg['commands']['training_motion_scale']
    rows = proof['all525_pairs']
    assert math.isclose(max(abs(row['case']['speed']) for row in rows),
        max(abs(v) for v in cfg['commands']['ranges_lin_vel_x']) * physical_scale, abs_tol=1.e-12)
    assert math.isclose(max(abs(row['case']['yaw']) for row in rows),
        max(abs(v) for v in cfg['commands']['ranges_ang_vel_yaw']) * physical_scale, abs_tol=1.e-12)
    original = retained.original
    tree = ast.parse(inspect.getsource(original.qualify))
    before = ast.dump(tree)
    expected = ast.dump(ast.parse("cfg['commands']['training_motion_scale']", mode='eval').body)
    points = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
              and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id == 'scale' and ast.dump(node.value) == expected]
    assert len(points) == 1
    prior = points[0].value
    points[0].value = ast.Name(id='_verified_retained_physical_scale', ctx=ast.Load())
    qualified = copy.deepcopy(tree)
    points[0].value = prior
    assert ast.dump(tree) == before
    namespace = dict(original.__dict__, _verified_retained_physical_scale=physical_scale)
    exec(compile(ast.fix_missing_locations(qualified), str(SCRIPT), 'exec'), namespace)
    previous, previous_script = original.qualify, retained.SCRIPT
    original.qualify, retained.SCRIPT = namespace['qualify'], SCRIPT
    try:
        retained.qualify(args)
    finally:
        original.qualify, retained.SCRIPT = previous, previous_script
    path = args.output / 'qualification.json'
    spec = original.read(path)
    spec['retention_scope_motion_scale'] = physical_scale
    spec['source_policy_training_motion_scale'] = cfg['commands']['training_motion_scale']
    spec['retained_grid_source_proof'] = str(args.retained_grid_proof.resolve())
    spec['all_retained_boundary_and_interior_cartesian_cells_included'] = spec.pop(
        'all_current_stage_boundary_and_interior_cartesian_cells_included')
    spec['only_qualifier_axis_metadata_binding_changed'] = True
    spec['original_qualifier_AST_restored_identical'] = True
    spec['source_pins'].update({str(args.retained_grid_proof.resolve()): sha(args.retained_grid_proof),
        str(SCRIPT): sha(SCRIPT), str(Path(retained.__file__).resolve()): sha(retained.__file__)})
    import json
    path.write_text(json.dumps(spec, indent=2) + '\n')
    return spec


def run(output):
    previous_script = retained.SCRIPT
    retained.SCRIPT = SCRIPT
    try:
        return retained.run(output)
    finally:
        retained.SCRIPT = previous_script


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualify-only', action='store_true')
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--pointer', type=Path)
    parser.add_argument('--source', type=Path, nargs='+')
    parser.add_argument('--after-owned-unit')
    parser.add_argument('--retained-grid-proof', type=Path)
    args = parser.parse_args()
    if args.qualify_only:
        assert all((args.recipe, args.pointer, args.source, args.after_owned_unit, args.retained_grid_proof))
        qualify(args)
    else:
        run(args.output)
