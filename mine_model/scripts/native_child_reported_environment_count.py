"""Bind only legacy child reporter population metadata to its actual native CLI.

The original subprocess lifetime, anomaly checks, logs and command stay
unchanged. This module is prepared for future entries; it never injects
code into a running training process.
"""
import ast
import copy
import inspect
from pathlib import Path
SCRIPT=Path(__file__).resolve()

def declared_training_num_envs(command):
    if command.count('--num_envs')!=1:
        raise ValueError('Require one native population option before reporting its count')
    value=int(command[command.index('--num_envs')+1])
    if value not in (4096,8192):
        raise ValueError('Report only a qualified native training population')
    return value

def install(api):
    tree=ast.parse(inspect.getsource(api.run_child));before=ast.dump(tree)
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.keyword) and n.arg=='num_envs']
    if len(nodes)!=1 or not isinstance(nodes[0].value,ast.Constant) or nodes[0].value.value!=8192:
        raise ValueError('Requalify the single original hardcoded reporter count')
    old=nodes[0].value;nodes[0].value=ast.parse('declared_training_num_envs(command)',mode='eval').body
    changed=copy.deepcopy(tree);nodes[0].value=old
    if ast.dump(tree)!=before:
        raise ValueError('Original child execution or anomaly logic changed beyond report metadata')
    api.declared_training_num_envs=declared_training_num_envs
    exec(compile(ast.fix_missing_locations(changed),str(SCRIPT),'exec'),api.__dict__)
    return dict(original_AST_restored_identical=True,only_reported_num_envs_replaced=True,
        subprocess_command_untouched=True,anomaly_checks_untouched=True,training_logic_changed=False)
