"""Prove native wrapper exclusion using real child processes and file locks."""
import os
from pathlib import Path
import shutil
import subprocess
import time


def test_headless_native_processes_share_one_slot_but_cpu_work_can_proceed(tmp_path):
    root=Path(__file__).resolve().parents[2]
    wrapper=tmp_path/'run_python.sh'
    shutil.copyfile(root/'run_python.sh',wrapper)
    wrapper.chmod(0o755)
    fake=tmp_path/'IsaacLab/_isaac_sim/python.sh'
    fake.parent.mkdir(parents=True)
    fake.write_text('#!/bin/bash\nset -eu\nif [[ "$1" == "--headless" ]]; then shift; fi\n'
                    'touch "$1"\nwhile [[ ! -e "$2" ]]; do sleep 0.01; done\n')
    fake.chmod(0o755)
    runtime=tmp_path/'runtime';runtime.mkdir()
    environment=dict(os.environ,XDG_RUNTIME_DIR=str(runtime))
    processes=[]
    def start(name,headless):
        arguments=[str(wrapper),*(['--headless'] if headless else []),str(tmp_path/(name+'.started')),str(tmp_path/(name+'.done'))]
        process=subprocess.Popen(arguments,env=environment)
        processes.append(process)
        return process
    def until(path):
        deadline=time.monotonic()+3
        while not path.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert path.exists()
    try:
        first=start('first',True);until(tmp_path/'first.started')
        second=start('second',True)
        cpu=start('cpu',False);until(tmp_path/'cpu.started')
        assert first.poll() is None and second.poll() is None and not (tmp_path/'second.started').exists()
        (tmp_path/'first.done').touch();assert first.wait(timeout=3)==0
        until(tmp_path/'second.started')
        (tmp_path/'second.done').touch();(tmp_path/'cpu.done').touch()
        assert second.wait(timeout=3)==0 and cpu.wait(timeout=3)==0
    finally:
        for name in ('first','second','cpu'):(tmp_path/(name+'.done')).touch()
        for process in processes:
            if process.poll() is None:process.terminate()
            process.wait(timeout=3)
