from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'mine_model/scripts'))
import audit_joint_pd_native_training as audit


@pytest.mark.parametrize('fail',[False,True])
def test_runpy_preserves_script_local_imports_argv_and_restores_caller(monkeypatch,fail):
    original_path=list(sys.path)
    original_argv=sys.argv
    def fake_run(path,run_name):
        assert run_name=='__main__'
        assert Path(sys.path[0])==Path(path).parent
        assert (Path(sys.path[0])/'cli_args.py').is_file()
        assert sys.argv==[path,'--task','WheelLeggedJointPD-Flat-v0','--headless']
        if fail:
            raise RuntimeError('fake simulation exit')
        return {'finished':True}
    monkeypatch.setattr(audit.runpy,'run_path',fake_run)
    if fail:
        with pytest.raises(RuntimeError,match='fake simulation'):
            audit.run_original_training(['--task','WheelLeggedJointPD-Flat-v0','--headless'])
    else:
        assert audit.run_original_training(['--task','WheelLeggedJointPD-Flat-v0','--headless'])=={'finished':True}
    assert sys.path==original_path
    assert sys.argv is original_argv
