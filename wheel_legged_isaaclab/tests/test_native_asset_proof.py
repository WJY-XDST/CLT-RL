import json

import pytest

from joint_pd_continuation import qualified_wheel_mass_asset
from wheel_legged_gym_isaaclab import robot_model


def test_stale_native_proof_cannot_admit_a_different_usd(tmp_path,monkeypatch):
    source=robot_model.ASSETS/robot_model.MANIFEST['wheel_mass_calibration']
    profile=json.loads(source.read_text())
    asset=tmp_path/profile['overlay']
    asset.write_bytes((source.parent/profile['overlay']).read_bytes())
    profile['validation']['physx_loading_smoke']['asset_sha256']='0'*64
    proof=tmp_path/'proof.json'
    proof.write_text(json.dumps(profile))
    monkeypatch.setattr(robot_model,'ASSETS',tmp_path)
    monkeypatch.setattr(robot_model,'MANIFEST',dict(wheel_mass_calibration=proof.name,training_usd=asset.name))
    with pytest.raises(ValueError,match='Native mass/inertia evidence does not match'):
        qualified_wheel_mass_asset(proof)
