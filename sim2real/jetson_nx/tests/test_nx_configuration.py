import ast
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from wheelleg_deploy.configuration import run_options,mcu_initializer,mcu_settings
from wheelleg_deploy.numpy_policy import NumpyPolicy

ROOT=Path(__file__).resolve().parents[1]


def arguments(**values):
    result=dict(config=None,bundle=None,port=None,baudrate=None,control_hz=None,
                state_timeout_s=None,duration=None,log=None,motion_command=None,execute=False)
    result.update(values)
    return SimpleNamespace(**result)


class ConfigurationTests(unittest.TestCase):
    def test_relative_paths_cli_overrides_and_explicit_port(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"nx.json"
            path.write_text(json.dumps(dict(bundle="../bundle",port=None,baudrate=2000000)))
            with self.assertRaisesRegex(ValueError,"Explicit"):
                run_options(arguments(config=path))
            result=run_options(arguments(config=path,port="/dev/fake",control_hz=500))
            self.assertEqual(result["bundle"],path.parent/"../bundle")
            self.assertEqual(result["control_hz"],500)
            self.assertEqual(result["state_timeout_s"],.03)
            self.assertFalse(result["execute"])
            path.write_text(json.dumps(dict(bundle="bundle",port="p",execute=True)))
            with self.assertRaisesRegex(ValueError,"Unknown"):
                run_options(arguments(config=path))

    def test_python38_syntax(self):
        for path in (ROOT/"wheelleg_deploy").glob("*.py"):
            ast.parse(path.read_text(),filename=str(path),feature_version=(3,8))

    @unittest.skipUnless((ROOT/"bundles/selected_nx_policy/manifest.json").exists(),"Selected bundle required")
    def test_generated_mcu_settings_and_c_initializer_match(self):
        policy=NumpyPolicy(ROOT/"bundles/selected_nx_policy")
        cfg=mcu_settings(policy)
        self.assertEqual(cfg["contract_sha256"],policy.metadata["contract_sha256"])
        for value,expected in zip(cfg["torque_limit_nm"],[35.574,35.574,4.80944,35.574,35.574,4.80944]):
            self.assertAlmostEqual(value,expected,places=12)
        self.assertEqual(cfg["command_min"][0],-.8)
        self.assertEqual(cfg["command_max"][0],.8)
        self.assertIn("30000u, 30000u",mcu_initializer(policy))
        header=ROOT.parent/"series_balance_rtos/control"
        if not header.exists():
            header=ROOT/"mcu_reference"
        if shutil.which("gcc") and (header/"rl_host.h").exists():
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/"config.c"
                path.write_text('#include "rl_host.h"\n'+mcu_initializer(policy)+
                                '\nint main(void){RLHost_Context c;return !RLHost_Init(&c,&rl_host_config);}\n')
                binary=Path(temp)/"config_test"
                subprocess.run(["gcc","-std=c99","-Wall","-Wextra","-Werror",
                                "-DRL_HOST_PORTABLE_TEST","-I"+str(header),str(path),
                                str(header/"rl_host.c"),"-lm","-o",str(binary)],
                               check=True,capture_output=True)
                subprocess.run([str(binary)],check=True,capture_output=True)


if __name__=="__main__":
    unittest.main()
