"""Build a small, explicit NX project; never copy the full MCU or training tree."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wheelleg_deploy.numpy_policy import NumpyPolicy


def build(bundle,output):
    policy=NumpyPolicy(bundle)
    output=output.expanduser().resolve()
    if output.exists():
        raise ValueError("Package already exists; choose a new output path")
    payload={}
    modes={}
    def add(path,name=None):
        name=name or path.relative_to(ROOT).as_posix()
        payload[name]=path.read_bytes()
        modes[name]=path.stat().st_mode&0o777
    for name in ["README.md","PROTOCOL.md","pyproject.toml","requirements-nx.txt","run_nx.sh",
                 "config/nx.example.json","config/usart3_selection_20261027.json"]:
        add(ROOT/name)
    for directory,pattern in [("wheelleg_deploy","*.py"),("tests","test_*.py"),
                              ("tools","*.py"),("verification","*.json")]:
        for path in sorted((ROOT/directory).glob(pattern)):
            add(path)
    for path in sorted(bundle.iterdir()):
        if path.is_file() and path.suffix in (".npz",".json"):
            add(path,"bundles/selected_nx_policy/"+path.name)
    for name in ["rl_host.c","rl_host.h"]:
        source=ROOT.parent/"series_balance_rtos/control"
        if not source.exists():
            source=ROOT/"mcu_reference"
        add(source/name,"mcu_reference/"+name)
    # Make source links work in the runtime archive; the full board project
    # intentionally stays in the user's workstation checkout.
    for name in ("README.md","PROTOCOL.md"):
        text=payload[name].decode("utf-8").replace("../series_balance_rtos/control/","mcu_reference/")
        if name=="README.md":
            note=("本压缩包是 NX 运行包。完整 STM32 工程保留在工作站；两个接口参考文件放在 "
                  "mcu_reference/。下文其余完整下位机路径指工作站工程。\n\n")
            first,rest=text.split("\n",1)
            text=first+"\n\n"+note+rest.lstrip("\n")
        payload[name]=text.encode("utf-8")
    entries={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),mode=modes[name])
             for name,data in payload.items()}
    manifest=dict(format="clt_nx_uart_runtime_archive_v1",
                  current_use_scope="Inference, benchmark and shadow UART; motor execution not validated",
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  checkpoint_sha256=policy.metadata["checkpoint_sha256"],
                  contract_sha256=policy.metadata["contract_sha256"],
                  files=entries,hardware_validated=False,policy_precision_accepted=False)
    payload["PACKAGE_MANIFEST.json"]=(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n").encode()
    modes["PACKAGE_MANIFEST.json"]=0o644
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,data in sorted(payload.items()):
            info=zipfile.ZipInfo(name)
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(0o100000|modes[name])<<16
            archive.writestr(info,data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
        for name,entry in entries.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=entry["sha256"]:
                raise ValueError("Archive content verification failed")
    print(json.dumps(dict(package=str(output),bytes=output.stat().st_size,files=len(payload),
                          sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                          contract_sha256=manifest["contract_sha256"]),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    build(args.bundle.expanduser().resolve(),args.output)
