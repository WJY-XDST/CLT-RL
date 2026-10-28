"""No-UART compute/scheduler probe to run on the actual Jetson device."""
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

from .numpy_policy import NumpyPolicy
from .nx_runtime import validate_timing,warmup
from .torque_control import TorqueController


def statistics(values):
    array=np.asarray(values,dtype=np.float64)*1000.
    return dict(count=len(values),mean_ms=float(array.mean()),p50_ms=float(np.percentile(array,50)),
                p95_ms=float(np.percentile(array,95)),p99_ms=float(np.percentile(array,99)),
                max_ms=float(array.max()))


def benchmark(bundle,*,duration_s=30.,control_hz=200,output=None):
    validate_timing(control_hz,.03)
    if isinstance(duration_s,bool) or not isinstance(duration_s,(int,float)) or not math.isfinite(duration_s) or duration_s<=0:
        raise ValueError("Benchmark duration must be finite and positive")
    if duration_s<1/control_hz:
        raise ValueError("Benchmark duration must cover at least one control period")
    policy=NumpyPolicy(bundle)
    warmup(policy)
    controller=TorqueController(policy)
    obs=[0.]*27
    obs[5]=-1.
    obs[8]=sum(policy.cfg["commands"]["ranges_height"])/2*policy.cfg["obs_scales"]["height_measurements"]
    raw=policy.cfg["leg_joint_position_offset"]+[0.]*6
    # Warm gravity/NumPy as well as Actor before timed samples.
    controller.step(obs,raw,infer=True)
    compute,inference,servo,lateness=[],[],[],[]
    period=round(1e9/control_hz)
    policy_period=round(policy.metadata["policy_period_s"]*1e9)
    started=time.perf_counter_ns()
    next_tick,next_inference=started,started
    skipped_slots=0
    while time.perf_counter_ns()-started<duration_s*1e9:
        now=time.perf_counter_ns()
        if now<next_tick:
            time.sleep((next_tick-now)/1e9)
        before=time.perf_counter_ns()
        lateness.append(max(0,before-next_tick)/1e9)
        infer=before>=next_inference
        if infer:
            next_inference+=max(1,(before-next_inference)//policy_period+1)*policy_period
        controller.step(obs,raw,infer=infer)
        after=time.perf_counter_ns()
        elapsed=(after-before)/1e9
        compute.append(elapsed)
        (inference if infer else servo).append(elapsed)
        next_tick+=period
        if after>next_tick:
            missed=(after-next_tick)//period+1
            skipped_slots+=missed
            next_tick+=missed*period
    model_path=Path("/proc/device-tree/model")
    model=model_path.read_bytes().rstrip(b"\x00").decode(errors="replace") if model_path.is_file() else None
    result=dict(format="clt_nx_no_uart_benchmark_v1",platform=platform.platform(),machine=platform.machine(),
                device_model=model,python=sys.version.split()[0],numpy=np.__version__,
                contract_sha256=policy.metadata["contract_sha256"],
                requested_control_hz=control_hz,policy_hz=50,duration_s=(time.perf_counter_ns()-started)/1e9,
                samples=len(compute),inference_samples=len(inference),
                compute=statistics(compute),actor_plus_pd=statistics(inference),
                pd_only=statistics(servo) if servo else None,wakeup_lateness=statistics(lateness),
                compute_deadline_misses=sum(value>1/control_hz for value in compute),
                skipped_scheduler_slots=int(skipped_slots),max_process_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                blas_threads={key:os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
                input_kind="synthetic nominal sensors, not a closed-loop robot",
                serial_opened=False,real_hardware_motion_validated=False,policy_precision_accepted=False,
                actual_jetson_measurement=bool(model and "NVIDIA" in model.upper()))
    if output is not None:
        with Path(output).open("x",encoding="utf-8") as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write("\n")
    return result
