"""Retain original actual-five-Adam, alignment and resource gates at both counts."""
from pathlib import Path

import check_plain27_wheel005_D100_yaw400_environment_capacity as original
import joint_pd_source7401_D100_C400_focusgrid_capacity_runtime as backend
import capacity_passive_resource_profile as passive_profile


SCRIPT = Path(__file__).resolve()
original.retained.retained.backend = backend
original.numeric.BACKEND, original.numeric.SCRIPT = backend.SCRIPT, SCRIPT
passive_profile.install(original.numeric)
require_capacity_selection, main = original.numeric.require_capacity_selection, original.numeric.main


if __name__ == "__main__":
    main()
