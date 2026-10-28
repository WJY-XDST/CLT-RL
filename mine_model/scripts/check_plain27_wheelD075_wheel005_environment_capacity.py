"""Keep the original five-update Adam/alignment/resource gates for wheel005."""
from pathlib import Path

import check_plain27_wheelD075_environment_capacity as original
import joint_pd_AF7001_wheelD075_wheel005_capacity_runtime as backend

SCRIPT=Path(__file__).resolve()
original.retained.backend=backend
original.original.BACKEND=backend.SCRIPT
original.original.SCRIPT=SCRIPT
original.original.probe_command=original.retained.probe_command
require_capacity_selection=original.original.require_capacity_selection
main=original.original.main


if __name__=='__main__':main()
