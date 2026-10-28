"""Original real-five-update/resource gates for exact source7201 driveD100.

The source8192 controller-transfer flag is replaced with the capacity count
flag. The original two-count verifier, saved Adam, alignment, physics and
resource gates are retained, including both4096 and8192 measurements.
"""
from pathlib import Path

import check_plain27_wheelD075_environment_capacity as original
import joint_pd_wheel005_source7201_D100_capacity_runtime as backend
import plain27_sampler_metadata_adapter as adapter


SCRIPT = Path(__file__).resolve()
retained = adapter.private_module(original.retained.SCRIPT,
    '_source8192_D100_original_capacity_command')
retained.backend = backend
SOURCE_METADATA = adapter.retarget(retained, ('probe_command',),
    comparisons={'probe_command': [("request['source_num_envs']", 4096, 8192)]},
    exact_strings={'Requalify the declared original4096-to8192 prototype':
                   'Requalify the reviewed8192 source and declared drive transfer'})
original.original.probe_command = retained.probe_command
original.original.BACKEND = backend.SCRIPT
original.original.SCRIPT = SCRIPT
require_capacity_selection = original.original.require_capacity_selection
main = original.original.main


if __name__ == '__main__':
    main()
