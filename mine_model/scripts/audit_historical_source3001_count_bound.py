"""Original five numerical readers bound to the admitted population."""
import argparse
import importlib
import json
from pathlib import Path

import plain27_sampler_metadata_adapter as adapter
from audit_parallel8192_count_bound import READERS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reader", choices=READERS, required=True)
    parser.add_argument("--through", type=int, required=True)
    args = parser.parse_args()
    count = json.loads((args.output / "experiment_plan.json").read_text())["native_num_envs"]
    assert count in (4096, 8192)
    original = importlib.import_module(READERS[args.reader])
    api = adapter.private_module(original.__file__, "_historical_count_" + args.reader)
    api.SCRIPT = Path(__file__).resolve()
    evidence = adapter.retarget(api, ("audit",), integers={"*": {4096: count}})
    result = api.audit(args.output, args.through)
    print(dict(reader=args.reader, count=count, population_metadata=evidence,
               original_numerical_reader_unchanged=True, optimizer_executed=False))
