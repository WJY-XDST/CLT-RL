"""Verify and snapshot the complete declared CAD collision asset chain."""
import hashlib
import json
from pathlib import Path


def verify_asset_chain(asset):
    hashes, manifests = {}, {}

    def visit(path):
        path = Path(path).resolve()
        key = str(path)
        if key in hashes:
            return
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = path.with_suffix('.json')
        if not manifest.is_file():
            return
        metadata = json.loads(manifest.read_text())
        expected = dict(metadata.get('sha256', {}))
        if metadata.get('overlay_sha256'):
            expected[str(path)] = metadata['overlay_sha256']
            expected[metadata['source']] = metadata['source_sha256']
        if not expected:
            return
        manifests[str(manifest)] = metadata
        for dependency, digest in expected.items():
            dependency = Path(dependency)
            if not dependency.is_absolute():
                dependency = manifest.parent / dependency
            if hashlib.sha256(dependency.read_bytes()).hexdigest() != digest:
                raise RuntimeError('Validated collision asset changed: ' + str(dependency))
            visit(dependency)

    visit(asset)
    return hashes, manifests
