"""A shared collider or underlying CAD edit must invalidate acceptance."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wheel_legged_gym_isaaclab.asset_integrity import verify_asset_chain


class AssetIntegrityTests(unittest.TestCase):
    def test_nested_source_and_shared_geometry_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, shared, overlay = [root / n for n in ('cad.usd', 'shared.usd', 'overlay.usd')]
            for path in (source, shared, overlay):
                path.write_bytes(path.name.encode())
            def digest(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()
            source.with_suffix('.json').write_text(json.dumps({
                'sha256': {str(shared): digest(shared), str(source): digest(source)}
            }))
            overlay.with_suffix('.json').write_text(json.dumps({
                'sha256': {str(source): digest(source), str(overlay): digest(overlay)}
            }))
            hashes, _ = verify_asset_chain(overlay)
            self.assertEqual(set(hashes), {str(p) for p in (overlay, source, shared)})
            shared.write_bytes(b'changed contact surface')
            with self.assertRaisesRegex(RuntimeError, 'asset changed'):
                verify_asset_chain(overlay)


if __name__ == '__main__':
    unittest.main()
