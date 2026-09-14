"""Exercise protection on temporary fixtures, never historical experiments."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from check_version_integrity import verify_manifest,permitted_edit

class Protection(unittest.TestCase):
    def test_frozen_changes(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); f=p/'config.json'; f.write_bytes(b'original')
            (p/'manifest_test.json').write_text(json.dumps({'frozen':True,'sha256':{'config.json':hashlib.sha256(f.read_bytes()).hexdigest()}}))
            self.assertEqual(verify_manifest(p),[])
            f.write_bytes(b'changed'); self.assertTrue(verify_manifest(p))
            f.write_bytes(b'original'); (p/'extra.py').write_text('x=1')
            self.assertTrue(verify_manifest(p))
    def test_append_only(self):
        self.assertTrue(permitted_edit('VERSION_HISTORY.md',b'old\n',b'old\nnew\n'))
        self.assertFalse(permitted_edit('VERSION_HISTORY.md',b'old\n',b'new\n'))
        self.assertFalse(permitted_edit('config.json',b'old',b'oldnew'))

if __name__=='__main__': unittest.main()
