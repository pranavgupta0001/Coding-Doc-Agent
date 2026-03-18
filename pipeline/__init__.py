# pipeline package
# Installs a meta-path finder so submodule pycs are loaded directly
# without Python ever needing to find/compile stub .py files.
import marshal, sys, types, importlib.abc, importlib.machinery
from pathlib import Path

_PIPELINE_DIR = Path(__file__).resolve().parent
_PYC_DIR = _PIPELINE_DIR / '__pycache__'
# Modules that have real .py source files (handled by normal importer)
_HAS_SOURCE = {'categorizer', '__init__'}


class _PycLoader(importlib.abc.Loader):
    def __init__(self, name, pyc_path):
        self.name = name
        self.pyc_path = pyc_path

    def create_module(self, spec):
        return None  # use default ModuleType

    def exec_module(self, module):
        data = Path(self.pyc_path).read_bytes()
        code = marshal.loads(data[16:])  # skip 16-byte pyc header
        module.__file__ = str(self.pyc_path)
        exec(code, module.__dict__)


class _PipelineFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith('pipeline.'):
            return None
        submod = fullname[len('pipeline.'):]
        if '.' in submod or submod in _HAS_SOURCE:
            return None  # let normal importer handle it
        pyc = _PYC_DIR / f'{submod}.cpython-313.pyc'
        if not pyc.exists():
            return None
        loader = _PycLoader(fullname, pyc)
        return importlib.machinery.ModuleSpec(fullname, loader, origin=str(pyc))


if not any(isinstance(f, _PipelineFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _PipelineFinder())
