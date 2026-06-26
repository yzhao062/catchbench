"""Dev-seed bridge: put the local ``grade`` and ``auditable`` checkouts on ``sys.path``.

The POST seed reuses GRADE's verified Who&When loader and localization eval, and computes the
``auditable`` baseline through ``auditable``'s own public kernel. A packaged build will depend
on ``auditable`` as a normal dependency and vendor the GRADE loader; until then this bridge
imports them in place. Importing this module is a side effect (it edits ``sys.path``).
"""
import os
import sys

_GRADE = os.environ.get("GRADE_DIR", r"C:/Users/yuezh/PycharmProjects/grade")
_AUDITABLE = os.environ.get("AUDITABLE_DIR", r"C:/Users/yuezh/PycharmProjects/auditable")

_paths = [os.path.join(_GRADE, "experiment"), os.path.join(_GRADE, "src"),
          os.path.join(_AUDITABLE, "src")]
_missing = [p for p in _paths if not os.path.isdir(p)]
if _missing:
    raise ImportError(
        "auditablebench dev-seed needs local GRADE and auditable checkouts on sys.path, but these "
        f"directories are missing: {_missing}. Point GRADE_DIR / AUDITABLE_DIR at your clones of "
        "github.com/yzhao062/grade and github.com/yzhao062/auditable. A packaged build will vendor "
        "the GRADE loaders and depend on auditable as a normal dependency."
    )
for _path in _paths:
    if _path not in sys.path:
        sys.path.insert(0, _path)

# GRADE's localization eval uses a conda BLAS stack that wants this set (matches the seed).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
