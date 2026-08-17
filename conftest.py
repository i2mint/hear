"""pytest configuration: make the in-tree `hear` win over any installed copy.

Without this, running `pytest tests/` from a source checkout can import a
*different* `hear` than the one being tested. `tests/` is not a package, so
pytest prepends `tests/` -- not the repo root -- to `sys.path`; the name `hear`
then falls through to whatever the environment has installed (an editable
install elsewhere, or a wheel from PyPI). The tests would then silently
validate someone else's code.

Putting the repo root first makes the checkout authoritative, which is what a
test run is supposed to mean. In CI this is a no-op -- the package is installed
from the checkout anyway -- but it keeps local runs honest.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)
