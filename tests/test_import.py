"""Smoke tests: importing the package, and its modules, shouldn't blow up.

These exist because `hear/__init__.py` carried a dead re-export
(`from hum.sound import Sound`) against a module that no longer exists in
`hum`. That made `import hear` fail outright, which in turn made every test
module in the repo fail at *collection* -- so the test suite could not report
on anything at all. A plain `import hear` is therefore a real regression test,
not a formality.
"""

import importlib

import pytest


def test_import_hear():
    """The top-level package imports."""
    import hear  # noqa: F401


def test_no_dead_hum_sound_reexport():
    """`hum.sound` does not exist; `hear` must not try to import from it.

    Guards the specific breakage: re-pointing this at another missing module
    would fail here too.
    """
    import hear

    assert not hasattr(hear, "Sound"), (
        "hear re-exports `Sound` again -- verify the source module actually "
        "exists before re-adding it"
    )
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("hum.sound")


@pytest.mark.parametrize(
    "module_name",
    [
        "hear.audio_segments",
        "hear.regular_panel_data",
        "hear.sequential",
        "hear.session_block_stores",
        "hear.stores",
        "hear.tools",
        "hear.util",
    ],
)
def test_submodules_import(module_name):
    """Every non-scrap submodule imports cleanly."""
    importlib.import_module(module_name)


def test_public_names_present():
    """The names `__init__` promises are actually bound."""
    import hear

    for name in ("AudioSegments", "wf_func_to_wfsr_func"):
        assert hasattr(hear, name), f"hear.{name} missing"
