"""Keep Streamlit UI tests from changing multiprocessing spawn's entry point."""
import sys

import pytest


@pytest.fixture(autouse=True)
def restore_main_after_ui_test():
    # AppTest installs the app script as __main__. Spawn must keep pytest's
    # original entry point rather than re-running the UI in worker processes.
    original = sys.modules.get("__main__")
    yield
    if original is not None:
        sys.modules["__main__"] = original
