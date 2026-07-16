"""I7 — Public facade tests (superseded by test_i7_config_contract.py).

The full I7 integration test suite (34 tests, IT1-IT6) lives in:
    test/test_i7_config_contract.py

This file exists as a redirect to avoid breaking roadmap references.
"""


def test_facade_tests_exist_elsewhere():
    """All I7 facade tests are in test_i7_config_contract.py.

    This test verifies that the real test file exists and is importable,
    so that any system referencing test_facade.py doesn't silently skip.
    """
    import importlib
    try:
        mod = importlib.import_module("test.test_i7_config_contract")
    except ModuleNotFoundError as e:
        pytest.fail(
            f"test_i7_config_contract.py not importable — "
            f"run via pytest (not python directly): {e}"
        )
    assert hasattr(mod, "TestPublicFacadePipeline"), (
        "I7 facade tests missing from test_i7_config_contract.py"
    )
