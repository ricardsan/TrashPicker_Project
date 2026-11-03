"""Keys from either place, and the ones pressed too early."""
from __future__ import annotations

import pytest

from robot.operator_keys import OperatorKeys


@pytest.fixture
def pressed(monkeypatch):
    """Scripts cv2.waitKey and the console. Each entry is the result of one poll."""

    def install(window=(), console=()):
        window, console = list(window), list(console)
        monkeypatch.setattr("cv2.waitKey",
                            lambda delay=0: window.pop(0) if window else -1)
        monkeypatch.setattr("robot.operator_keys.console_key",
                            lambda: console.pop(0) if console else None)
        monkeypatch.setattr("robot.operator_keys.time.sleep", lambda seconds: None)

    return install


def test_nothing_pressed_is_none(pressed):
    pressed()
    assert OperatorKeys().poll() is None


def test_a_key_in_the_window(pressed):
    pressed(window=[ord("n")])
    assert OperatorKeys().poll() == "n"


def test_a_key_in_the_terminal(pressed):
    pressed(console=["n"])
    assert OperatorKeys().poll() == "n"


def test_caps_lock_does_not_matter(pressed):
    pressed(window=[ord("N")], console=["C"])
    keys = OperatorKeys()
    assert keys.poll() == "n"
    assert keys.poll() == "c"


def test_the_high_bits_of_waitkey_are_ignored(pressed):
    """cv2.waitKey can return modifier bits above the low byte on some platforms."""
    pressed(window=[0x100000 | ord("s")])
    assert OperatorKeys().poll() == "s"


def test_wait_for_ignores_other_keys(pressed):
    pressed(window=[-1, ord("x"), ord("q"), -1, ord("s")])
    assert OperatorKeys().wait_for("cs") == "s"


def test_a_key_pressed_during_the_last_move_is_not_a_go_ahead(pressed):
    """Commands block for seconds at a time and keys queue up meanwhile. These two 'c's were pressed
    before the prompt, so the answer has to be the 's' that comes after it."""
    pressed(window=[ord("c"), ord("c"), -1, -1, ord("s")])
    assert OperatorKeys().wait_for("cs") == "s"


def test_a_stale_console_key_is_forgotten_too(pressed):
    pressed(console=["c", None, "s"])
    assert OperatorKeys().wait_for("cs") == "s"
