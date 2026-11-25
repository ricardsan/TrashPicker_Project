"""Keys from the OpenCV window or the terminal, whichever the operator is looking at.

cv2.waitKey only hears a key while the image window has focus, and the instructions are printed in
the terminal, which is where the focus usually is after reading them. Listening to one and not the
other is how an operator ends up pressing 'c' at a robot that ignores it.
"""
from __future__ import annotations

import os
import sys
import time

import cv2

ESC = "\x1b"
QUIT_KEYS = ("q", ESC)

if os.name == "nt":
    import msvcrt

    def console_key() -> str | None:
        if not msvcrt.kbhit():
            return None
        key = msvcrt.getwch()
        # Arrow and function keys arrive as a prefix and then a code. Swallow both, or the code
        # would be read as a letter on the next poll.
        if key in ("\x00", "\xe0"):
            msvcrt.getwch()
            return None
        return key
else:
    import select

    def console_key() -> str | None:
        # A POSIX terminal is line-buffered, so here a key in the terminal needs Enter after it.
        if not sys.stdin.isatty() or not select.select([sys.stdin], [], [], 0)[0]:
            return None
        return sys.stdin.readline().strip()[:1] or None


class OperatorKeys:
    def __init__(self):
        # Keys that act at once wherever they are pressed, and are then not seen by the caller:
        # a key the run is not waiting for would otherwise be lost, in wait_for and in drain.
        self.actions: dict[str, object] = {}
        self._acted = False

    def on(self, key: str, action) -> None:
        """Call `action()` whenever `key` is pressed, at any prompt, instead of returning it."""
        self.actions[key.lower()] = action

    def poll(self, delay_ms: int = 1) -> str | None:
        """One pending key in lower case, or None. Also what keeps the image window painted."""
        key = self._read(delay_ms)
        return None if key is None else self._act(key)

    def _read(self, delay_ms: int) -> str | None:
        key = cv2.waitKey(delay_ms) & 0xFF
        if key != 0xFF:
            return chr(key).lower()
        key = console_key()
        return key.lower() if key else None

    def _act(self, key: str) -> str | None:
        action = self.actions.get(key)
        if action is None:
            return key
        action()
        self._acted = True
        return None

    def pump(self, delay_ms: int = 1) -> None:
        """Keep the window painted while something else blocks, e.g. a move: an action key still
        acts, and any other key is dropped, as the drain at the next prompt would drop it."""
        key = self._read(delay_ms)
        if key is not None:
            self._act(key)

    def drain(self) -> None:
        """Forget everything pressed so far, though an action key still acts.

        Motion commands block for seconds at a time, and whatever is pressed meanwhile queues up.
        A 'c' pressed during the last move is not a go-ahead for the next one. This reads until
        nothing is left rather than until poll() says None, which an action key also does.
        """
        while True:
            self._acted = False
            if self.poll() is None and not self._acted:
                return

    def wait_for(self, allowed: str) -> str:
        """Block until one of the characters in `allowed` is pressed, and return it."""
        self.drain()
        while True:
            key = self.poll(30)
            if key is not None and key in allowed:
                return key
            # cv2.waitKey returns at once when no window is open, which would make this a spin.
            time.sleep(0.01)
