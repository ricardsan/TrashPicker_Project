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
    def poll(self, delay_ms: int = 1) -> str | None:
        """One pending key in lower case, or None. Also what keeps the image window painted."""
        key = cv2.waitKey(delay_ms) & 0xFF
        if key != 0xFF:
            return chr(key).lower()
        key = console_key()
        return key.lower() if key else None

    def drain(self) -> None:
        """Forget everything pressed so far.

        Motion commands block for seconds at a time, and whatever is pressed meanwhile queues up.
        A 'c' pressed during the last move is not a go-ahead for the next one.
        """
        while self.poll() is not None:
            pass

    def wait_for(self, allowed: str) -> str:
        """Block until one of the characters in `allowed` is pressed, and return it."""
        self.drain()
        while True:
            key = self.poll(30)
            if key is not None and key in allowed:
                return key
            # cv2.waitKey returns at once when no window is open, which would make this a spin.
            time.sleep(0.01)
