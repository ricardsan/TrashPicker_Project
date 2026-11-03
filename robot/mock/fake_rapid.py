"""A stand-in for robot/rapid/ArtCamera.modx, for running the client with no controller.

It follows the same message handling as main, ParseMessage and InWorkspace, and reads its limits out
of the .modx itself, so editing the RAPID side moves this too. It records what it was told and keeps
track of the pose and the gripper.

**What it cannot tell you** is anything about motion: reachability, ConfL, singularities, the
gripper's I/O, or how StrToVal really treats an odd string. Passing against this says the client
sends the right messages in the right order, and nothing about whether the robot can follow them.

Also runs on its own, so the real command line can be pointed at it:

    python robot/mock/fake_rapid.py --port 5002
"""
from __future__ import annotations

import argparse
import re
import socket
import threading
from pathlib import Path

MODX = Path(__file__).resolve().parents[1] / "rapid" / "ArtCamera.modx"

HANG = "<no reply>"     # a scripted "reply" that never comes
DROP = None             # a scripted "reply" that closes the connection instead

_CONST_NUM = re.compile(r"^\s*CONST\s+num\s+(\w+)\s*:=\s*(-?[\d.]+)\s*;", re.M)
_CONST_ROBTARGET = re.compile(r"^\s*CONST\s+robtarget\s+(\w+)\s*:=\s*\[\[([^\]]+)\]", re.M)
_CONST_HOST = re.compile(r'^\s*CONST\s+string\s+HOST\s*:=\s*"([^"]+)"\s*;', re.M)
_RZ_LIMIT = re.compile(r"Abs\(nRz\)\s*>\s*([\d.]+)")
_NUMBER = re.compile(r"^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$")


def read_modx(path=MODX) -> dict:
    """The CONST num values, the robtargets' positions, HOST and the rz limit, from the RAPID source."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    values: dict = {name: float(value) for name, value in _CONST_NUM.findall(text)}
    values.update({name: tuple(float(part) for part in body.split(","))
                   for name, body in _CONST_ROBTARGET.findall(text)})
    values["HOST"] = _CONST_HOST.search(text).group(1)
    values["RZ_MAX"] = float(_RZ_LIMIT.search(text).group(1))
    return values


def parse_message(message: str) -> tuple[float, float, float, float, float] | None:
    """ParseMessage: (class, x, y, z, rz), or None. No fifth field means rz = 0. A comma with nothing
    after it fails, and so does a sixth field, which lands in rz."""
    parts = message.split(",", 4)
    if len(parts) < 4 or not all(_NUMBER.match(part) for part in parts):
        return None
    numbers = [float(part) for part in parts]
    return (numbers[0], numbers[1], numbers[2], numbers[3], numbers[4] if len(numbers) == 5 else 0.0)


class FakeRapid:
    """Serves one client at a time, as main does, until stop().

    `script` maps a message's number (from 0) to a forced reply, or to HANG or DROP. A scripted
    message has no effect on the pose, as if the controller had refused it.
    """

    def __init__(self, port: int = 0, script: dict | None = None, verbose: bool = False):
        self.limits = read_modx()
        self.script = script or {}
        self.verbose = verbose
        self.received: list[str] = []
        self.pose: tuple[float, float, float] | None = None
        self.rz: float | None = None
        self.gripper: str | None = None
        self.connections = 0

        self._stopping = threading.Event()
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", port))
        self._server.listen()
        self._server.settimeout(0.05)
        self.port = self._server.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self) -> "FakeRapid":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stopping.set()
        self._thread.join(timeout=2)
        self._server.close()

    def send_late(self, reply: str) -> None:
        """Answer a message scripted as HANG, as a controller finishing a long move would."""
        self._connection.sendall(reply.encode("ascii"))

    def reply_to(self, message: str) -> str:
        """main's WHILE body: parse, validate, act, answer."""
        parsed = parse_message(message)
        if parsed is None:
            return "PARSE_ERROR"
        cls, x, y, z, rz = parsed
        limits = self.limits
        if cls >= 0 and not (limits["X_MIN"] <= x <= limits["X_MAX"]
                             and limits["Y_MIN"] <= y <= limits["Y_MAX"]
                             and abs(rz) <= limits["RZ_MAX"]):
            return "REJECTED"
        if cls == -1:
            self.pose = limits["pHome"]
            self.rz = 0.0
        elif cls == -2:
            self.gripper = "open"
        elif cls == -3:
            self.gripper = "closed"
        elif cls >= 0:
            self.pose = (x, y, z)
            self.rz = rz
        return "DONE"

    def _serve(self) -> None:
        while not self._stopping.is_set():
            try:
                connection, _ = self._server.accept()
            except TimeoutError:
                continue
            self.connections += 1
            self._connection = connection
            with connection:
                connection.settimeout(0.05)
                self._talk(connection)

    def _talk(self, connection: socket.socket) -> None:
        while not self._stopping.is_set():
            try:
                data = connection.recv(1024)
            except TimeoutError:
                continue
            except OSError:
                return
            if not data:
                return

            message = data.decode("ascii", errors="replace")
            number = len(self.received)
            self.received.append(message)
            reply = self.script[number] if number in self.script else self.reply_to(message)
            if self.verbose:
                print(f"RX {message!r:44s} TX {reply!r}  pose={self.pose} rz={self.rz} "
                      f"gripper={self.gripper}")
            if reply is DROP:
                return
            if reply != HANG:
                connection.sendall(reply.encode("ascii"))


def main() -> None:
    parser = argparse.ArgumentParser(description="A fake ArtCamera.modx server.")
    parser.add_argument("--port", type=int, default=5002)
    args = parser.parse_args()
    with FakeRapid(args.port, verbose=True):
        print(f"fake ArtCamera listening on 127.0.0.1:{args.port}; Ctrl-C to stop")
        threading.Event().wait()


if __name__ == "__main__":
    main()
