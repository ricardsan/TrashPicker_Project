"""The Python side of robot/rapid/ArtCamera.modx: one message out, one reply back.

The protocol
------------
Every message is the ASCII string "class,x,y,z" or "class,x,y,z,rz", with the position in millimetres
in the robot's base frame.

    class >= 0   move the TCP to (x, y, z) with MoveL, after a workspace check on x and y
    class == -1  GoHome       x, y and z are required by the parser and otherwise ignored
    class == -2  GripOpen     likewise
    class == -3  GripClose    likewise

rz turns the tool on a move: degrees about the base's +z, anticlockwise seen from above, applied as
OrientZYX(rz, 0, 0) * pHome.rot. Without it, the tool has pHome's orientation.

Every message gets exactly one reply:

    DONE          the motion or gripper action has finished ("fine" point, so this blocks)
    REJECTED      x or y is outside the workspace, or rz beyond +-180; nothing moved
    PARSE_ERROR   the message was not four or five numbers

TCP has no message boundaries and RAPID's SocketReceive takes whatever has arrived, so two messages
sent back to back would be parsed as one. Waiting for each reply before sending again is what keeps
them apart, which is why exchange() is the only way out of this class.

**The controller does not check z** -- InWorkspace has a TODO where that belongs -- so nothing
stops a bad z from driving the gripper through the table. pick_sequence.py guards it on this side.

Disconnecting ends the RAPID program: the ERROR handler in main closes both sockets and RETURNs on
ERR_SOCK_CLOSED, so `main` may have to be restarted between runs.
"""
from __future__ import annotations

import math
import select
import socket
import time
from dataclasses import dataclass

# =============================================================================
# 1. MIRRORED FROM ArtCamera.modx
# =============================================================================
# test_rapid_mirror.py diffs every one of these against the .modx, so editing the RAPID side without
# editing this fails a test rather than quietly moving the robot somewhere else.
DEFAULT_HOST = "127.0.0.1"      # the virtual controller; the real one is --host 192.168.125.1
DEFAULT_PORT = 5002

X_MIN_MM, X_MAX_MM = -1000.0, 1000.0
Y_MIN_MM, Y_MAX_MM = -1000.0, 1000.0
Z_TABLE_MM = 70.0               # the table's surface in the base frame
Z_SAFE_MM = 100.0               # RAPID's approach clearance above the table
HOME_Z_MM = 400.0               # the z of pHome. There is no message that reads the pose back.
RZ_MAX_DEG = 180.0              # main's Abs(nRz) check

CMD_HOME, CMD_GRIP_OPEN, CMD_GRIP_CLOSE = -1, -2, -3
DONE, REJECTED, PARSE_ERROR = "DONE", "REJECTED", "PARSE_ERROR"

CONNECT_TIMEOUT_S = 10.0
# DONE only comes back once the robot has stopped. v100 is 100 mm/s, so the longest move in the
# workspace is under 15 s; this is for "the controller has stopped answering", not for a slow move.
REPLY_TIMEOUT_S = 120.0
# How often a waiting exchange() looks up to call on_wait.
WAIT_STEP_S = 0.1


class RobotError(RuntimeError):
    """Stop commanding the robot: the link is gone, or a reply made no sense."""


class MoveRejected(Exception):
    """The controller refused a move and nothing moved. Whether that is fatal is the caller's call."""


# =============================================================================
# 2. THE CLIENT
# =============================================================================
def format_move(cls: int, x: float, y: float, z: float, rz_deg: float | None = None) -> str:
    """A move message, to the tenth of a millimetre yolo_robot_test.py already sends.

    With no rz there is no fifth field, and the tool has pHome's orientation.
    """
    position = f"{cls},{x:.1f},{y:.1f},{z:.1f}"
    return position if rz_deg is None else f"{position},{rz_deg:.1f}"


class ArtCameraClient:
    """A connection to the RAPID socket server. Use it as a context manager.

    `on_wait` is called every WAIT_STEP_S while a reply is awaited. The image window is what needs
    it: a move takes longer than the 5 s after which Windows marks a window that has stopped
    pumping as Not Responding, and offers to end the process.
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 connect_timeout_s: float = CONNECT_TIMEOUT_S,
                 reply_timeout_s: float = REPLY_TIMEOUT_S, on_wait=None):
        self.host = host
        self.port = port
        self.connect_timeout_s = connect_timeout_s
        self.reply_timeout_s = reply_timeout_s
        self.on_wait = on_wait or (lambda: None)
        self._sock: socket.socket | None = None

    def __enter__(self) -> "ArtCameraClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def connect(self) -> None:
        try:
            self._sock = socket.create_connection((self.host, self.port),
                                                  timeout=self.connect_timeout_s)
        except OSError as error:
            raise RobotError(
                f"cannot connect to {self.host}:{self.port}: {error}\n"
                f"Is the RAPID program (main) running on the controller?") from error
        self._sock.settimeout(self.reply_timeout_s)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def exchange(self, message: str) -> str:
        """Send one message and block until its reply. Anything but a reply is a RobotError."""
        if self._sock is None:
            raise RobotError("not connected to the controller")
        try:
            self._sock.sendall(message.encode("ascii"))
            deadline = time.monotonic() + self.reply_timeout_s
            while not select.select([self._sock], [], [], WAIT_STEP_S)[0]:
                if time.monotonic() >= deadline:
                    raise RobotError(f"no reply to {message!r} within "
                                     f"{self.reply_timeout_s:.0f} s; the robot may still be moving")
                self.on_wait()
            reply = self._sock.recv(1024)
        except OSError as error:
            raise RobotError(f"lost the controller while sending {message!r}: {error}") from error
        if not reply:
            raise RobotError(f"the controller closed the connection after {message!r}. A RAPID "
                             f"error stops main; check the FlexPendant or RobotStudio.")
        return reply.decode("ascii", errors="replace").strip()

    def _expect_done(self, message: str) -> None:
        reply = self.exchange(message)
        if reply != DONE:
            raise RobotError(f"{message!r} got {reply!r}, expected {DONE}")

    def home(self) -> None:
        self._expect_done(f"{CMD_HOME},0,0,0")

    def grip_open(self) -> None:
        self._expect_done(f"{CMD_GRIP_OPEN},0,0,0")

    def grip_close(self) -> None:
        self._expect_done(f"{CMD_GRIP_CLOSE},0,0,0")

    def move_to(self, cls: int, x: float, y: float, z: float, rz_deg: float | None = None) -> None:
        """Move the TCP in a straight line. MoveRejected if the controller refuses."""
        # A negative class is a command, so a bad class id would turn a move into a GoHome.
        if cls < 0:
            raise ValueError(f"class {cls} is a RAPID command, not an object class")
        if not all(math.isfinite(value) for value in (x, y, z)):
            raise ValueError(f"not a position: ({x}, {y}, {z})")
        # Also catches a NaN, which fails every comparison.
        if rz_deg is not None and not abs(rz_deg) <= RZ_MAX_DEG:
            raise ValueError(f"not a rotation within +-{RZ_MAX_DEG:.0f} degrees: {rz_deg}")
        message = format_move(cls, x, y, z, rz_deg)
        reply = self.exchange(message)
        if reply == REJECTED:
            raise MoveRejected(f"the controller rejected {message!r}: x must be in "
                               f"{X_MIN_MM:.0f}..{X_MAX_MM:.0f}, y in "
                               f"{Y_MIN_MM:.0f}..{Y_MAX_MM:.0f} and rz within "
                               f"+-{RZ_MAX_DEG:.0f}")
        if reply != DONE:
            raise RobotError(f"{message!r} got {reply!r}, expected {DONE}")


# =============================================================================
# 3. THE MOCK RUN
# =============================================================================
@dataclass(frozen=True)
class MockStep:
    message: str
    expected: str
    note: str


# yolo_robot_test.py's MOCK_TESTS, checked instead of only printed, with four deliberate changes:
#
#   * it starts with GoHome, so the moves that follow start from a known pose;
#   * valid moves are at Z_TABLE + Z_SAFE = 170 rather than 70, because 70 *is* the table and this
#     also runs against a real robot;
#   * the first entry there is described as "Robot moved", but x = 0.52 is below X_MIN, so it is
#     rejected. It stays because it is what a position sent in metres looks like;
#   * it also covers the rz field, which that script predates.
MOCK_STEPS = (
    MockStep("-1,0,0,0", DONE, "go home first"),
    MockStep("0,0.523362338,0.177566765,-0.000906113", REJECTED, "metres by mistake: x < X_MIN"),
    MockStep("0,400,0,170", DONE, "valid point in the middle of the workspace"),
    MockStep("1,700,0,170", REJECTED, "x = 700 > X_MAX = 650"),
    MockStep("2,400,400,170", REJECTED, "y = 400 > Y_MAX = 350"),
    MockStep("abc,def", PARSE_ERROR, "not numbers"),
    MockStep("1,300", PARSE_ERROR, "too few fields"),
    MockStep("3,300.5,-120.2,172", DONE, "decimals"),
    MockStep("0,400,0,170,90", DONE, "valid point, with the tool turned 90 degrees"),
    MockStep("0,400,0,170,181", REJECTED, "rz = 181 > 180"),
    MockStep("0,400,0,170,abc", PARSE_ERROR, "rz is not a number"),
    MockStep("0,400,0,170,90,1", PARSE_ERROR, "too many fields"),
    MockStep("0,400,0,170,", PARSE_ERROR, "a comma with no rz after it"),
    MockStep("-1, 1000,2000,3000", DONE, "home, ignoring a wild x,y,z"),
    MockStep("-3, 1000,2000,3000", DONE, "close the gripper, ignoring a wild x,y,z"),
    MockStep("-2, 1000,2000,3000", DONE, "open the gripper, ignoring a wild x,y,z"),
)


def run_mock(client: ArtCameraClient, steps=MOCK_STEPS) -> bool:
    """Send each step and compare its reply. True if every one came back as expected.

    Stops at the first reply that differs. These steps move a robot, and a controller that has just
    answered something unexpected is not one to keep sending motion to.
    """
    print("=== Mock run: socket and RAPID logic only, no model and no camera ===")
    print("This MOVES THE ROBOT: home, then three valid points, the last with the tool turned. "
          "Keep the work area clear.\n")
    for position, step in enumerate(steps, 1):
        reply = client.exchange(step.message)
        matched = reply == step.expected
        print(f"{'ok  ' if matched else 'FAIL'}  TX {step.message!r:44s} RX {reply!r:14s} "
              f"expected {step.expected}  ({step.note})")
        if not matched:
            print(f"\nMock run stopped at step {position} of {len(steps)}: unexpected reply.")
            return False
    print(f"\nMock run finished: {len(steps)} of {len(steps)} replies as expected.")
    return True
