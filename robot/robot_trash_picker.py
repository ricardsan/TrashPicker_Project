"""Pick trash with the robot: segment a frame, then take the objects one by one, nearest first.

The robot end is robot/rapid/ArtCamera.modx, which has to be running on the controller first.

    # Socket and RAPID logic only: no model, no camera. This still moves the robot.
    python robot/robot_trash_picker.py --mock

    # An unpacked clip from rosbags/, using the detections of its last frame, or of --frame N.
    python robot/robot_trash_picker.py --rosbag 20261016_231841_clutter_med.db3

    # The RealSense on the real controller. Press 'n' when the scene is ready. --calibration is needed
    # here: the hardcoded extrinsics in extrinsics.py, which a run loads by default, are refused for
    # any host but this machine.
    python robot/robot_trash_picker.py --host 192.168.125.1 --calibration robot/calibration.txt

The run: load the model and the frames, send the robot home, run inference. On the chosen frame, or
when 'n' is pressed, measure every mask and sort nearest the camera first. Then for each target:
over it, down, close the gripper, back up, to its class's drop location, open, home, and wait for
'c' (next target) or 's' (stop). The list is not re-detected between picks.

The moving parts:

    art_camera_client.py    the socket protocol, and the mock run
    extrinsics.py           the HARDCODED EXTRINSICS, and calibration.txt -> T_base_cam
    frame_selection.py      rosbag or camera, and which frame
    perception.py           mask -> centroid, orientation, z, class, base position
    pick_sequence.py        the motion sequence, the DUMMY DROP LOCATIONS, the grasp z PLACEHOLDER
    operator_keys.py        keys from the image window or the terminal
    align_to_recording.py   writes calibration_rosbag_demo.txt; not part of a run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from camera.segmentation_realsense import (RealSenseSource, YoloDetector,  # noqa: E402
                                           check_weights, resolve_custom_weights)
from robot.art_camera_client import (DEFAULT_HOST, DEFAULT_PORT,  # noqa: E402
                                     ArtCameraClient, RobotError, run_mock)
from robot.extrinsics import (HARDCODED_NAME, HARDCODED_T_BASE_CAM,  # noqa: E402
                              check_transform, is_synthetic, load_T_base_cam)
from robot.frame_selection import (open_recording, preview_until,  # noqa: E402
                                   resolve_frame_index, settle_frames, started)
from robot.experiment import (DEFAULT_DIR as EXPERIMENTS_DIR, ExperimentLog,  # noqa: E402
                              ask_verdict, calibration_meta, read_ground_truth)
from robot.operator_keys import OperatorKeys  # noqa: E402
from robot.perception import format_targets  # noqa: E402
from robot.pick_sequence import PickAborted, check_drop_locations, run_picks  # noqa: E402

DEFAULT_WEIGHTS = (ROOT / "runs" / "segment" / "runs" / "segment" / "original-seg"
                   / "original-finetune-augment_v" / "weights" / "best.pt")
LAST_FRAME = -1
DEFAULT_CONF = 0.25
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pick objects with the robot, from a rosbag clip or the RealSense.")
    p.add_argument("--mock", action="store_true",
                   help="connect and run a fixed list of commands, with no model and no camera. "
                        "This still moves the robot.")
    p.add_argument("--host", default=DEFAULT_HOST,
                   help=f"the controller (default: {DEFAULT_HOST}, the virtual one; the real one "
                        f"is 192.168.125.1)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"default: {DEFAULT_PORT}, as in ArtCamera.modx")
    p.add_argument("--model", default=None, metavar="BEST.PT",
                   help=f"a YOLO segmentation checkpoint, or a run directory holding one (default: "
                        f"{DEFAULT_WEIGHTS.relative_to(ROOT)})")
    p.add_argument("--conf", type=float, default=None,
                   help=f"confidence threshold (default: {DEFAULT_CONF})")
    p.add_argument("--rosbag", default=None, metavar="BAG.db3",
                   help="use a clip instead of the camera: a file name in rosbags/, served from "
                        "its unpacked copy in rosbags/export/, and unpacked first if there is none")
    p.add_argument("--frame", type=int, default=None, metavar="N",
                   help="with --rosbag: which frame's detections to use, from 0, or negative to "
                        "count from the end (default: -1, the last frame). The depth of a clip's "
                        "first second is unreliable.")
    p.add_argument("--calibration", default=None, metavar="FILE",
                   help=f"T_base_cam, a text file like calibration.txt or a .npy (default: "
                        f"{HARDCODED_NAME}, for a virtual controller on this machine only)")
    p.add_argument("--experiment", default=None, metavar="NAME",
                   help="log this run as an experiment: snapshot, targets, every pick with your "
                        "y/n/x verdict, and the scene point cloud, in experiments/<date>_NAME/")
    p.add_argument("--experiments-dir", default=None, metavar="DIR",
                   help=f"where experiments go (default: {EXPERIMENTS_DIR.relative_to(ROOT)})")
    p.add_argument("--ground-truth", default=None, metavar="CSV",
                   help="with --experiment: class,x_mm,y_mm,axis_deg of each object in the base "
                        "frame, to measure position and angle errors")
    p.add_argument("--viz", action="store_true",
                   help="open a live 3D view (Rerun) of the RGB image, point cloud and planned "
                        "grasps in the base frame, to run beside RobotStudio")
    return p


def check_args(parser: argparse.ArgumentParser, args) -> None:
    """Reject flags that mean nothing together, rather than quietly ignoring them."""
    if args.mock:
        for flag in ("model", "conf", "rosbag", "frame", "calibration", "experiment",
                     "ground_truth"):
            if getattr(args, flag) is not None:
                parser.error(f"--{flag.replace('_', '-')} means nothing with --mock, which loads "
                             f"no model and no frames")
        if args.viz:
            parser.error("--viz means nothing with --mock, which loads no model and no frames")
    if args.frame is not None and args.rosbag is None:
        parser.error("--frame needs --rosbag: with the camera, the frame is the one on screen "
                     "when 'n' is pressed")
    if (args.ground_truth or args.experiments_dir) and not args.experiment:
        parser.error("--ground-truth and --experiments-dir need --experiment NAME")


def run_pick_session(args) -> int:
    # Everything that can be wrong with the inputs is checked before the robot moves, so a bad
    # --frame, a missing checkpoint or an absent camera costs nothing but a message.
    if args.calibration:
        calibration, T_base_cam = args.calibration, load_T_base_cam(args.calibration)
        synthetic = is_synthetic(args.calibration)
    else:
        calibration, T_base_cam = HARDCODED_NAME, check_transform(HARDCODED_T_BASE_CAM, HARDCODED_NAME)
        synthetic = True
    if synthetic and args.host not in LOCAL_HOSTS:
        raise SystemExit(f"{calibration} is synthetic and describes nothing in the room, so it is "
                         f"refused for {args.host}. It is only for a virtual controller on this "
                         f"machine. For a real run, pass --calibration robot/calibration.txt.")
    check_drop_locations()
    try:
        weights = check_weights(resolve_custom_weights(args.model or DEFAULT_WEIGHTS))
    except (FileNotFoundError, RuntimeError) as failure:
        raise SystemExit(str(failure)) from failure
    if args.ground_truth:
        read_ground_truth(args.ground_truth)        # a bad file fails here, not after the picks
    view = None
    if args.viz:
        from robot.live_view import LiveView
        view = LiveView(T_base_cam)

    if args.rosbag:
        source = open_recording(args.rosbag)
        stop_at = resolve_frame_index(LAST_FRAME if args.frame is None else args.frame,
                                      len(source.frames))
        if stop_at < settle_frames(source):
            print(f"WARNING: frame {stop_at} is in the first {settle_frames(source)} frames, "
                  f"where a clip's depth was measured at 33-77% holes and up to 158 mm off.")
    else:
        source, stop_at = RealSenseSource(), None

    keys = OperatorKeys()
    print(f"connecting to {args.host}:{args.port} ...")
    # Pumping the window while a move is awaited, or Windows marks it as Not Responding.
    with ArtCameraClient(args.host, args.port, on_wait=lambda: cv2.waitKey(1)) as robot:
        print(f"loading {weights}")
        detector = YoloDetector(weights, DEFAULT_CONF if args.conf is None else args.conf)
        print(f"frames from: {source.describe()}")
        print(f"calibration: {calibration}")

        with started(source):
            print("sending the robot home")
            robot.home()
            # The state of the gripper is unknown here, and a closed one cannot pick anything.
            print("opening the gripper")
            robot.grip_open()

            if stop_at is None:
                print("\nLive preview. Press 'n' when the scene is ready, or 'q' or ESC to quit. "
                      "The keys work in the image window or in this terminal.")
            else:
                print(f"\nRunning inference up to frame {stop_at}. Press 'q' or ESC to quit.")
            snapshot = preview_until(source, detector, T_base_cam, keys, stop_at=stop_at,
                                     view=view)

        if snapshot is None:
            print("Quit before picking anything. The robot is at home.")
            return 0

        print(f"\nDetections on frame {snapshot.frame_index}, nearest the camera first:")
        for reason in snapshot.dropped:
            print(f"  not in the list -- {reason}")
        if not snapshot.targets:
            print("Nothing to pick. The robot is at home.")
            return 0
        print(format_targets(snapshot.targets))

        log = None
        if args.experiment:
            log = ExperimentLog(args.experiment, {
                "source": source.describe(), "rosbag": args.rosbag or "",
                "frame": snapshot.frame_index, "model": str(weights),
                "conf": DEFAULT_CONF if args.conf is None else args.conf,
                "calibration": str(calibration), "synthetic_calibration": synthetic,
                "T_base_cam": calibration_meta(T_base_cam), "host": args.host,
            }, root=args.experiments_dir or EXPERIMENTS_DIR, ground_truth=args.ground_truth)
            log.snapshot(snapshot, T_base_cam)
            print(f"experiment: {log.dir}")
        if view is not None:
            view.snapshot(snapshot.frame_index, snapshot.targets)

        def on_picked(rank, target, ran):
            verdict = (ask_verdict(keys) if log is not None else "") if ran else "not run"
            if log is not None:
                log.pick(rank, target, ran, verdict)
            if view is not None:
                view.pick_result(rank, target, verdict or "done")

        hook = on_picked if (log is not None or view is not None) else None
        completed = run_picks(robot, snapshot.targets, keys, on_picked=hook)
        print(f"\nFinished {completed} pick sequence(s).")
        if log is not None:
            log.finish(completed)
            print(f"experiment written to {log.dir}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    check_args(parser, args)

    try:
        if args.mock:
            with ArtCameraClient(args.host, args.port) as robot:
                return 0 if run_mock(robot) else 1
        return run_pick_session(args)
    except (RobotError, PickAborted) as failure:
        print(f"\nSTOPPED: {failure}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. The robot finishes the move it is on; no more are sent.",
              file=sys.stderr)
        return 130
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
