"""
Camera -> Robot kalibrācija ar ChArUco dēli (viss vienā skriptā).

1. Dēlis guļ darba zonā, NEKUSTINĀT visu procedūras laiku.
2. Kamera nosaka dēļa pozu, ekrānā parāda P0/P1/P2 punktus.
3. Ar robotu (jogging) aizved TCP uz tiem punktiem, ievadi koordinātas (mm).
4. Skripts izrēķina T_base_cam, parāda kļūdu, saglabā .npy.

    python camera/calibrate_T_base_cam.py                        # A2 dēlis -> T_base_cam.npy
    python camera/calibrate_T_base_cam.py --board home50 --output rosbags/<batch>/T_base_cam.npy

Dēļu parametri ir camera/charuco.py (BOARDS).
"""

import argparse
import sys
from pathlib import Path

import pyrealsense2 as rs
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camera.charuco import (BOARDS, detect_pose, make_board,  # noqa: E402
                            reference_distances_mm, reference_points)
from robot.board_demo_calibration import MAX_RMS_PX, MIN_FRAMES, combine_poses  # noqa: E402

MAX_DISTANCE_ERROR_MM = 15.0    # P0-P1 / P0-P2 pret dēli: vairāk = sajaukti punkti
MAX_MEAN_ERROR_MM = 5.0         # Kabsch atlikums: vairāk = TCP nebija uz stūra vai dēlis kustējās

_parser = argparse.ArgumentParser(description="Camera -> robot calibration with a ChArUco board.")
_parser.add_argument("--board", choices=sorted(BOARDS), default="a2",
                     help="which board lies in the work area (default: a2, the lab board)")
_parser.add_argument("--output", default="T_base_cam.npy",
                     help="where to save T_base_cam (default: T_base_cam.npy in the current folder)")
_parser.add_argument("--frames", type=int, default=30,
                     help="how many frames after SPACE the board pose is averaged over (default: 30)")
_parser.add_argument("--force", action="store_true",
                     help="save even when the distance or error checks fail")
_args = _parser.parse_args()

# ====================================
# ChArUco dēļa parametri (camera/charuco.py)
# ====================================

BOARD_SPEC = BOARDS[_args.board]
SQUARES_X = BOARD_SPEC.squares_x
SQUARES_Y = BOARD_SPEC.squares_y
SQUARE_LENGTH = BOARD_SPEC.square_m
MARKER_LENGTH = BOARD_SPEC.marker_m
EXPECTED_D01_MM, EXPECTED_D02_MM = reference_distances_mm(BOARD_SPEC)

OUTPUT_FILE = _args.output
print(f"Dēlis: {_args.board} ({SQUARES_X}x{SQUARES_Y}, {SQUARE_LENGTH * 1000:.0f} mm), "
      f"saglabās: {OUTPUT_FILE}")

# ====================================
# RealSense
# ====================================

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

color_profile = profile.get_stream(rs.stream.color)
intr = color_profile.as_video_stream_profile().get_intrinsics()

camera_matrix = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.array(intr.coeffs, dtype=np.float64)

# ====================================
# ChArUco dēlis
# ====================================

board, charuco_detector = make_board(BOARD_SPEC)

# Atsauces punkti dēļa koordinātās (metros) - iekšējie šaha stūri.
# P1 ir pa X no P0, P2 pa Y no P0 (A2: +240 mm un +300 mm).
ref_points_board = reference_points(BOARD_SPEC)


def transform_points(T, pts):
    """Pielieto 4x4 transformāciju Nx3 punktiem."""
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    return (T @ pts_h.T).T[:, :3]


def kabsch(A, B):
    """Atrod rigid transformāciju T (4x4), lai T @ A ~= B."""
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ====================================
# 1. SOLIS: dēļa poza no kameras
# ====================================

print("=" * 50)
print("1. SOLIS: dēļa pozas noteikšana")
print("Novieto dēli darba zonā. NEKUSTINI to pēc šī brīža!")
print("SPACE - fiksēt pozu | ESC - iziet")
print("=" * 50)

T_cam_board = None

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            charuco_detector.detectBoard(gray)

        display = frame.copy()
        pose_ok = False
        rvec, tvec = None, None

        if charuco_ids is not None and len(charuco_ids) >= 6:
            # sturu zimesana manuali (OpenCV 5 saderiba)
            for i in range(len(charuco_ids)):
                c = charuco_corners[i].ravel()
                cv2.circle(display, (int(c[0]), int(c[1])), 4, (0, 255, 0), -1)

            obj_pts, img_pts = board.matchImagePoints(
                charuco_corners, charuco_ids)

            if obj_pts is not None and len(obj_pts) >= 6:
                obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
                img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)

                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, camera_matrix, dist_coeffs)
                if ok:
                    pose_ok = True
                    cv2.drawFrameAxes(display, camera_matrix, dist_coeffs,
                                      rvec, tvec, 0.1)

                    # Uzzīmē atsauces punktus P0, P1, P2
                    proj, _ = cv2.projectPoints(
                        ref_points_board, rvec, tvec,
                        camera_matrix, dist_coeffs)
                    for i, p in enumerate(proj.reshape(-1, 2)):
                        pt = (int(p[0]), int(p[1]))
                        cv2.circle(display, pt, 8, (0, 0, 255), -1)
                        cv2.putText(display, f"P{i}",
                                    (pt[0] + 10, pt[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 0, 255), 2)

        status = "POZA OK - spied SPACE" if pose_ok else "Mekle deli..."
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if pose_ok else (0, 0, 255), 2)

        cv2.imshow("Kalibracija", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("Partraukts.")
            raise SystemExit

        if key == 32 and pose_ok:
            # Viens kadrs ir trokšņains: vidējo pozu no nākamajiem kadriem, tikai ar labu
            # reprojekcijas kļūdu (tas pats, ko dara board_demo_calibration.py).
            print(f"Vacu {_args.frames} kadrus, nekustini deli un kameru...")
            poses, tried = [], 0
            while tried < _args.frames:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                tried += 1
                gray = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2GRAY)
                pose = detect_pose(gray, camera_matrix, dist_coeffs, board, charuco_detector)
                if pose is not None and pose.rms_px <= MAX_RMS_PX:
                    poses.append(pose.T_cam_board)
            if len(poses) < MIN_FRAMES:
                print(f"Tikai {len(poses)} no {tried} kadriem ar kludu <= {MAX_RMS_PX} px. "
                      f"Uzlabo apgaismojumu vai pieliec kameru tuvak, un spied SPACE velreiz.")
                continue
            T_cam_board, spread_mm, spread_deg = combine_poses(poses)
            print(f"Dela poza fikseta: {len(poses)} no {tried} kadriem, izkliede "
                  f"{spread_mm:.1f} mm / {spread_deg:.2f} grad.")
            print("Dela origin kameras rami (m):", T_cam_board[:3, 3])
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# Atsauces punkti KAMERAS rāmī (metros)
ref_points_cam = transform_points(T_cam_board, ref_points_board)

print()
print("Atsauces punkti kameras rami (mm):")
for i, p in enumerate(ref_points_cam):
    print(f"  P{i}: X={p[0]*1000:.1f}  Y={p[1]*1000:.1f}  Z={p[2]*1000:.1f}")

# ====================================
# 2. SOLIS: robota koordinātas
# ====================================

print()
print("=" * 50)
print("2. SOLIS: robota koordinatas")
print("Ar jogging aizved TCP smaili PRECIZI uz katru punktu")
print("(sarkanie punkti ekrana: P0, P1, P2).")
print("Nolasi robota poziciju (mm, Base frame) un ievadi seit.")
print("SVARIGI: FlexPendant jabut Tool = toolRG6 (pirkstu gali, [0,0,253]) un")
print("koordinatu sistemai Base (vai World ar wobj0). Ar tool0 punkti bus 253 mm nost.")
print("=" * 50)

ref_points_robot = np.zeros((3, 3))

for i in range(3):
    while True:
        try:
            raw = input(f"P{i} robota koordinatas 'x,y,z' mm: ")
            x, y, z = [float(v.strip()) for v in raw.split(",")]
            ref_points_robot[i] = [x / 1000.0, y / 1000.0, z / 1000.0]
            break
        except ValueError:
            print("Nepareizs formats. Piemers: 412.5, -87.3, 12.0")

# Geometrijas parbaude pirms aprekina
d01 = np.linalg.norm(ref_points_robot[1] - ref_points_robot[0]) * 1000
d02 = np.linalg.norm(ref_points_robot[2] - ref_points_robot[0]) * 1000
print()
print(f"Parbaude: P0-P1 attalums = {d01:.1f} mm (jabut ~{EXPECTED_D01_MM:.0f})")
print(f"          P0-P2 attalums = {d02:.1f} mm (jabut ~{EXPECTED_D02_MM:.0f})")
problems = []
if abs(d01 - EXPECTED_D01_MM) > MAX_DISTANCE_ERROR_MM or abs(d02 - EXPECTED_D02_MM) > MAX_DISTANCE_ERROR_MM:
    problems.append("attalumi neatbilst delim - punkti var but sajaukti vai nepareizs --board")
    print(f"BRIDINAJUMS: {problems[-1]}!")

# ====================================
# 3. SOLIS: T_base_cam aprēķins
# ====================================

T_base_cam = kabsch(ref_points_cam, ref_points_robot)

check = transform_points(T_base_cam, ref_points_cam)
errors = np.linalg.norm(check - ref_points_robot, axis=1) * 1000  # mm

print()
print("=" * 50)
print("REZULTATS")
print("=" * 50)
print("T_base_cam:")
print(T_base_cam)
print()
print("Atbilstibas kluda katra punkta:")
for i, e in enumerate(errors):
    print(f"  P{i}: {e:.2f} mm")
print(f"Videja kluda: {errors.mean():.2f} mm")

if errors.mean() > MAX_MEAN_ERROR_MM:
    problems.append(f"videja kluda {errors.mean():.1f} mm > {MAX_MEAN_ERROR_MM:.0f} mm")
    print()
    print(f"BRIDINAJUMS: kluda > {MAX_MEAN_ERROR_MM:.0f} mm. Iespejamie iemesli:")
    print("  - TCP nebija precizi uz stura (vai nebija toolRG6)")
    print("  - delis pakustejas starp 1. un 2. soli")
    print("  - nepareizi dela parametri (--board?)")
    print("Iesaku atkartot.")

if problems and not _args.force:
    print()
    print("NESAGLABATS: " + "; ".join(problems) + ".")
    print("Atkarto merijumus, vai ar --force saglaba tik un ta.")
    raise SystemExit(1)

Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT_FILE, T_base_cam)
print()
print(f"Saglabats: {OUTPUT_FILE}")
print()
print("Lietosana:")
print(f"  python robot/robot_trash_picker.py --host 192.168.125.1 --calibration {OUTPUT_FILE}")
print("  (T ir METROS: p_robot_mm = (T @ [X_m, Y_m, Z_m, 1])[:3] * 1000)")