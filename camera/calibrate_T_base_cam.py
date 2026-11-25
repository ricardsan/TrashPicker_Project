"""
Camera -> Robot kalibrācija ar ChArUco dēli (viss vienā skriptā).

1. Dēlis guļ darba zonā, NEKUSTINĀT viena mērījuma laikā.
2. Kamera nosaka dēļa pozu, ekrānā parāda P0/P1/P2 punktus.
3. Ar robotu (jogging) aizved TCP uz tiem punktiem, ievadi koordinātas (mm).
4. Skripts izrēķina T_base_cam no VISIEM paturētajiem mērījumiem, parāda kļūdas, saglabā .npy.

    python camera/calibrate_T_base_cam.py                        # A2 dēlis -> T_base_cam.npy
    python camera/calibrate_T_base_cam.py --board home50 --output rosbags/<batch>/T_base_cam.npy
    python camera/calibrate_T_base_cam.py --rosbag 20261119_200126_board_calib_final --board a2

Vairāki mērījumi (run): pēc katra ievadīšanas skripts izdrukā dēļa reprojekcijas kļūdu un kopējā
novērtējuma atlikumus (mm un px, ar RMS), un jautā:
    d - izmest šo mērījumu un mēģināt vēlreiz      c - paturēt un turpināt ar nākamo
    f - paturēt un beigt                            x - izmest un beigt (no jau paturētajiem)
('f', nevis 's': S jau nozīmē "ņem šo kadru" 1. solī.) Starp mērījumiem dēli pārliek citur,
plakaniski; kamera paliek uz vietas. Katrs mērījums sākas ar svaigu straumi (pipeline restartē).

Ar --rosbag 1. solis notiek ierakstā (.db3, atskaņojas pa apli no sākuma): S - paņem šo kadru,
SPACE - vidējo nākamos --frames kadrus. Dēlis un kamera NEDRĪKST būt kustināti kopš ieraksta,
jo robota punkti (2. solis) tiek mērīti uz īstā dēļa. Vairākām dēļa vietām vajag ierakstu, kurā
dēlis tiek pārlikts, un S katrā vietā.

RobotStudio: pēc S vai SPACE izvēlētais kadrs (ar asīm un P0 P1 P2) tiek uzlikts uz grīdas un
nosūtīts kā BoardOverlay tekstūra (robot/station_feed.py, add-in robot/rapid/TrashPickerStation),
ar TUKŠU punktu mākoni, tāpēc LiveCloud paliek tukšs līdz nākamajam robot_trash_picker.py kadram.
1. mērījumā T_base_cam vēl nav zināms, tāpēc dēlis tiek novietots pats: plakaniski uz z = 0,
P0->P1 gar +X, un tā, lai kamera būtu --camera-at X,Y (izmērīts no krusta uz grīdas). No 2.
mērījuma attēlu novieto ar līdzšinējo novērtējumu, tāpēc dēli var likt jebkur un jebkurā virzienā
(plakaniski). Logs paliek atvērts, kamēr ievadīti punkti. --no-station neko nesūta.

    python camera/calibrate_T_base_cam.py --board a2 --camera-at 670,-5,1180

Dēļu parametri ir camera/charuco.py (BOARDS). Ja BoardOverlay RobotStudio tiek pārvietots,
jāmaina arī kods: robot/rapid/TrashPickerStation/BOARD_OVERLAY_MOVE.md.
Apraksts citam aģentam: camera/CALIBRATE_T_BASE_CAM_RUNS.md.
"""

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pyrealsense2 as rs
import numpy as np
import cv2

try:
    import msvcrt       # Windows: taustiņi no termināļa
except ImportError:
    msvcrt = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camera.charuco import (BOARDS, detect_pose, make_board,  # noqa: E402
                            reference_distances_mm, reference_points)
from robot.board_demo_calibration import (DEFAULT_BOARD_AT_MM, MAX_RMS_PX, MIN_FRAMES,  # noqa: E402
                                          _xyz, board_in_base, combine_poses, compose)
from robot.floor_overlay import (FLOOR_X_MM, FLOOR_Y_MM, MM_PER_PX, floor_picture,  # noqa: E402
                                 for_robotstudio)
from robot.frame_selection import resolve_bag  # noqa: E402
from robot.station_feed import PART_NAME, StationFeed, build_request, station_dir, write_cloud  # noqa: E402

MOVE_DOC = "robot/rapid/TrashPickerStation/BOARD_OVERLAY_MOVE.md"
OVERLAY_CENTRE_MM = ((FLOOR_X_MM[0] + FLOOR_X_MM[1]) / 2, (FLOOR_Y_MM[0] + FLOOR_Y_MM[1]) / 2)
MAX_CAMERA_OFF_CENTRE_MM = 20.0     # --camera-at tālāk no BoardOverlay centra = tas nav pārvietots
CALIBRATION_FLOOR_FILE = "calibration_floor.png"    # ne station_floor.png: tas ir robot_trash_picker.py
CALIBRATION_CLOUD_FILE = "calibration_cloud.bin"
BOARD_POINT_BGR = (255, 0, 255)
CAMERA_BGR = (0, 255, 255)
LIVE_WARMUP_FRAMES = 5              # pēc restarta: pirmie kadri vēl pielāgo ekspozīciju
DECISION_KEYS = {"d": "discard", "c": "continue", "f": "finish", "x": "discard-finish"}

MAX_DISTANCE_ERROR_MM = 15.0    # P0-P1 / P0-P2 pret dēli: vairāk = sajaukti punkti
MAX_MEAN_ERROR_MM = 5.0         # Kabsch atlikums: vairāk = TCP nebija uz stūra vai dēlis kustējās

_parser = argparse.ArgumentParser(description="Camera -> robot calibration with a ChArUco board.")
_parser.add_argument("--board", choices=sorted(BOARDS), default="a2",
                     help="which board lies in the work area (default: a2, the lab board)")
_parser.add_argument("--output", default="T_base_cam.npy",
                     help="where to save T_base_cam (default: T_base_cam.npy in the current folder)")
_parser.add_argument("--rosbag", default=None, metavar="BAG.db3",
                     help="play a RealSense recording instead of the live camera; S takes the "
                          "current frame (a bare name is looked up in rosbags/). Each run replays it "
                          "from the start: for several board positions, record the board being moved")
_parser.add_argument("--frames", type=int, default=30,
                     help="how many frames after SPACE the board pose is averaged over (default: 30)")
_parser.add_argument("--force", action="store_true",
                     help="save even when the distance or error checks fail")
_parser.add_argument("--camera-at", type=_xyz, default=None, metavar="X,Y,Z",
                     help="the camera centre measured from the cross on the floor, mm: places the "
                          "first run's frame on BoardOverlay under it (default: board centre at "
                          f"{DEFAULT_BOARD_AT_MM[0]:.0f},{DEFAULT_BOARD_AT_MM[1]:.0f})")
_parser.add_argument("--no-station", action="store_true",
                     help="do not send the chosen frame to RobotStudio's BoardOverlay")
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
if not _args.no_station:
    print(f"RobotStudio: {PART_NAME} pienemts ar centru X {OVERLAY_CENTRE_MM[0]:.0f}, "
          f"Y {OVERLAY_CENTRE_MM[1]:.0f} mm (FLOOR_X_MM / FLOOR_Y_MM robot/floor_overlay.py).")
    print(f"UZMANIBU: ja {PART_NAME} RobotStudio tiek parvietots, jamaina ari kods: {MOVE_DOC}")
    if _args.camera_at is None:
        print(f"BRIDINAJUMS: nav --camera-at, dela centrs tiek likts X {DEFAULT_BOARD_AT_MM[0]:.0f}, "
              f"Y {DEFAULT_BOARD_AT_MM[1]:.0f} mm, nevis zem kameras.")
    else:
        _off = np.hypot(_args.camera_at[0] - OVERLAY_CENTRE_MM[0],
                        _args.camera_at[1] - OVERLAY_CENTRE_MM[1])
        if _off > MAX_CAMERA_OFF_CENTRE_MM:
            print(f"BRIDINAJUMS: kamera X {_args.camera_at[0]:.0f}, Y {_args.camera_at[1]:.0f} ir "
                  f"{_off:.0f} mm no {PART_NAME} centra. Parvieto {PART_NAME} zem kameras un "
                  f"izmaini kodu, ka aprakstits {MOVE_DOC}.")

# ====================================
# RealSense
# ====================================

pipeline = rs.pipeline()
config = rs.config()
BAG = None if _args.rosbag is None else resolve_bag(_args.rosbag)
if BAG is None:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
else:
    # Ieraksts nosaka izšķirtspēju; atskaņo pa apli reālā laikā, līdz nospiež S vai SPACE.
    if not BAG.exists():
        raise SystemExit(f"no such rosbag: {BAG}")
    try:
        config.enable_device_from_file(str(BAG), repeat_playback=True)
    except RuntimeError as error:
        raise SystemExit(f"pyrealsense2 could not open {BAG}: {error}\n"
                         f"These must be librealsense's own bags (topics /device_0/sensor_*/...).") from error
    print(f"Ieraksts: {BAG}")
profile = pipeline.start(config)

color_profile = profile.get_stream(rs.stream.color)
intr = color_profile.as_video_stream_profile().get_intrinsics()
pipeline.stop()                     # katrs mērījums palaiž savu, svaigu straumi (start_stream)

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


WINDOW = "Kalibracija"
NO_KEY = 0xFF


@dataclass
class Run:
    """Viens mērījums: dēļa poza kamerā, tās kadrs un robota punkti P0 P1 P2."""
    number: int
    T_cam_board: np.ndarray
    image: np.ndarray
    rms_px: list                        # dēļa reprojekcijas kļūda (px) katram izmantotajam kadram
    spread: tuple | None = None         # SPACE: (mm, grad) starp kadriem
    points_robot: np.ndarray = field(default=None)     # 3 x 3, metros

    @property
    def points_cam(self):
        return transform_points(self.T_cam_board, ref_points_board)


def read_key(delay_ms=1):
    """Nospiestais taustiņš no OpenCV loga VAI no termināļa, NO_KEY ja nav.

    Windows neļauj no termināļa palaistam procesam paņemt fokusu, tāpēc logs bieži atveras
    bez tā un cv2.waitKey taustiņus nedabū - tie aiziet uz termināli. Tur tos arī nolasām.
    """
    key = cv2.waitKey(delay_ms) & 0xFF
    if key != NO_KEY or msvcrt is None or not msvcrt.kbhit():
        return key
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):      # bultas/F taustiņi: otrais kods jāizlasa un jāizmet
        msvcrt.getwch()
        return NO_KEY
    return ord(ch) if ord(ch) < NO_KEY else NO_KEY


def drain_keys():
    """Izmet taustiņus, kas nospiesti pirms šī brīža (logā vai terminālī)."""
    for _ in range(50):
        if cv2.waitKey(1) & 0xFF == NO_KEY:
            break
    while msvcrt is not None and msvcrt.kbhit():
        msvcrt.getwch()


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


def project(points_cam):
    """Nx3 punkti kameras rāmī (m) -> Nx2 pikseļi, ar tiem pašiem K un kropļojumu kā poza."""
    pixels, _ = cv2.projectPoints(np.asarray(points_cam, dtype=np.float64).reshape(-1, 1, 3),
                                  np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    return pixels.reshape(-1, 2)


@dataclass
class Fit:
    T_base_cam: np.ndarray
    residual_mm: np.ndarray             # N: |T p_cam - p_robot|
    reprojection_px: np.ndarray         # N: |proj(T^-1 p_robot) - proj(p_cam)|
    labels: list                        # N: "r2 P0"

    @staticmethod
    def rms(values):
        return float(np.sqrt(np.mean(np.square(values))))


def fit(runs):
    """T_base_cam ar Kabsch no visu mērījumu punktiem, un katra punkta atlikumi."""
    cam = np.vstack([run.points_cam for run in runs])
    robot = np.vstack([run.points_robot for run in runs])
    T = kabsch(cam, robot)
    residual = np.linalg.norm(transform_points(T, cam) - robot, axis=1) * 1000.0
    # Reprojekcija: kur T liek robota punktu attēlā, pret kur dēļa poza liek to pašu stūri.
    robot_in_cam = transform_points(np.linalg.inv(T), robot)
    reprojection = np.linalg.norm(project(robot_in_cam) - project(cam), axis=1)
    # Aiz kameras (z <= 0) projekcija ir spoguļota un maldinoša: tāds punkts ir rupja kļūda.
    reprojection[robot_in_cam[:, 2] <= 0.0] = np.inf
    labels = [f"r{run.number} P{i}" for run in runs for i in range(3)]
    return Fit(T, residual, reprojection, labels)


def distances_mm(points_robot):
    """(|P0 P1|, |P0 P2|, vai atbilst dēlim) ievadītajiem punktiem."""
    d01 = np.linalg.norm(points_robot[1] - points_robot[0]) * 1000
    d02 = np.linalg.norm(points_robot[2] - points_robot[0]) * 1000
    ok = abs(d01 - EXPECTED_D01_MM) <= MAX_DISTANCE_ERROR_MM and abs(d02 - EXPECTED_D02_MM) <= MAX_DISTANCE_ERROR_MM
    return d01, d02, ok


def annotate(image, T_cam_board):
    """Galīgais kadrs: dēļa asis un P0 P1 P2 no fiksētās pozas (S - šī kadra, SPACE - vidējā)."""
    picture = image.copy()
    rvec, _ = cv2.Rodrigues(T_cam_board[:3, :3])
    tvec = T_cam_board[:3, 3]
    cv2.drawFrameAxes(picture, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
    proj, _ = cv2.projectPoints(ref_points_board, rvec, tvec, camera_matrix, dist_coeffs)
    for i, p in enumerate(proj.reshape(-1, 2)):
        pt = (int(p[0]), int(p[1]))
        cv2.circle(picture, pt, 8, (0, 0, 255), -1)
        cv2.putText(picture, f"P{i}", (pt[0] + 10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
    return picture


def place_board(T_cam_board, camera_at_mm):
    """(T_base_board, T_base_cam) RobotStudio attēlam, pirms īstais T_base_cam ir zināms.

    Dēlis guļ plakaniski uz z = 0, P0->P1 gar +X (yaw 0). Kameras X, Y attiecībā pret dēli nemainās,
    kur dēli novieto, tāpēc dēli pabīda tā, lai kamera nonāk izmērītajā X, Y. Bez mērījuma dēļa
    centrs ir DEFAULT_BOARD_AT_MM, kā robot/board_demo_calibration.py.
    """
    if camera_at_mm is None:
        T_base_board = board_in_base(BOARD_SPEC, DEFAULT_BOARD_AT_MM, 0.0)
    else:
        camera_from_board = compose(board_in_base(BOARD_SPEC, (0.0, 0.0), 0.0), T_cam_board)[:2, 3]
        at_mm = np.array(camera_at_mm[:2]) - camera_from_board * 1000.0
        T_base_board = board_in_base(BOARD_SPEC, at_mm, 0.0)
    return T_base_board, compose(T_base_board, T_cam_board)


def turned_px(x_mm, y_mm):
    """(kolonna, rinda) grīdas punktam for_robotstudio pagrieztajā attēlā: +X pa labi, +Y uz augšu.

    Kolonna c satur X = Xmin + (c + 0.5) mm, rinda r satur Y = Ymax - (r + 0.5) mm (station_feed.floor_uv).
    """
    return (int(round((x_mm - FLOOR_X_MM[0]) / MM_PER_PX - 0.5)),
            int(round((FLOOR_Y_MM[1] - y_mm) / MM_PER_PX - 0.5)))


def draw_marks_turned(picture, marks):
    """Atzīmes uz JAU pagrieztā attēla, lai uzraksti RobotStudio būtu lasāmi, nevis sāniski."""
    for label, x, y, colour in marks:
        centre = turned_px(x, y)
        cv2.circle(picture, centre, 10, colour, 2, cv2.LINE_AA)
        cv2.drawMarker(picture, centre, colour, cv2.MARKER_CROSS, 14, 2)
        cv2.putText(picture, label, (centre[0] + 12, centre[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, colour, 2, cv2.LINE_AA)
    return picture


def push_to_station(annotated, T_cam_board, frame_index, T_estimate=None):
    """Galīgais kadrs uz BoardOverlay, ar tukšu mākoni. StationFeed (sūta fonā) vai None.

    Bez T_estimate (1. mērījums) dēli novieto place_board; ar to - kur novērtējums to liek, tāpēc
    dēlis var būt jebkur un jebkurā virzienā. Neizdošanās ir tikai brīdinājums.
    """
    try:
        if T_estimate is None:
            T_base_board, T_overlay = place_board(T_cam_board, _args.camera_at)
            how = "pec dela (P0->P1 gar +X), kamera --camera-at vieta"
        else:
            T_overlay = T_estimate
            T_base_board = T_overlay @ T_cam_board
            how = "ar lidzsinejo novertejumu no paturetajiem merijumiem"
        corners = transform_points(T_base_board, ref_points_board) * 1000.0
        camera_mm = T_overlay[:3, 3] * 1000.0
        marks = [(f"P{i} ({p[0]:.0f}, {p[1]:.0f})", p[0], p[1], BOARD_POINT_BGR)
                 for i, p in enumerate(corners)]
        marks.append((f"camera ({camera_mm[0]:.0f}, {camera_mm[1]:.0f})", camera_mm[0], camera_mm[1],
                      CAMERA_BGR))
        print()
        print(f"RobotStudio {PART_NAME}: attels novietots {how} (mm, robota baze):")
        for i, p in enumerate(corners):
            print(f"  P{i}: X={p[0]:.1f}  Y={p[1]:.1f}  Z={p[2]:.1f}")
        print(f"  kamera: X={camera_mm[0]:.1f}  Y={camera_mm[1]:.1f}  Z={camera_mm[2]:.1f}")
        if _args.camera_at is not None:
            print(f"  izmeritais kamera X {_args.camera_at[0]:.1f}  Y {_args.camera_at[1]:.1f}  "
                  f"Z {_args.camera_at[2]:.1f}: starpiba Z {camera_mm[2] - _args.camera_at[2]:+.1f} mm")
        outside = [f"P{i}" for i, p in enumerate(corners)
                   if not (FLOOR_X_MM[0] <= p[0] <= FLOOR_X_MM[1] and FLOOR_Y_MM[0] <= p[1] <= FLOOR_Y_MM[1])]
        if outside:
            print(f"BRIDINAJUMS: {' '.join(outside)} ir arpus {PART_NAME} (X {FLOOR_X_MM[0]:.0f}..{FLOOR_X_MM[1]:.0f}, "
                  f"Y {FLOOR_Y_MM[0]:.0f}..{FLOOR_Y_MM[1]:.0f} mm), RobotStudio tos nerada. Pabida deli "
                  f"tuvak kameras vidum, vai parvieto {PART_NAME} ({MOVE_DOC}).")
        picture = draw_marks_turned(for_robotstudio(floor_picture(annotated, T_overlay, intr)), marks)
        out_dir = station_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        floor_path, cloud_path = out_dir / CALIBRATION_FLOOR_FILE, out_dir / CALIBRATION_CLOUD_FILE
        if not cv2.imwrite(str(floor_path), picture):
            raise OSError(f"could not write {floor_path}")
        # Tikai tekstūra: add-in "show" vienmēr nomaina arī LiveCloud, tāpēc mākonis ir tukšs.
        write_cloud(cloud_path, np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8))
    except Exception as failure:  # noqa: BLE001 - tikai attēls
        print(f"BRIDINAJUMS: kadrs netika nosūtīts uz RobotStudio: {failure}")
        return None
    feed = StationFeed(np.eye(4))
    # floor_visible: robot_trash_picker.py --hide-rgb paslēpj BoardOverlay, un add-in to atceras.
    feed.send(build_request(frame_index, floor_path, cloud_path, 0, floor_visible=True))
    return feed


class _WindowPump:
    """StationFeed.wait(keys=...) sauc keys.poll(ms): te tas tikai uztur logu dzīvu."""

    @staticmethod
    def poll(delay_ms=1):
        cv2.waitKey(delay_ms)


def ask_points():
    """P0 P1 P2 robota koordinātas metros, no termināļa."""
    points = np.zeros((3, 3))
    for i in range(3):
        while True:
            try:
                raw = input(f"P{i} robota koordinatas 'x,y,z' mm: ")
                x, y, z = [float(v.strip()) for v in raw.split(",")]
                points[i] = [x / 1000.0, y / 1000.0, z / 1000.0]
                break
            except ValueError:
                print("Nepareizs formats. Piemers: 412.5, -87.3, 12.0")
    return points


def ask_points_with_window(picture):
    """ask_points, kamēr logs rāda galīgo kadru. Logs paliek atvērts arī pēc tam (lēmumam).

    input() bloķē, tāpēc tas iet citā pavedienā, un šis turpina cv2.waitKey: citādi Windows logu
    atzīmē kā 'Not Responding'.
    """
    result = {}

    def ask():
        try:
            result["points"] = ask_points()
        except BaseException as failure:  # noqa: BLE001 - EOF, Ctrl+C: nodod galvenajam pavedienam
            result["error"] = failure

    cv2.imshow(WINDOW, picture)
    worker = threading.Thread(target=ask, name="calibration-points", daemon=True)
    worker.start()
    while worker.is_alive():
        cv2.waitKey(30)
    if "error" in result:
        raise result["error"]
    return result["points"]


# ====================================
# 1. SOLIS: dēļa poza no kameras (katram mērījumam)
# ====================================

def start_stream():
    """Svaiga straume katram mērījumam: neviens kadrs no 2. soļa laika netiek paņemts."""
    pipeline.start(config)
    if BAG is None:
        for _ in range(LIVE_WARMUP_FRAMES):
            pipeline.wait_for_frames()


def capture_run(number):
    """1. solis: dzīva straume, līdz S (ieraksts) vai SPACE. Run bez robota punktiem. ESC iziet."""
    print()
    print("=" * 50)
    print(f"1. SOLIS (merijums {number}): dela pozas noteiksana")
    print("Novieto deli darba zona, plakaniski. NEKUSTINI to, kamer ievadi punktus!")
    if BAG is None:
        print("SPACE - fiksēt pozu | ESC - iziet, NEKO nesaglabājot (beigt ar paturētajiem: 'x' pēc mērījuma)")
    else:
        print("S - fiksēt pozu no šī kadra | SPACE - vidējot nākamos kadrus | ESC - iziet, NEKO nesaglabājot")
    print("Taustiņi strādā gan kalibrācijas logā, gan šajā terminālī.")
    print("=" * 50)

    start_stream()
    drain_keys()                    # taustiņi no 2. soļa / lēmuma nedrīkst fiksēt pozu uzreiz
    # Logs virs visiem citiem, lai to neaizsedz terminālis (fokusu Windows tam tik un tā nedod).
    cv2.namedWindow(WINDOW)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
    global frame_index
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame_index += 1

            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            charuco_corners, charuco_ids, marker_corners, marker_ids = \
                charuco_detector.detectBoard(gray)

            display = frame.copy()
            pose_ok = False

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

            if not pose_ok:
                status = "Mekle deli..."
            elif BAG is None:
                status = "POZA OK - spied SPACE"
            else:
                status = "POZA OK - S: so kadru | SPACE: videjot"
            cv2.putText(display, f"RUN {number} - LIVE | {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if pose_ok else (0, 0, 255), 2)

            cv2.imshow(WINDOW, display)
            key = read_key()

            if key == 27:
                lost = f" ({number - 1} paturetie merijumi izmesti; 'x' pec merijuma tos butu izmantojis)" \
                    if number > 1 else ""
                print(f"Partraukts. Nekas netika saglabats{lost}.")
                raise SystemExit(1)

            if key in (ord("s"), ord("S")) and pose_ok and BAG is not None:
                # Atskaņošana apstājas, kalibrācijai iet tieši redzamais kadrs.
                pose = detect_pose(gray, camera_matrix, dist_coeffs, board, charuco_detector)
                if pose is None or pose.rms_px > MAX_RMS_PX:
                    rms = "nav pozas" if pose is None else f"kluda {pose.rms_px:.2f} px"
                    print(f"Sis kadrs neder ({rms}, jabut <= {MAX_RMS_PX} px). Spied S uz cita kadra.")
                    continue
                print(f"Dela poza fikseta no viena kadra, reprojekcijas kluda {pose.rms_px:.2f} px.")
                return Run(number, pose.T_cam_board, frame.copy(), [pose.rms_px])

            if key == 32 and pose_ok:
                # Viens kadrs ir trokšņains: vidējo pozu no nākamajiem kadriem, tikai ar labu
                # reprojekcijas kļūdu (tas pats, ko dara board_demo_calibration.py).
                print(f"Vacu {_args.frames} kadrus, nekustini deli un kameru...")
                poses, rms_values, tried, image = [], [], 0, None
                while tried < _args.frames:
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    tried += 1
                    frame_index += 1
                    color = np.asanyarray(color_frame.get_data()).copy()
                    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                    pose = detect_pose(gray, camera_matrix, dist_coeffs, board, charuco_detector)
                    if pose is not None and pose.rms_px <= MAX_RMS_PX:
                        poses.append(pose.T_cam_board)
                        rms_values.append(pose.rms_px)
                        image = color      # pedejais labais kadrs: RobotStudio grida attelam
                if len(poses) < MIN_FRAMES:
                    print(f"Tikai {len(poses)} no {tried} kadriem ar kludu <= {MAX_RMS_PX} px. "
                          f"Uzlabo apgaismojumu vai pieliec kameru tuvak, un spied SPACE velreiz.")
                    continue
                try:
                    T_cam_board, spread_mm, spread_deg = combine_poses(poses)
                except SystemExit as moved:     # kadri neatbilst vienai pozai: neizmet paturētos
                    print(f"{moved} Spied SPACE velreiz.")
                    continue
                print(f"Dela poza fikseta: {len(poses)} no {tried} kadriem, izkliede "
                      f"{spread_mm:.1f} mm / {spread_deg:.2f} grad.")
                return Run(number, T_cam_board, image, rms_values, (spread_mm, spread_deg))
    finally:
        pipeline.stop()
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 0)


# ====================================
# 2. SOLIS: robota koordinātas (katram mērījumam)
# ====================================

def touch_points(run, T_estimate):
    """Kadrs uz BoardOverlay, tad P0 P1 P2 no termināļa, kamēr logs rāda galīgo kadru."""
    picture = annotate(run.image, run.T_cam_board)
    cv2.imshow(WINDOW, picture)
    cv2.waitKey(1)
    station = None if _args.no_station else push_to_station(picture, run.T_cam_board, frame_index,
                                                            T_estimate)
    if station is not None:
        # Lai RobotStudio atbilde neielaužas P0 jautājumā; logs tikmēr tiek uzturēts.
        station.wait(keys=_WindowPump(), quiet=True)

    print()
    print("Atsauces punkti kameras rami (mm):")
    for i, p in enumerate(run.points_cam):
        print(f"  P{i}: X={p[0]*1000:.1f}  Y={p[1]*1000:.1f}  Z={p[2]*1000:.1f}")
    print()
    print("=" * 50)
    print(f"2. SOLIS (merijums {run.number}): robota koordinatas")
    print("Ar jogging aizved TCP smaili PRECIZI uz katru punktu")
    print("(sarkanie punkti ekrana: P0, P1, P2).")
    if station is not None:
        print(f"RobotStudio: {PART_NAME} rada so kadru ar P0 P1 P2 (rozi, X/Y augstak). Jogo tur")
        print("pirkstu galus (toolRG6 TCP) uz katru punktu un nolasi TCP poziciju.")
    print("Nolasi robota poziciju (mm, Base frame) un ievadi seit.")
    print("SVARIGI: aktivajam rikam jabut Tool = toolRG6 (pirkstu gali, [0,0,253]) un")
    print("koordinatu sistemai Base (vai World ar wobj0). Ar tool0 punkti bus 253 mm nost.")
    if BAG is not None:
        print(f"IERAKSTS: delim jaatrodas tiesi tur, kur {BAG.name}, un kamera nav kustinata.")
    print("Logs paliek atverts, kamer ievaditi visi tris punkti.")
    print("=" * 50)
    points = ask_points_with_window(picture)
    if station is not None:
        station.wait(quiet=True)
    return points


# ====================================
# Kļūdas, lēmums, padomi
# ====================================

def report(kept, current):
    """Šī mērījuma un kopējā novērtējuma (paturētie + šis) kļūdas. Fit."""
    print()
    print("=" * 50)
    print(f"MERIJUMS {current.number}: kludas")
    print("=" * 50)
    rms = np.asarray(current.rms_px)
    if len(rms) == 1:
        print(f"Dela reprojekcijas kluda (poza attela): {rms[0]:.2f} px")
    else:
        print(f"Dela reprojekcijas kluda (poza attela): RMS {Fit.rms(rms):.2f} px, mediana "
              f"{np.median(rms):.2f} px, sliktaka {rms.max():.2f} px ({len(rms)} kadri), izkliede "
              f"{current.spread[0]:.1f} mm / {current.spread[1]:.2f} grad")
    d01, d02, ok = distances_mm(current.points_robot)
    print(f"Parbaude: P0-P1 = {d01:.1f} mm (jabut ~{EXPECTED_D01_MM:.0f}), "
          f"P0-P2 = {d02:.1f} mm (jabut ~{EXPECTED_D02_MM:.0f})" + ("" if ok else "  <- NEATBILST"))

    result = fit(kept + [current])
    print()
    print(f"Kopejais novertejums no {len(kept) + 1} merijumiem ({3 * (len(kept) + 1)} punkti),"
          f" * = sis merijums:")
    print("  punkts     atlikums mm   reprojekcija px")
    for label, mm, px in zip(result.labels, result.residual_mm, result.reprojection_px):
        mark = "*" if label.startswith(f"r{current.number} ") else " "
        print(f" {mark}{label:9s} {mm:9.2f}     {px:9.2f}")
    print(f"  RMS        {Fit.rms(result.residual_mm):9.2f}     {Fit.rms(result.reprojection_px):9.2f}")
    print(f"  sliktakais {result.residual_mm.max():9.2f}     {result.reprojection_px.max():9.2f}")
    print(f"  videjais   {result.residual_mm.mean():9.2f}     {result.reprojection_px.mean():9.2f}")
    if not kept:
        print("  (viens merijums: 3 punkti nosaka 6 brivibas pakapes, atlikumi pieradi maz)")
    return result


def ask_decision():
    """d / c / f / x no loga vai termināļa."""
    print()
    print("d - izmest so merijumu un merit velreiz | c - paturet un turpinat ar nakamo")
    print("f - paturet un beigt                    | x - izmest un beigt (no jau paturetajiem)")
    drain_keys()
    while True:
        key = read_key(30)
        if key != NO_KEY and chr(key).lower() in DECISION_KEYS:
            choice = chr(key).lower()
            print(f"-> {choice}")
            return choice


def print_placement_ideas():
    """Kur likt dēli nākamajā mērījumā. Vienmēr tas pats teksts."""
    print()
    print("KUR LIKT DELI NAKAMAJA MERIJUMA (kamera paliek uz vietas):")
    print("  - delis guļ PLAKANISKI uz gridas (z = 0), neko zem ta nepaliekot")
    print("  - viss delis redzams kamera, neaizsegts, labi apgaismots")
    print("  - izkliede pa redzeslauku: centrs, tad pie katra stura un malam")
    print("  - pagriez deli par ~45 vai ~90 grad starp merijumiem")
    print("  - neatkarto jau izmantotu vietu")
    print(f"  - turi to {PART_NAME} robezas (X {FLOOR_X_MM[0]:.0f}..{FLOOR_X_MM[1]:.0f}, "
          f"Y {FLOOR_Y_MM[0]:.0f}..{FLOOR_Y_MM[1]:.0f} mm), lai RobotStudio to rada")
    print("  - visi tris punkti robota sniedzamiba")
    print("  - 4-6 merijumi parasti pietiek")


# ====================================
# Mērījumi
# ====================================

frame_index = 0
kept_runs = []
try:
    while True:
        if kept_runs and not _args.no_station:
            print()
            print(f"UZMANIBU: no si merijuma {PART_NAME} attels tiek novietots ar novertejumu no "
                  f"{len(kept_runs)} paturetajiem merijumiem, nevis ka 1. merijuma.")
            print(f"  Tas RobotStudio var but parvietojies vai pagriezies. Aizskar P0 P1 P2 tur, "
                  f"kur tie ir TAGAD.")
        run = capture_run(len(kept_runs) + 1)
        T_estimate = fit(kept_runs).T_base_cam if kept_runs else None
        run.points_robot = touch_points(run, T_estimate)
        report(kept_runs, run)
        choice = ask_decision()
        if choice == "d":
            print(f"Merijums {run.number} izmests. Atkarto.")
            continue
        if choice in ("c", "f"):
            kept_runs.append(run)
        if choice == "c":
            print_placement_ideas()
            continue
        break
finally:
    cv2.destroyAllWindows()

# ====================================
# 3. SOLIS: galīgais T_base_cam aprēķins
# ====================================

if not kept_runs:
    print()
    print("Nav neviena patureta merijuma. NESAGLABATS.")
    raise SystemExit(1)

result = fit(kept_runs)
T_base_cam = result.T_base_cam

print()
print("=" * 50)
print(f"REZULTATS ({len(kept_runs)} merijumi, {3 * len(kept_runs)} punkti)")
print("=" * 50)
print("T_base_cam:")
print(T_base_cam)
print()
print("Atbilstibas kluda katra punkta (mm) un reprojekcija (px):")
for label, mm, px in zip(result.labels, result.residual_mm, result.reprojection_px):
    print(f"  {label}: {mm:.2f} mm   {px:.2f} px")
print(f"Videja kluda: {result.residual_mm.mean():.2f} mm   RMS {Fit.rms(result.residual_mm):.2f} mm   "
      f"reprojekcijas RMS {Fit.rms(result.reprojection_px):.2f} px")

problems = []
bad_runs = [run.number for run in kept_runs if not distances_mm(run.points_robot)[2]]
if bad_runs:
    problems.append(f"attalumi neatbilst delim merijumos {bad_runs} - punkti var but sajaukti vai "
                    f"nepareizs --board")
    print(f"BRIDINAJUMS: {problems[-1]}!")

if result.residual_mm.mean() > MAX_MEAN_ERROR_MM:
    problems.append(f"videja kluda {result.residual_mm.mean():.1f} mm > {MAX_MEAN_ERROR_MM:.0f} mm")
    print()
    print(f"BRIDINAJUMS: kluda > {MAX_MEAN_ERROR_MM:.0f} mm. Iespejamie iemesli:")
    print("  - TCP nebija precizi uz stura (vai nebija toolRG6)")
    print("  - delis pakustejas starp 1. un 2. soli")
    print("  - nepareizi dela parametri (--board?)")
    print("Iesaku atkartot, vai izmest merijumu ar lielako atlikumu.")

if problems and not _args.force:
    print()
    print("NESAGLABATS: " + "; ".join(problems) + ".")
    print("Atkarto merijumus, vai ar --force saglaba tik un ta.")
    raise SystemExit(1)

Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT_FILE, T_base_cam)
print()
print(f"Saglabats: {OUTPUT_FILE}")

# Kalibracijas kadrs uz gridas RobotStudio virsmai (pedeja merijuma kadrs): rozā = kur T_base_cam
# liek dela punktus, zaļš = kur TCP tos aizskara, visiem paturetajiem merijumiem.
# Ja attels neizdodas, kalibracija tik un ta ir saglabata.
try:
    from robot.floor_overlay import write_floor_overlay
    _marks = []
    for _run in kept_runs:
        _fitted = transform_points(T_base_cam, _run.points_cam)
        _marks += [(f"r{_run.number} P{i}", p[0] * 1000, p[1] * 1000, (255, 0, 255))
                   for i, p in enumerate(_fitted)]
        _marks += [(f"r{_run.number} P{i} TCP", p[0] * 1000, p[1] * 1000, (0, 255, 0))
                   for i, p in enumerate(_run.points_robot)]
    _floor = write_floor_overlay(kept_runs[-1].image, T_base_cam, intr, points=_marks,
                                 label="calibration")
    print(f"RobotStudio grida attels: {_floor}")
except Exception as _failure:  # noqa: BLE001 - tikai attels
    print(f"BRIDINAJUMS: RobotStudio grida attels netika izveidots: {_failure}")
print()
print("Lietosana:")
print(f"  python robot/robot_trash_picker.py --host 192.168.125.1 --calibration {OUTPUT_FILE}")
print("  (T ir METROS: p_robot_mm = (T @ [X_m, Y_m, Z_m, 1])[:3] * 1000)")
