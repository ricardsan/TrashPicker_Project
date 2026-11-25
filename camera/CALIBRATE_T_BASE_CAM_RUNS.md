# Several estimation runs in `calibrate_T_base_cam.py` (instructions for a Claude agent)

This file describes the multi-run calibration in `camera/calibrate_T_base_cam.py` in enough detail
to rebuild it from an empty context, or to check that it still behaves this way. Read the script's
module docstring and `robot/rapid/TrashPickerStation/STATION_SETUP.md` (section *Calibration*)
first. The station side (BoardOverlay, the add-in, the empty cloud) is explained there.

## What it does

The script fits `T_base_cam` (camera frame → robot base frame, **metres**, 4×4) from a ChArUco
board, from **several runs**. The camera stays still; the board is moved, flat on the floor, between
runs. One run is:

1. **Capture** (`capture_run(number)`).
   - A fresh stream is started; see *Fresh stream* below.
   - Live view in the window `Kalibracija`, with the status line `RUN n - LIVE | ...`.
   - **S** (clip only) fixes the pose from the frame on screen.
   - **SPACE** averages the next `--frames` frames, using only frames with board RMS ≤ 1 px
     (`MAX_RMS_PX`), through `combine_poses`.
   - **ESC** quits and saves nothing. The message says how many kept runs are dropped and that `x`
     would have kept them.
   - It returns a `Run`: `T_cam_board`, the frame, and each frame's board RMS (`rms_px`). SPACE
     also gives the spread (mm, degrees).
2. **Touch** (`touch_points(run, T_estimate)`).
   - The final annotated frame (`annotate`) goes to BoardOverlay with an **empty cloud**
     (`push_to_station`).
   - The script waits for the add-in's reply while keeping the window painted (`_WindowPump`).
   - It then asks for P0, P1 and P2 (`x, y, z` in mm, the TCP of toolRG6 = the fingertips) on a
     helper thread. The main thread keeps calling `cv2.waitKey(30)`, so the window never goes "Not
     Responding".
3. **Report** (`report(kept, current)`). See *Errors printed* below.
4. **Decision** (`ask_decision()`): one key, read from the window or the terminal (`read_key`).
   Keys typed earlier are thrown away first (`drain_keys`).

| key | meaning |
| --- | --- |
| `d` | discard this run and capture it again (same run number) |
| `c` | keep it and continue; prints the board-placement ideas |
| `f` | keep it and finish |
| `x` | discard it and finish with the runs kept so far; none kept → nothing saved, exit 1 |

**Why `f` and not `s` for "keep and finish":** `s` already means "take this frame" in the capture
step on a clip. A stray or doubled S would then end the calibration. The user agreed to `f`.

After `f`, or `x` with at least one kept run:
- a final fit over all kept runs, and the same report;
- the checks;
- `np.save(--output)`;
- `robot/robotstudio/live/floor.png`, made from the last run's frame, with every kept point
  labelled `rN Pi` (magenta: fitted; green: typed).

Saving is refused, unless `--force`, if any kept run fails the distance check or the **mean** 3-D
residual is above 5 mm (`MAX_MEAN_ERROR_MM`).

## The mathematics

Notation: `P_board` = the three reference corners in the board frame (`camera.charuco.reference_points`,
metres). P1 is +x from P0, P2 is +y from P0. For a2 these are 240 mm and 300 mm.

- **Points in the camera:** `p_cam = T_cam_board · P_board` (`Run.points_cam`, metres).
- **Points typed:** `p_robot` in mm, stored /1000 as metres.
- **Fit** (`fit(runs)`): stack the points of every run (3·N rows) and solve **Kabsch**:
  - `A = p_cam`, `B = p_robot`; `cA`, `cB` are their centroids;
  - `H = (A − cA)ᵀ (B − cB)`; `U S Vᵀ = svd(H)`;
  - `R = V Uᵀ`. If `det R < 0`, negate the last row of `Vᵀ` and recompute, so `R` is a rotation,
    never a mirror;
  - `t = cB − R cA`.
  It is exact on coplanar points (a flat board), which was checked on random transforms to
  1e-13.
- **3-D residual** per point: `|T·p_cam − p_robot|` in mm. The report prints each one, the
  **RMS** `sqrt(mean(r²))`, the worst and the mean.
- **Reprojection error** per point, in px: `|proj(T⁻¹·p_robot) − proj(p_cam)|`.
  - `proj` is `cv2.projectPoints` with the stream's `K` and distortion, the same ones the pose came
    from.
  - It shows where the fitted `T` puts the typed point in the image, compared with where the board
    pose puts that corner.
  - A point that lands behind the camera (`z ≤ 0`) is shown as `inf`, not projected (projection
    would mirror it).
  - RMS, worst and mean are printed.
- **Board pose fit** per run: S prints that frame's `rms_px`. SPACE prints the RMS, median and
  worst over the kept frames, plus `combine_poses`' spread. If SPACE's frames scatter beyond
  3 mm / 1°, `combine_poses` raises SystemExit. It is caught: the message is printed and the live
  view continues, so kept runs aren't lost.
- **Distance check** per run: `|P0P1|`, `|P0P2|` typed against the board, ±15 mm.

What more runs buy: a simulation with 1 mm touch noise on the a2 corners gave, as a mean over 2000
trials:
- **1 run** (3 points): 0.43° rotation error, 4.6 mm error at 0.7 m;
- **5 runs spread over about 1 m**: 0.054° and 0.7 mm.

## Placing the frame on BoardOverlay (`push_to_station`)

- **Run 1**, with no estimate yet, uses the **board-only placement** (`place_board`):
  - the board is flat at z = 0 with yaw 0 (P0→P1 along +X): `board_in_base(spec, at, 0)` from
    `robot/board_demo_calibration.py`;
  - it is shifted so the camera lands at `--camera-at X,Y`:
    `at = camera_xy − (board_in_base(spec,(0,0),0) · T_cam_board⁻¹)[:2,3]`;
  - `T_overlay = compose(T_base_board, T_cam_board) = T_base_board · T_cam_board⁻¹`.
- **Runs 2 and later** use the current estimate: `T_overlay = fit(kept).T_base_cam`, and
  `T_base_board = T_overlay · T_cam_board`. The board may then lie at any flat position and
  rotation.
- **A printed warning starts every run from run 2:** the BoardOverlay picture is now placed with
  the estimate, so it may have moved or turned since run 1. Touch the marks where they are now.
- **The picture:** `for_robotstudio(floor_picture(annotated, T_overlay, intr))`. The marks are drawn
  **after** the turn, so the labels read upright. `turned_px(x, y)` gives column
  `x − Xmin − 0.5` and row `Ymax − y − 0.5`, the `station_feed.floor_uv` convention.
- **The marks:** magenta `Pi (x, y)` and a yellow `camera (x, y)`. A warning names any P outside
  `FLOOR_X_MM`/`FLOOR_Y_MM`.
- **The request:** `build_request(frame, calibration_floor.png, calibration_cloud.bin, 0,
  floor_visible=True)`. `floor_visible=True` matters because an earlier `robot_trash_picker.py
  --hide-rgb` would otherwise leave BoardOverlay hidden. The cloud is written with 0 points.

## Fresh stream every run

- The pipeline is started once, only to read the intrinsics, and then stopped.
- Each run calls `start_stream()`:
  - `pipeline.start(config)`;
  - for the live camera, the first `LIVE_WARMUP_FRAMES` (5) frames are thrown away while exposure
    settles;
  - for a clip, playback restarts from the beginning and loops.
- `capture_run` stops the pipeline in its `finally`.
- So no frame grabbed while points were being typed can be picked.
- **With `--rosbag`, every run replays the same clip.** Several board positions need a clip in which
  the board is moved; press S at each one.

## Placement ideas (printed after every `c`, same text each time)

- the board flat on the floor, nothing under it;
- fully in view, not covered, well lit;
- spread over the view: centre, then near each corner and edge;
- turned about 45° or 90° between runs;
- no spot used twice;
- inside BoardOverlay's X/Y range (printed);
- all three points within the robot's reach;
- 4–6 runs are usually enough.

## Keys and windows on Windows

- `read_key(delay)` returns `cv2.waitKey` or, failing that, a key typed in the terminal
  (`msvcrt.kbhit` / `getwch`). The OpenCV window usually has no keyboard focus when started from a
  terminal.
- `drain_keys()` empties both before the live view and before the decision.
- The window is `WND_PROP_TOPMOST` only while capturing, so it doesn't cover RobotStudio while you
  jog.

## Testing without a person (what was done)

1. Run the script in **its own console**:
   `Start-Process <venv>\Scripts\python.exe -ArgumentList "camera\calibrate_T_base_cam.py --rosbag 20261119_200126_board_calib_final --board a2 --camera-at 670,-5,1180 --output <scratch>\T.npy" -RedirectStandardOutput out.txt -WindowStyle Minimized`.
2. Type keys into that console from another `pwsh` with `AttachConsole(pid)` +
   `WriteConsoleInput`. Use each character's real virtual-key code (`VkKeyScan`); `'\n'` becomes
   Enter (VK 0x0D).
3. Follow the run in `out.txt`:
   - wait for `SOLIS (merijums` (one per capture), send `s` or a space;
   - wait for `P0 robota`, then read the three `Pi: X= Y= Z=` lines after the last
     `attels novietots` and type them back;
   - wait for `d - izmest so` (one per decision), send `d`/`c`/`f`/`x`.
4. Cases run against the board clip, with the live RobotStudio add-in:

| key sequence | expected | seen |
| --- | --- | --- |
| `s c s d s f` | run 2 discarded and redone; fit over runs 1 and 2; saved | yes. Residuals 0.03 mm / 0.02 px (rounding of the typed 0.1 mm); stream frames 55, 110, 167 = a fresh start every run; `shown in RobotStudio ... 0 points` each time |
| `␣ x` | nothing kept; exit 1, no `.npy` | yes |
| `s c ␣ x` | saved from run 1 only | yes |
| `s c ␣ f` | SPACE line shows RMS, median and worst | yes (RMS 0.15 px, 30 frames) |

## Found while building it

- **PowerShell pitfalls:**
  - Don't name a PowerShell function `Type`: it collides with the built-in alias `type`
    (Get-Content).
  - Escape `(` in `Select-String` patterns: it is a regular expression.
- **The venv's `python.exe` is a launcher.** The process with the window is its child `python.exe`;
  `conhost.exe` is another child.
- **Typed back exactly, residuals are about 0.** With only the RobotStudio robot, the touched points
  come from the texture placed above. Typing them back gives residuals at rounding level. Bigger
  residuals come only from typing, or from touching off the marks.
- **All prints in steps 2/3 are plain ASCII**, so a cp1252 console can't fail on them. The older
  step-1 banners keep their Latvian diacritics.
