# TrashPicker Station: the snapshot in RobotStudio

This is for a person or a Claude agent setting this up on another machine. It covers what the feature does, how to install and check it, and which assumptions were true on the machine where it was built. Check each of those assumptions before trusting the result.

## What it does

`robot/robot_trash_picker.py` picks the frame the robot works from: the operator presses **'n'** on the live RealSense, or the run reaches **`--frame N`** (by default the last frame) of a `--rosbag` clip. That one frame is then shown in the running RobotStudio station:

- **BoardOverlay** (a Part in the station) gets the annotated frame as its texture. That includes the masks, boxes and labels from the detector, plus each target's red cross, rank, long axis and finger line. The frame is laid on the floor at z = 0, 1 px = 1 mm.
- **LiveCloud** (a point cloud the add-in creates in the station) gets the same frame's depth as points in the robot base frame, in metres, coloured from the raw RGB image.

Nothing polls and nothing is sent on other frames. A frame with no detections is still shown on 'n', so the camera can be checked with an empty scene.

Building the two files and talking to RobotStudio both happen on a background daemon thread, so 'n' doesn't freeze the image window and a slow RobotStudio never holds up the robot. One frame is sent at a time. A frame chosen while another is on its way goes next ("frame 6 goes next"), replacing any frame already waiting, so the station always ends on the latest frame chosen. On a normal exit the program waits for the thread, still hearing 'h'. After an error or Ctrl+C it doesn't wait.

**'h'** stops the frame that thread is sending, if RobotStudio hangs: it closes the socket. The run and the robot carry on. A frame that was waiting is sent next, and the next 'n' tries again. 'h' works at every prompt: the preview, during a move, between picks ('c'/'s'), and at the experiment's y/n/x verdict, where it is not taken as an answer. The script prints this when it starts. `--no-station` turns the feature off.

**`--hide-cloud`** hides LiveCloud and **`--hide-rgb`** hides BoardOverlay; both can be given. They are sent to RobotStudio as a `visibility` command when the script starts, before it connects to the robot, and every `show` carries them too. The startup command always states both items, so a run without the options shows again what an earlier run hid. A hidden item is still updated with each frame, so showing it again shows the latest frame. The options mean nothing with `--no-station` or `--mock`, and are refused there.

### The pieces

| Where | What |
| --- | --- |
| `robot/station_feed.py` | copies the frame, then on a daemon thread builds both files and sends one JSON request (latest frame wins); also `--test-pattern` and `--screenshot` |
| `robot/frame_selection.py` | `preview_until(..., station=)` calls `station.show()` on an accepted 'n' or on `stop_at` only |
| `robot/operator_keys.py` | `OperatorKeys.on(key, action)` for 'h'; `pump()` keeps the window alive during moves and still hears 'h' |
| `robot/robot_trash_picker.py` | creates the feed (unless `--no-station`), binds 'h', and on a normal exit waits for a send in flight (`station_waits`) |
| `robot/rapid/TrashPickerStation/` | the RobotStudio add-in: `StationAddin.cs`, `.csproj`, `.rsaddin`, `manifest.xml`, `install.ps1`, and the prebuilt `package/` |

### The protocol (127.0.0.1:5003, one JSON line each way)

```
-> {"cmd": "show", "frame": 233, "floor": "<abs>\\station_floor.png", "cloud": "<abs>\\station_cloud.bin",
    "points": 57600, "floor_x_mm": [-40, 1380], "floor_y_mm": [-390, 380], "mm_per_px": 1.0,
    "part": "BoardOverlay", "cloud_visible": true, "floor_visible": true[, "look_down": true]}
<- {"ok": true, "frame": 233, "points": 57600, "mapped_faces": 1, "view": null, "undo_steps": 0,
    "release_ms": 50}
-> {"cmd": "visibility", "part": "BoardOverlay", "cloud_visible": false, "floor_visible": true}
<- {"ok": true, "floor_visible": true, "cloud_visible": false, "cloud_exists": true}
-> {"cmd": "screenshot", "path": "<abs>.png", "floor_x_mm": [...], "floor_y_mm": [...][, "look_down": true]}
<- {"ok": true, "path": "..."}
```

The files are written to `robot/robotstudio/live/station/`, which git ignores. `station_cloud.bin` is `b"TPC1"`, then uint32 N, then float32 N×3 xyz in metres in the base frame, then uint8 N×3 RGB.

## Why the texture and the cloud line up exactly, and when they don't

**The one condition:** the calibration must put the real floor at z = 0.

- Texture and cloud agree exactly **along every camera ray**: each cloud point and the texel holding its pixel's colour lie on the same ray from the camera. The texel is where that ray meets z = 0.
- Seen from above, they coincide only for surfaces at z = 0. Objects above the floor "lean" in the texture (see `floor_overlay.py`), as expected.
- If the floor itself is not at z = 0, the floor in the cloud and the floor in the texture separate sideways by about `floor height × (horizontal distance from the camera's foot ÷ camera height)`.
- **This is the case for the rosbag clips with `calibration_rosbag_demo.txt`.** That file predates commit `8cf7fd1`, which put the floor at z = 0, and it puts the clips' floor at **z ≈ +70 mm**: the median of the cloud on frame 142 of clutter_med is 70.5 mm. The peer review measured a median of 39 mm (p95 73 mm) between each cloud point and its texel, seen from above.
- The slow test `test_the_recorded_sheet_reads_level_through_the_demo_file` fails for the same reason, at the commit this work started from too.
- Regenerating the file (`robot/align_to_recording.py`) currently stops with "the clips do not agree on one camera position: up to 5.30 degrees and 18.7 mm".
- Before trusting the alignment on a machine, check where the floor lands: the median z of `robot/robotstudio/live/station/station_cloud.bin` after a run (`robot.station_feed.read_cloud`).

What makes it exact along the rays:


1. **One frame.** Both files come from one `(image, depth, intrinsics)` grab and are named in one request. The add-in reads both files before it changes anything, then applies both in one call on the UI thread.
2. **One geometry.**
   - The floor texel at (x, y, 0) takes the colour of the pixel that `K · R^T (p − t)` projects to (`floor_overlay.lay_on_floor`).
   - The cloud deprojects each pixel with the same K and moves it with the same `T_base_cam` (`scene_cloud.cloud_in_base`).
   - Both refuse non-zero distortion coefficients through `scene_cloud.require_pinhole`, so one path can never model distortion while the other ignores it.
3. **One mapping.** On every request, the add-in sets BoardOverlay's texture coordinates from its own vertices' world positions. That is one face, so it's cheap, and a moved or replaced part still maps right: `u = (x − xMin) / width`, `v = (yMax − y) / height`. This is the same formula as `station_feed.floor_uv`, and a test checks that the two still match. It does not rely on a texture mapping applied by hand.
4. **Tests.**
   - `robot/tests/test_station_feed.py` builds a synthetic floor seen by a straight-down camera and by a tilted, offset one.
   - **Exact texel check:** an image whose pixels hold their own (u, v) is laid on the floor. Each texel must hold the pixel its centre projects to, within 0.05 px, which is `cv2.remap`'s 1/32-px precision. A half-texel (0.5 mm) shift is about 0.3 px and fails.
   - Every floor point in the cloud must sample its own colour from the texture (max error ≤ 2/255 away from block edges), and a spot's centre must agree within 0.5 mm.
   - The renderer's V direction is checked only by the test pattern in RobotStudio (below).
   - The peer review independently re-projected the real clip (frame 142): every cloud point lands on its source pixel within 4e-4 px, and the best colour-matching texture shift is (0.0, 0.0) mm. In RobotStudio's own render of the test pattern, the dot grids are centred on their squares within 0.1–0.9 mm, about one rendered pixel.

## Install on a new machine

1. **RobotStudio** 2026 (tested: 26.3.11844.1).
   - The prebuilt DLL targets **.NET 10**, which is what RobotStudio 2026 runs on (see `<RobotStudio>\Bin\RobotStudio.runtimeconfig.json`, `"tfm"`).
   - If that file names another runtime, edit `<TargetFramework>` in the `.csproj` and the `Add-In-net10.0` folder name in `install.ps1`, then rebuild.
2. **Install the prebuilt package**, which needs no SDK:
   ```powershell
   robot\rapid\TrashPickerStation\install.ps1
   ```
   This copies `package\` to `%LOCALAPPDATA%\ABB\DistributionPackages2\TrashPickerStation-1.0.0\`, holding `manifest.xml` and `RobotStudio\Add-In-net10.0\{TrashPickerStation.dll, .rsaddin}`.
3. **Or rebuild first**, which needs the .NET 10 SDK: `winget install Microsoft.DotNet.SDK.10`.
   ```powershell
   robot\rapid\TrashPickerStation\install.ps1 -Build
   ```
   - Without `-RobotStudioBin`, `install.ps1` (`Find-RobotStudioBin`) looks for RobotStudio's `Bin` in this order: the development machine's folder, then a running `RobotStudio.exe`, then the newest `ABB\RobotStudio*\Bin` under both Program Files folders. It prints the folder it used. If RobotStudio is installed somewhere else, start it once so it is found, or pass `-RobotStudioBin "<RobotStudio>\Bin"`.
   - Building straight with `dotnet build` doesn't search. The `.csproj` default is the development machine's folder, so pass `-p:RobotStudioBin="<RobotStudio>\Bin"`.
   A running RobotStudio holds the installed DLL open, so **close RobotStudio before installing**. `install.ps1` refuses while it runs; `-Build` alone still refreshes `package\` before that check.
4. **Start RobotStudio and open the station** (`robot/rapid/Project6.rspag`, or an unpacked copy).
   - The Output window says: `TrashPicker station: listening on 127.0.0.1:5003 ...`.
   - Check from a shell: `Get-NetTCPConnection -LocalPort 5003 -State Listen`.
5. **Run the alignment check:**
   ```powershell
   robot\.venv-win\Scripts\python.exe -m robot.station_feed --test-pattern --screenshot pattern.png --look-down
   ```
   - The reply is `ok` with `mapped_faces` ≥ 1 (BoardOverlay has one face). `pattern.png` is RobotStudio's own render of its 3D view, turned straight down on the board with +X to the right and +Y up.
   - Expect six 80 mm squares, each labelled with its X/Y: X0 Y0 white, X600 Y0 red, X600 Y300 green, X1200 Y−300 blue, X300 Y−300 yellow, X1200 Y300 magenta.
   - Each square should have a 5×5 grid of cloud dots centred in it, with equal margins.
   - Y300 squares should be at the **top**. If they are at the bottom, this renderer's V runs the other way: flip the `v` formula in **both** `StationAddin.cs` and `station_feed.floor_uv`, and the test that compares them.
   - `--look-down` changes the operator's view. A real run never does.
6. **Run the program:**
   ```powershell
   robot\.venv-win\Scripts\python.exe robot\robot_trash_picker.py --rosbag 20261016_231841_clutter_med.db3 --calibration robot\calibration_rosbag_demo.txt
   robot\.venv-win\Scripts\python.exe robot\robot_trash_picker.py --calibration robot\calibration_home_board.txt
   ```
   Each run prints `shown in RobotStudio: frame N on BoardOverlay, M points in LiveCloud`. If RobotStudio isn't listening, it prints a WARNING naming the files and carries on.

   To check what RobotStudio shows without anyone at the screen, run `python -m robot.station_feed --screenshot view.png`. Add `--look-down` to turn the view straight down first.
7. **Run the tests:** `robot\.venv-win\Scripts\python.exe -m pytest robot\tests`.
   - The slow tests skip unless the default checkpoint is present and **all three** clips are unpacked under `rosbags\export\`. Unpack them with `robot.frame_selection.open_recording('<clip name>')`.
   - The tests never contact RobotStudio: `conftest.station_requests` replaces `StationFeed._exchange` with a recorder. `test_station_feed` and two `test_end_to_end` tests put the real one back against a stand-in add-in (`FakeAddin`).
   - On the development machine, 21 tests already failed at the commit this work started from (`eb6a0ee`), and the same 21 fail with this work. None are caused by this feature; compare against a clean checkout (`git archive HEAD robot camera`) before blaming it.
     - 17 fast ones in `test_art_camera_client`, `test_end_to_end`, `test_pick_sequence` and `test_rapid_mirror` come from the workspace limits widened in `f70c892` (X/Y ±1000).
     - 4 slow ones in `test_align_to_recording` come from the stale demo calibration described above.

## Assumptions from the development machine: check that each one holds

| Assumption | How it was true here | How to check |
| --- | --- | --- |
| RobotStudio version and path | 2026 (26.3.11844.1) in `C:\Program Files (x86)\Development\RobotStudio2026` (not the default `ABB\RobotStudio 2026`) | `Get-Process RobotStudio \| % Path`. Python never uses the path. `install.ps1 -Build` finds it (see Install step 3) or takes `-RobotStudioBin`. |
| RobotStudio's .NET | net10.0 | `Bin\RobotStudio.runtimeconfig.json` |
| Package folder RobotStudio scans | `%LOCALAPPDATA%\ABB\DistributionPackages2\<Name>-<Version>\RobotStudio\Add-In-net10.0\`. `DistributionPackages` (no 2) was **not** scanned. | After a restart, port 5003 listens. If not, look where ABB's own packages are (`ManifestCachev4.xml` in `%LOCALAPPDATA%\ABB\RobotStudio\` lists them). The DLL also names `Add-In` and `Add-In-net8.0` folders. |
| Add-in trust / license prompts | none appeared | watch RobotStudio's start-up and the Add-Ins tab |
| BoardOverlay | a Part named exactly `BoardOverlay`, identity transform, one flat face spanning X −0.04..1.38 m, Y −0.39..0.38 m at z 0, which equals `FLOOR_X_MM`/`FLOOR_Y_MM` in `floor_overlay.py` | the reply's `mapped_faces`, and the test pattern. The UVs come from vertex world positions, so another transform still maps right, but a part that doesn't cover the rectangle shows only part of it. **Moving BoardOverlay needs `FLOOR_X_MM`/`FLOOR_Y_MM` shifted by the same amount: follow `BOARD_OVERLAY_MOVE.md`.** |
| Station frame = robot base frame | robot base frame identity at the world origin | the robot's `BaseFrame` in the station. Otherwise the cloud and the floor picture are both off by that transform. |
| Camera distortion | D435 colour stream: Inverse Brown-Conrady, all coefficients 0 (live and in every clip) | `rs.video_stream_profile(...).get_intrinsics().coeffs`. Non-zero coefficients are refused (a warning, nothing shown); supporting them needs distortion in both `lay_on_floor` and `cloud_in_camera`. |
| Camera link | **USB 2.1** on this machine, for both cameras tried. The D435 (fw 5.17.3.10) delivered no frames at all. The D415 (fw 5.12.7.150) delivers a burst of 11–15 framesets at 640×480@30 colour+depth after each start or reset, then stalls. Nothing at 15 or 6 fps, with one stream alone, or at 424×240. Windows' camera privacy setting allows desktop apps and both devices report OK, so the limit is the port or cable. The program needs 30 settled frames before it accepts 'n', so a normal live run stalls first and ends with its USB 2 explanation. | `rs-enumerate-devices` → `usb_type_descriptor` should be 3.x; try another port or a known USB 3 cable. Then run the program without `--rosbag`, press 'n' after a second, and expect `shown in RobotStudio`, even with nothing detectable in view. |
| Port 5003 free | yes (the RAPID socket is 5002) | `Get-NetTCPConnection -LocalPort 5003` before starting RobotStudio |
| Python env | `robot/.venv-win` with `requirements-robot.txt` | `pip install -r requirements-robot.txt` |
| Calibration puts the floor at z = 0 | **No** for `calibration_rosbag_demo.txt` with the clips (floor at +70 mm). **No** for the D415 that replaced the D435 with `calibration_home_board.txt`, which was measured for the old camera: the live cloud's median z was −673 mm, and the pick guard refused a target at z −294 mm. Calibrate the camera where it is mounted (`camera/calibrate_T_base_cam.py` or `robot/board_demo_calibration.py`). | median z of the station cloud after a run should be about 0; board edges and tape should line up with BoardOverlay's 100 mm grid |
| Screen capture | GDI captures (`CopyFromScreen`, `PrintWindow`) of RobotStudio's 3D view returned **stale** frames | use `--screenshot`, which RobotStudio renders itself |
| RobotStudio memory per frame shown | **Fixed.** It grew about 4.5 MB per push without bound: one 1420×770 RGBA texture each time. `undo_steps` stayed 0, so the undo history was not the cause. A `Texture` has no `Dispose` and frees its native image only in its finalizer, and the collector rarely ran. The add-in now runs one collection after each frame shown (`ReleaseOldTextures`, about 50 ms, reported as `release_ms`). Over 60 pushes of a real 57k-point frame, memory then stayed between 520 and 550 MB. | push the same frame 40 or more times and watch RobotStudio's private memory (`(Get-Process RobotStudio).PrivateMemorySize64`). After a rise over the first few pushes it should level off. If it climbs steadily again, look for another object without `Dispose` that is replaced per frame. |
| The PC stays unlocked during RobotStudio tests | Windows locked part way through this work. A pending close of RobotStudio then sat behind the lock screen. | restarting RobotStudio for an add-in update needs the desktop unlocked |

## The live-camera test on this machine

The camera's link only gave a short burst (see the camera row above). The live path was tested with a harness that used the real `robot_trash_picker.main()`, the real D415, model, station feed and RobotStudio add-in. Only three things were stand-ins:

- the controller (`robot/mock/fake_rapid.FakeRapid`);
- the keys: 'n' on the first frame, then 's' at the pick prompt;
- the settling time, set to 0 (`frame_selection.SETTLE_S`) so 'n' landed inside the burst.

The result was `shown in RobotStudio: frame 1 on BoardOverlay, 24656 points in LiveCloud`, followed by a clean exit. BoardOverlay showed the live frame with its masks, boxes, labels and the TCP cross (checked with `--screenshot --look-down`). The cloud lay below the board because of the calibration mismatch above.

On a link that works, repeat this without the harness: the normal program, 'n' after a second.

If the camera stalls *after* 'n', the program ends with the stall explanation. An exit on an error doesn't wait for the background send (so errors are reported at once), so that frame may not reach RobotStudio.

## Things found while building it

- RobotStudio's API is UI-thread only. The add-in listens on a background thread and marshals each request with the `SynchronizationContext` captured in `AddinMain`.
- `Camera.UpDirection` must be set before `LookFrom`/`LookAt` for a straight-down view. The default +Z is parallel to the look direction and throws. `GraphicControl.Camera` may be null, so take `FieldOfView` from the control. Setting the camera needs `view.Camera = cam` (plus `SyncCamera` and `Update`).
- `Station.UpdateGraphics` is static.
- `OperatorKeys.drain()` must not stop at an action key: a stale 'c' queued behind an 'h' would otherwise start the next pick. A test covers this.
- The send runs on a daemon thread, so on a normal exit the program waits for it, still hearing 'h'; otherwise a quick 'q' after 'n' would cut it off. It doesn't wait after an error or Ctrl+C, so a robot error is reported at once.
- A peer review (a second agent) found, and these were then fixed:
  - a frame refused while busy (now it waits as "next");
  - the build blocking the image window for about 0.4 s (now on the thread);
  - the wait delaying error messages;
  - a stale "already shows it" hint printed with `--no-station`;
  - UVs set only once per station;
  - a POSIX-only false warning after 'h';
  - unchecked arithmetic and an Accept spin in the add-in;
  - tests that missed a half-texel shift, could hang, or never ran the real thread end to end.
- The review also claimed the installed DLL isn't locked while RobotStudio runs. That was wrong here: deleting it was denied, hence the check in `install.ps1`.

## Calibration: the chosen frame on BoardOverlay (instructions for a Claude agent)

`camera/calibrate_T_base_cam.py` solves `T_base_cam` (camera → robot base) from a ChArUco board. It
works on the live RealSense or on a recorded clip (`--rosbag <clip>`). Right after the operator
picks the board frame, that frame is put on BoardOverlay in the running RobotStudio station. The
operator then jogs the RobotStudio robot onto the board's three reference corners on that texture
and types their coordinates. Follow the steps in order.

### 1. First: measure the camera, then set up the station

1. Ask the user to measure the **camera centre** (the middle of the colour lens) as X, Y, Z in mm,
   from the cross taped on the floor:
   - the cross is the robot base origin;
   - +X runs along the x tape line;
   - +Y is +X turned 90° to the left, seen from above;
   - Z is the height above the floor.
2. In RobotStudio, move the **camera model**, which already exists in the station, to that X, Y, Z.
   Then move **BoardOverlay** so that its centre is under the camera at (X, Y), with z = 0:
   - BoardOverlay's X/Y must keep the same position relative to the camera's X/Y. When the camera
     moves by (dx, dy), BoardOverlay moves by the same (dx, dy).
   - **If BoardOverlay moves, the code must change too:** shift `FLOOR_X_MM`/`FLOOR_Y_MM` in
     `robot/floor_overlay.py` by the same amount.
   - Follow `BOARD_OVERLAY_MOVE.md` in this folder step by step; it explains why.
   - Both programs print this at start-up. `calibrate_T_base_cam.py` also warns when `--camera-at`
     is more than 20 mm from BoardOverlay's centre.
3. Lay the board flat on the floor, in the camera's view, with its **P0 → P1 edge along +X**
   (yaw 0). P1 is +X from P0; P2 is −Y from P0 (the board's own +Y is the base −Y when it lies face
   up). `camera/charuco.py` `reference_points` defines P0 P1 P2 for each board.

### 2. Run the calibration

The script now works in **runs**: capture, then touch P0–P2, then decide (`d` discard and redo,
`c` keep and continue, `f` keep and finish, `x` discard and finish), repeated with the board moved,
flat, between runs. All kept points go into one fit. From run 2 on, the BoardOverlay picture is
placed with the current estimate, and the script warns that it may have moved. The steps below
describe one run. `camera/CALIBRATE_T_BASE_CAM_RUNS.md` has the run loop, the maths and how it was
tested.

```
python camera/calibrate_T_base_cam.py --board <a2|home40|home50> --camera-at X,Y,Z [--rosbag <clip>] [--output <file.npy>]
```

1. **Start-up lines.** They say which BoardOverlay centre the code assumes and point to
   `BOARD_OVERLAY_MOVE.md`. Without `--camera-at`, the board centre is placed at (450, 100) mm
   instead of under the camera, with a warning.
2. **Pick the frame.** A window `Kalibracija` shows the camera (or the clip, looping) with the board
   corners, its axes and P0 P1 P2.
   - **S** (clip only) takes the frame on screen.
   - **SPACE** averages the next `--frames` frames (default 30).
   - **ESC** quits.
   - The keys work in that window **or in the terminal the script runs in**. Windows usually doesn't
     give the OpenCV window keyboard focus, so the terminal is the reliable place.
3. **What happens after S or SPACE:**
   - The window freezes on the **final** frame. With SPACE, that is the averaged pose drawn on the
     last good frame.
   - The frame is laid on the floor and sent as **BoardOverlay's texture**, through
     `robot/station_feed.py` and the add-in.
   - **No point cloud is sent.** The add-in's `show` always replaces LiveCloud, so calibration sends
     an **empty** cloud (0 points). LiveCloud stays empty until the next `robot_trash_picker.py`
     frame.
   - The files are `robot/robotstudio/live/station/calibration_floor.png` and
     `calibration_cloud.bin`. A pick run's `station_floor.png` and `station_cloud.bin` are not
     touched.
   - The terminal prints `shown in RobotStudio: frame N on BoardOverlay, 0 points in LiveCloud`.
   - `--no-station` sends nothing.
4. **How the picture is placed** before `T_base_cam` exists:
   - The board lies flat at z = 0 with yaw 0, and is shifted so that the camera lands at the
     measured X, Y. The board's pose relative to the camera comes from the camera.
   - The terminal prints P0 P1 P2 and the camera in base mm, plus the gap between the measured Z and
     the Z the board pose gives. A few mm is normal; tens of mm means a mis-measurement or the wrong
     `--board`.
   - It warns about any P point outside BoardOverlay, which RobotStudio would not show.
5. **On BoardOverlay** (labels read upright, +X right, +Y up when looking down):
   - magenta circles `P0 (x, y)`, `P1 (x, y)`, `P2 (x, y)`;
   - a yellow `camera (x, y)` mark, which must be at BoardOverlay's centre, under the camera model;
   - the red P0 P1 P2 dots and the board axes from the camera image, which must coincide with the
     magenta circles.

### 3. After the image is on BoardOverlay: touch the three points

The calibration needs the position of the **fingertips**, which is the TCP of the tool
**toolRG6**: `robot/rapid/ArtCamera.modx`, `tooldata toolRG6 := [TRUE,[[0,0,253],...]]`, i.e. 253 mm
along the flange's z axis.

1. In RobotStudio, make **toolRG6** the active tool and **wobj0** the work object (the base frame).
   - With tool0 active, the position read is the flange, 253 mm off. The fit then fails its checks
     and nothing is saved.
   - Check that toolRG6 exists in the station's controller. If the RAPID was not synchronised it may
     be missing.
2. Jog the robot (Jog Linear, or Freehand → Jog Linear) until the fingertips sit on **P0**, as drawn
   on BoardOverlay.
3. Read the TCP position in mm (X, Y, Z, in the base/wobj0 frame), and type it at the prompt
   `P0 robota koordinatas 'x,y,z' mm:` as `x, y, z`.
   - RobotStudio 2026 shows the TCP position while jogging in the status bar and in the jog
     dialog. The exact place is **unverified**: confirm it on screen.
4. Repeat for **P1**, then **P2**. The `Kalibracija` window stays open, showing the final frame,
   until all three points are typed. It closes after the third.
5. The script then:
   - checks |P0 P1| and |P0 P2| against the board;
   - fits `T_base_cam` (Kabsch) and prints the error per point;
   - saves the `.npy` (refused if the checks fail, unless `--force`);
   - writes `robot/robotstudio/live/floor.png`.
6. **What the result means.** Here the corners are touched on the texture, which was placed by the
   board-only placement above. So the fitted `T_base_cam` reproduces that placement: it is right
   relative to the board and the measured camera X/Y. Expect a mean error near 0 mm when the
   fingertips are exactly on the marks.

### 4. Check on this machine (each one is specific to the machine or to RobotStudio 2026)

| What | How it was here | How to check |
| --- | --- | --- |
| RobotStudio path | `C:\Program Files (x86)\Development\RobotStudio2026` | `Get-Process RobotStudio \| % Path`. Python never uses it. |
| Add-in listening | yes, 127.0.0.1:5003, package `%LOCALAPPDATA%\ABB\DistributionPackages2\TrashPickerStation-1.0.0` | `Get-NetTCPConnection -LocalPort 5003 -State Listen` |
| A 0-point cloud is accepted | **yes**: reply `ok`, LiveCloud emptied, texture shown (RobotStudio 26.3.11844.1) | the `shown in RobotStudio ... 0 points` line; `python -m robot.station_feed --screenshot <png> --look-down` |
| The camera model | exists in the station; its name wasn't recorded | the Layout browser; check where its origin is relative to the lens |
| BoardOverlay centre | (670, −5) mm, from `FLOOR_X_MM`/`FLOOR_Y_MM` | read the constants; compare with the part in RobotStudio |
| toolRG6 TCP at the fingertips, [0, 0, 253] | from `ArtCamera.modx` | the tool in the station's controller |
| How RobotStudio 2026 shows the TCP position while jogging, and how to set the active tool and work object | **unverified** | look at the RobotStudio UI |
| The verification view | `--screenshot --look-down` turns the user's 3D view straight down | tell the user before using `--look-down` |

### 5. Found while building it

- **Windows doesn't give the OpenCV window keyboard focus** when the script is started from a
  terminal; `SetForegroundWindow` from another process is refused too. So `read_key()` in
  `calibrate_T_base_cam.py` reads `cv2.waitKey` and the console (`msvcrt`). The window is set
  `WND_PROP_TOPMOST` so the terminal doesn't hide it.
- **During the three prompts, `input()` runs on a helper thread** while the main thread keeps
  calling `cv2.waitKey(30)`. Otherwise the frozen window stops painting and Windows marks it Not
  Responding.
- **Calibration marks are drawn on the turned picture.** `floor_overlay.draw_points_on_floor` draws
  before `for_robotstudio` turns the picture, which leaves labels sideways in RobotStudio. So
  `calibrate_T_base_cam.py` draws its marks after the turn, at column `x − Xmin − 0.5` and row
  `Ymax − y − 0.5`, the `floor_uv` convention.
- **The board clip** `rosbags/20261119_200126_board_calib_final.db3` holds a bgr8 colour stream at
  1280×720, 15 fps, 9.5 s, plus z16 depth at 848×480. The a2 board is found on every frame
  (reprojection about 0.13 px).
- **With that clip and the camera at BoardOverlay's centre (670, −5),** P0 and P1 land at Y ≈ 395,
  just outside BoardOverlay (Y ≤ 380). The script warns about it. A real setup should lay the board
  nearer the middle of the camera's view.
- **Driving the script without a person, for tests:**
  - The venv's `python.exe` is a launcher: the process with the window is its **child**
    `python.exe`. Another child is `conhost.exe` when it has its own console.
  - Keys reach `msvcrt` when written into that console with `AttachConsole` + `WriteConsoleInput`.
    Give each key event its real virtual-key code (`VkKeyScan`): sending a character's code point
    as the VK turns `.` into Delete, and the digits after it get lost.
  - In PowerShell, a `$null` passed to a P/Invoke `string` arrives as `""`. So
    `FindWindow($null, title)` never matches; pass `[NullString]::Value`.
