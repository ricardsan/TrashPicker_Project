# Moving BoardOverlay (instructions for a Claude agent)

Read this whole file before acting. It assumes you start with an empty context. Read
`STATION_SETUP.md` in this folder first: it explains what BoardOverlay is and how pictures reach it.

## The goal

The user has measured where the camera is. The RobotStudio station must match that:

- the camera model sits at the measured position, and
- BoardOverlay is centred under the camera.

BoardOverlay's X/Y must stay in the same position relative to the camera's X/Y. Whenever the camera
moves in X/Y, BoardOverlay moves by the same amount. The Python code must then be changed to match,
because it decides where on the floor the picture goes.

## Why moving BoardOverlay needs a code change

- **The picture is tied to fixed floor positions, not to the part.**
  - `robot/floor_overlay.py` lays every picture on the fixed floor rectangle
    `FLOOR_X_MM` × `FLOOR_Y_MM`, at 1 px = 1 mm.
  - On every `show`, the add-in (`StationAddin.cs`, `MapTexture`) sets BoardOverlay's texture
    coordinates from its vertices' **world** positions against that same rectangle, which is sent in
    each request: `u = (x − Xmin) / width`, `v = (Ymax − y) / height`.
- **Moving only the part shifts nothing.** If the part is moved and the constants are not, every
  texel still lands at its old world X/Y. The moved part then shows a cut-off picture (texture
  coordinates outside 0–1 on one side), and it is no longer under the camera.
- **The fix is the two constants.**
  - Shift `FLOOR_X_MM` and `FLOOR_Y_MM` by exactly the distance the part moved. They are the only
    place the rectangle is defined: `lay_on_floor`, `floor_px`, `floor_size_px`, `with_grid`,
    `station_feed.floor_uv`, `station_feed.build_request` and the tests all read them.
  - The add-in needs no change and no rebuild.
  - Keep the size (width × height) unchanged unless the part is also resized. The picture's pixel
    size follows the rectangle.

## Inputs to get from the user

Ask the user for the **camera centre** (the middle of the colour lens) as X, Y, Z in mm, measured
from the cross taped on the floor:

- the cross is the robot base origin;
- +X runs along the x tape line;
- +Y is +X turned 90° to the left, seen from above;
- Z is the height above the floor.

Example: `X=670, Y=-5, Z=1180`. Don't guess these numbers.

## Steps

1. **Read the current rectangle.** Open `robot/floor_overlay.py` and read `FLOOR_X_MM` and
   `FLOOR_Y_MM`. Don't trust any numbers quoted in docs: they may be out of date. The current centre
   is `cx = (FLOOR_X_MM[0] + FLOOR_X_MM[1]) / 2` and `cy = (FLOOR_Y_MM[0] + FLOOR_Y_MM[1]) / 2`.
2. **Compute the shift:** `dx = X − cx`, `dy = Y − cy`.
3. **Move the camera model in RobotStudio** to (X, Y, Z) mm in the station/world frame.
   - The camera model already exists in the station. Find it in the Layout browser. Its name isn't
     recorded here; ask the user if it isn't obvious.
   - Place its **lens centre** at (X, Y, Z). The model's own origin may not be the lens; check where
     the origin is (e.g. its local frame) and offset if needed.
   - The user does this in RobotStudio, or you do it with computer use if it's available and the user
     agrees. In RobotStudio 2026 the usual way is right-click → *Position* → *Set Position*. The exact
     menu path is **unverified**: confirm it on screen.
4. **Move BoardOverlay** by (dx, dy), with Z unchanged (it must stay on the floor, z = 0), and its
   orientation and size unchanged. Its centre is then at (X, Y), under the camera. Use the same
   *Set Position* route: new position = old position + (dx, dy, 0).
5. **Shift the constants** in `robot/floor_overlay.py`:
   ```python
   FLOOR_X_MM = (old_x0 + dx, old_x1 + dx)
   FLOOR_Y_MM = (old_y0 + dy, old_y1 + dy)
   ```
   Round to whole mm, and keep `x1 − x0` and `y1 − y0` exactly as they were. No test hard-codes the
   rectangle; they all read the constants.
6. **Run the tests:**
   ```
   python -m pytest robot/tests/test_station_feed.py robot/tests/test_floor_overlay.py -q
   ```
7. **Check alignment in RobotStudio.** RobotStudio must be running with the station open, and the
   add-in listening on 127.0.0.1:5003 (`Get-NetTCPConnection -LocalPort 5003 -State Listen`).
   ```
   python -m robot.station_feed --test-pattern --screenshot <scratch>\pattern.png --look-down
   ```
   - Read the PNG. Each coloured square is labelled with its X/Y.
   - The squares are at **fixed** positions (`PATTERN_POINTS_MM` in `robot/station_feed.py`), so
     after a move some may fall outside the new rectangle and be cut off. That's expected. Check the
     ones inside: each must sit on BoardOverlay at its labelled X/Y, with a 5×5 dot grid centred on it.
   - `--look-down` turns the user's 3D view straight down; tell the user.
8. **Check the camera mark.** Run `camera/calibrate_T_base_cam.py --camera-at X,Y,Z ...` (see
   `STATION_SETUP.md`, "Calibration"). The startup lines must no longer warn that the camera is away
   from BoardOverlay's centre. After S or SPACE, the yellow `camera (X, Y)` mark on BoardOverlay
   must be at the centre of the part, under the camera model.
9. **Update the docs that quote numbers.**
   - In `STATION_SETUP.md`, the BoardOverlay row of the assumptions table quotes the rectangle.
   - `robot/robotstudio/board_floor_overlay.txt` describes an old picture of the old rectangle; leave
     it, or note that it's historical.
   - The startup lines of `calibrate_T_base_cam.py` and `robot_trash_picker.py` take the centre from
     the constants, so they need no edit.

## Check on this machine (don't assume they hold)

| What | How to check |
| --- | --- |
| RobotStudio is installed and running. Its folder differs between machines (it was `C:\Program Files (x86)\Development\RobotStudio2026` on the machine this was written on). | `Get-Process RobotStudio \| % Path`. Nothing in Python uses the path. Only `install.ps1 -Build` does, and it finds the folder itself. |
| The add-in is installed and listening | `Get-NetTCPConnection -LocalPort 5003 -State Listen`; RobotStudio's Output window says `TrashPicker station: listening on 127.0.0.1:5003` |
| BoardOverlay is one flat face at z = 0, with no rotation | the `mapped_faces` in the add-in's reply is 1; the test pattern lines up |
| The station frame is the robot base frame (robot base at the world origin, no rotation) | the robot's base frame in RobotStudio. If not, both the camera and BoardOverlay positions are off by that transform. |
| The camera model's origin vs its lens centre | the model's local frame in RobotStudio |
| The RobotStudio 2026 UI for *Set Position* and for reading a position | look at it on screen; menus change between versions |
