"""Record RealSense D435i color + depth (+ intrinsics) to a .db3 (ROS 2 bag) file.

The recording can be replayed on any PC as if the camera were connected,
so a colleague can run any model / grasp pipeline on the exact same frames.

Usage:
    python record_bag.py --seconds 10 --fps 15 --name pile_mixed
    (defaults: color 1280x720, depth 848x480; old setup: --cw 640 --ch 480 --dw 640 --dh 480)
Press q / Esc in the preview window to stop early.
"""
import argparse
import time
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

p = argparse.ArgumentParser()
p.add_argument("--seconds", type=float, default=30, help="recording length")
p.add_argument("--name", default="scene", help="short scene description for the file name")
p.add_argument("--cw", type=int, default=1280, help="color width  (D435i: 640/848/1280/1920)")
p.add_argument("--ch", type=int, default=720,  help="color height (480/480/720/1080)")
p.add_argument("--dw", type=int, default=848,  help="depth width  (640/848/1280)")
p.add_argument("--dh", type=int, default=480,  help="depth height (480/480/720)")
p.add_argument("--fps", type=int, default=30)
a = p.parse_args()

out = f"{datetime.now():%Y%m%d_%H%M%S}_{a.name}.db3"

pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.color, a.cw, a.ch, rs.format.bgr8, a.fps)
cfg.enable_stream(rs.stream.depth, a.dw, a.dh, rs.format.z16, a.fps)
cfg.enable_record_to_file(out)
pipe.start(cfg)
print(f"Recording {a.seconds:.0f}s  color {a.cw}x{a.ch}  depth {a.dw}x{a.dh}  @ {a.fps} fps -> {out}   (q / Esc = stop early)")

colorizer = rs.colorizer()
t0, n = time.time(), 0
try:
    while time.time() - t0 < a.seconds:
        frames = pipe.wait_for_frames()
        n += 1
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(colorizer.colorize(frames.get_depth_frame()).get_data())
        # preview only: scale both to the same height so they fit side by side
        ph = 480
        c_small = cv2.resize(color, (int(color.shape[1] * ph / color.shape[0]), ph))
        d_small = cv2.resize(depth, (int(depth.shape[1] * ph / depth.shape[0]), ph))
        view = np.hstack([c_small, d_small])
        cv2.putText(view, f"REC {time.time() - t0:4.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("Recording (color | depth)", view)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
finally:
    pipe.stop()          # finalises the recording - never skip this
    cv2.destroyAllWindows()

print(f"Saved {n} framesets -> {out}")