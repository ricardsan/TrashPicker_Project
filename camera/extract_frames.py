"""Extract color frames (and optionally aligned depth) from RealSense .db3 / .bag recordings.

Neighbouring frames of a static scene are almost identical, so only every Nth
frame is kept. The first frames are skipped while auto-exposure settles.

Usage:
    python extract_frames.py --input recordings_HD --output frames_for_labeling --every 30
    python extract_frames.py --input recordings_HD --output frames_for_labeling --every 30 --depth
    python extract_frames.py --input recordings_HD\20261016_231116_stacked_train.db3 --output frames --every 15
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

p = argparse.ArgumentParser()
p.add_argument("--input", required=True, help="a .db3/.bag file or a folder of them")
p.add_argument("--output", required=True, help="output folder")
p.add_argument("--every", type=int, default=30, help="keep every Nth frame (30 at 15 fps = 1 every 2 s)")
p.add_argument("--skip", type=int, default=15, help="skip first N frames (auto-exposure settling)")
p.add_argument("--depth", action="store_true", help="also save depth aligned to color (16-bit PNG, mm)")
a = p.parse_args()

src = Path(a.input)
files = [src] if src.is_file() else sorted(list(src.glob("*.db3")) + list(src.glob("*.bag")))
out_color = Path(a.output) / "color"
out_color.mkdir(parents=True, exist_ok=True)
if a.depth:
    out_depth = Path(a.output) / "depth"
    out_depth.mkdir(parents=True, exist_ok=True)

total = 0
for f in files:
    # scene name = file name without the date/time prefix, e.g. "stacked_train"
    parts = f.stem.split("_", 2)
    scene = parts[2] if len(parts) == 3 else f.stem

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device_from_file(str(f), repeat_playback=False)
    profile = pipe.start(cfg)
    profile.get_device().as_playback().set_real_time(False)  # read every frame, no dropping
    align = rs.align(rs.stream.color)

    i, saved = 0, 0
    try:
        while True:
            try:
                frames = pipe.wait_for_frames(2000)
            except RuntimeError:  # end of recording
                break
            i += 1
            if i <= a.skip or (i - a.skip - 1) % a.every:
                continue
            frames = align.process(frames) if a.depth else frames
            color = np.asanyarray(frames.get_color_frame().get_data())
            name = f"{scene}_f{i:04d}"
            cv2.imwrite(str(out_color / f"{name}.jpg"), color, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if a.depth:
                depth = np.asanyarray(frames.get_depth_frame().get_data())  # uint16, millimetres
                cv2.imwrite(str(out_depth / f"{name}.png"), depth)
            saved += 1
    finally:
        pipe.stop()

    print(f"{scene:30s} {i:5d} frames read -> {saved:3d} saved")
    total += saved

print(f"\nDone: {total} images in {out_color}")
