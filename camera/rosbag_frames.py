"""Unpack a RealSense rosbag and serve its frames to the live segmentation loop.

A helper for camera/segmentation_realsense.py, which keeps the CLI. Nothing here imports from
there -- the frame-source contract is satisfied by duck typing and a builtin StopIteration -- so
the dependency runs one way only and this module is testable on its own.

Why this exists
---------------
The clips in rosbags/ are the test material rosbags/README.txt describes -- dense clutter,
occlusion, stacking -- and the pipeline could only ever be pointed at a camera. Replaying them
means every architecture is scored on byte-identical input, which no live run can offer.

Two passes, because the frames are wanted on disk in their own right (for labelling, and for the
point clouds):

    export_bag()           bag -> colour PNGs, 16-bit depth PNGs, .ply clouds, meta.json
    ExportedFrameSource    that directory -> (colour image, depth frame, intrinsics) per frame

No ROS 2 and no Docker are involved. These .db3 files are librealsense's own ROS 2 bag format --
their topics are /device_0/sensor_1/Color_0/image/data and friends, not realsense-ros's
/camera/... -- so pyrealsense2 opens them itself as a playback device. That needs librealsense
>= 2.56; 2.58.4 is what this was written against.

Fidelity
--------
Pass 2 hands the loop a duck-typed stand-in for rs.depth_frame and a hand-built rs.intrinsics, so
segmentation_realsense.annotate() -- boxes, centre dot, deprojected Z -- is reused verbatim rather
than reimplemented. Both are faithful:

  * a 16-bit depth PNG round-trips bit-exactly through cv2, and depth * depth_scale reproduces
    rs.depth_frame.get_distance() to float32 precision;
  * rs2_deproject_pixel_to_point accepts an intrinsics built by hand, so halving fx/fy/ppx/ppy
    alongside the half-scale colour image gives the same metric X/Y/Z as the native frame.

On the frame size
-----------------
The bags are 1280x720 but the live loop streams 640x480, and ultralytics would letterbox a native
bag frame to 1280x736 -- 3x the pixels of a live frame -- making a replayed run look slow for
reasons that have nothing to do with the model. Colour is therefore halved to 640x360, which keeps
16:9 and so keeps object shapes the models were trained on. It is not quite the live pixel count
(ultralytics pads it to 640x384, 246k px, against 307k live), so a replayed latency sits somewhat
below the same checkpoint's live figure; the startup line says so. Depth stays native and is
indexed through the scale factor, so distances are unaffected by any of this.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

# How much the colour frame is shrunk before the model sees it. See "On the frame size" above.
COLOR_SCALE = 0.5

META_NAME = "meta.json"
COLOR_DIR, DEPTH_DIR, CLOUD_DIR = "color", "depth", "cloud"

# Rough per-frame cost of each artefact, measured on rosbags/20261016_233733_stacked.db3 at
# 1280x720. Only used for the estimate printed before an export starts, so that gigabytes of
# point clouds is a decision rather than a surprise. The cloud figure is the shakiest of the
# three: export_to_ply writes only the pixels with valid depth, which on these clips ranges from
# 13 MB to 36 MB a frame depending on how much of the scene the projector reached.
MB_PER_COLOR, MB_PER_DEPTH, MB_PER_CLOUD = 0.84, 0.17, 30.0
SECONDS_PER_CLOUD = 0.5

# Above this much decoded imagery, ExportedFrameSource stops holding the whole clip in RAM. The
# 10 s clips in rosbags/ come to ~350 MiB, so the limit only bites on a recording many times
# longer than anything here.
PRELOAD_LIMIT_BYTES = 4 * 2**30


# =============================================================================
# 1. METADATA
# =============================================================================
def intrinsics_to_dict(intrinsics) -> dict:
    """An rs.intrinsics as JSON. `model` goes out by name so it survives the round trip."""
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "model": str(intrinsics.model).split(".")[-1],
        "coeffs": list(intrinsics.coeffs),
    }


def intrinsics_from_dict(data: dict, scale: float = 1.0):
    """Rebuild an rs.intrinsics, optionally for a frame scaled by `scale`.

    A pinhole intrinsics scales with the image: halve the image and fx, fy, ppx, ppy all halve,
    which is what makes deprojection in the 640x360 model frame agree with the native one.
    Distortion coefficients are dimensionless and do not scale.

    rs2_deproject_pixel_to_point accepts an object built this way, so no device is needed.
    """
    import pyrealsense2 as rs

    intrinsics = rs.intrinsics()
    intrinsics.width = int(round(data["width"] * scale))
    intrinsics.height = int(round(data["height"] * scale))
    intrinsics.ppx = data["ppx"] * scale
    intrinsics.ppy = data["ppy"] * scale
    intrinsics.fx = data["fx"] * scale
    intrinsics.fy = data["fy"] * scale
    # getattr rather than a dict: the enum is the authority on its own spellings, and an
    # unrecognised one should say so here instead of silently deprojecting with the wrong model.
    try:
        intrinsics.model = getattr(rs.distortion, data["model"])
    except AttributeError as error:
        raise ValueError(
            f"{META_NAME} names distortion model {data['model']!r}, which this pyrealsense2 "
            f"({rs.__version__}) does not have; it knows "
            f"{', '.join(rs.distortion.__members__)}") from error
    intrinsics.coeffs = list(data["coeffs"])
    return intrinsics


def read_meta(export_dir) -> dict:
    """An export's meta.json, with a legible error when the directory is not one."""
    path = Path(export_dir) / META_NAME
    if not path.exists():
        raise SystemExit(
            f"{export_dir} is not an unpacked bag: no {META_NAME} in it.\n"
            f"Unpack one with --rosbag <bag.db3> --export-only.")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        # is_exported treats this as "not exported" and re-unpacks, but --from-export comes
        # straight here, where a bare JSONDecodeError says nothing about what to do next.
        raise SystemExit(
            f"{path} is not readable JSON ({error}), most likely an unpack that was "
            f"interrupted.\nRe-unpack it with --rosbag <bag.db3> --export-only --force.") \
            from error


def scaled_intrinsics(meta: dict, scale: float = COLOR_SCALE):
    """The colour intrinsics of an export, for a frame scaled by `scale`."""
    return intrinsics_from_dict(meta["color_intrinsics"], scale)


def is_exported(export_dir) -> bool:
    """True when `export_dir` already holds a complete export, so it can be reused."""
    directory = Path(export_dir)
    if not (directory / META_NAME).exists():
        return False
    try:
        meta = json.loads((directory / META_NAME).read_text())
    except (OSError, json.JSONDecodeError):
        # A half-written meta.json, e.g. from an export killed partway. Treat it as absent so
        # --rosbag re-exports rather than failing on it later.
        return False
    return bool(meta.get("complete")) and bool(meta.get("frames"))


def reuse_shortfall(meta: dict, clouds: bool, cloud_stride: int) -> str | None:
    """Why an existing export does not satisfy this request, or None if it does.

    Only the point clouds can be short: the colour and depth frames are written for every frame
    whatever the flags, so an export made with --no-clouds has everything a later run needs.
    Without this check, asking for clouds after an earlier --no-clouds (or a coarser
    --cloud-stride) silently reuses the thinner export and quietly produces no clouds at all.
    """
    if not clouds:
        return None
    if not meta.get("clouds"):
        return "it was unpacked with --no-clouds"
    existing = int(meta.get("cloud_stride") or 1)
    if existing > cloud_stride:
        return (f"its clouds are every {existing} frames, coarser than the "
                f"{cloud_stride} asked for")
    return None


# =============================================================================
# 2. PASS 1 -- UNPACK THE BAG
# =============================================================================
def check_cloud_stride(cloud_stride) -> int:
    """A usable --cloud-stride, or a complaint. 0 would divide by zero; <0 would export every
    frame while reading as if it thinned them."""
    cloud_stride = int(cloud_stride)
    if cloud_stride < 1:
        raise SystemExit(f"--cloud-stride must be 1 or more, not {cloud_stride}")
    return cloud_stride


def write_image(path: Path, image) -> None:
    """cv2.imwrite, but loud about failing.

    imwrite returns False rather than raising -- on a full disk, a bad path, or an unwritable
    directory. Ignoring that is how an export ends up stamped `complete: True` with missing or
    truncated frames in it, which is then reused forever and only blows up much later, inside a
    run, after a checkpoint has been loaded. A default cloud export of one 10 s clip is several
    GB, so running out of space is the realistic failure here rather than a hypothetical one.
    """
    if not cv2.imwrite(str(path), image):
        raise SystemExit(
            f"could not write {path}. The usual cause is a full disk -- a default export of one "
            f"10 s clip is several GB. Free some space, or pass --no-clouds, and re-run with "
            f"--force.")



def export_estimate(n_frames: int, clouds: bool, cloud_stride: int) -> str:
    """What an export is about to cost, printed before it starts."""
    cloud_stride = check_cloud_stride(cloud_stride)
    megabytes = n_frames * (MB_PER_COLOR + MB_PER_DEPTH)
    seconds = n_frames * 0.04
    if clouds:
        n_clouds = (n_frames + cloud_stride - 1) // cloud_stride
        megabytes += n_clouds * MB_PER_CLOUD
        seconds += n_clouds * SECONDS_PER_CLOUD
    return f"about {megabytes / 1024:.1f} GiB and {seconds:.0f} s" if megabytes > 1024 \
        else f"about {megabytes:.0f} MiB and {seconds:.0f} s"


def bag_frame_count(bag_path) -> int:
    """How many colour frames a bag holds, read straight out of its sqlite index.

    Only wanted for the size estimate, and a cheap query beats playing the bag through twice.
    Returns 0 if the file is not the expected shape, in which case the estimate is simply skipped.
    """
    import sqlite3
    from contextlib import closing

    try:
        # closing() as well as the transaction context manager: `with sqlite3.connect(...)`
        # commits but does not close, leaving the handle on the bag open until GC.
        with closing(sqlite3.connect(f"file:{Path(bag_path).as_posix()}?mode=ro",
                                     uri=True)) as db:
            (count,), = db.execute(
                "SELECT COUNT(*) FROM messages m JOIN topics t ON t.id = m.topic_id "
                "WHERE t.name LIKE '%/Color_0/image/data'")
        return int(count)
    except sqlite3.Error:
        return 0


def open_bag(bag_path):
    """A playback pipeline over a bag, wound to the start and paced by the consumer.

    set_real_time(False) makes the device wait for whoever is reading rather than dropping frames
    to keep wall-clock pace, which is what an export needs: it must be lossless whatever pacing
    the later run uses.

    Returns (pipeline, profile, rs module).
    """
    import pyrealsense2 as rs

    bag_path = Path(bag_path)
    if not bag_path.exists():
        raise SystemExit(f"no such bag: {bag_path}")

    config = rs.config()
    try:
        config.enable_device_from_file(str(bag_path), repeat_playback=False)
    except RuntimeError as error:
        raise SystemExit(
            f"pyrealsense2 could not open {bag_path}: {error}\n"
            f"These must be librealsense's own bags (topics /device_0/sensor_*/...). A bag "
            f"recorded by realsense-ros is a different format and is not supported.") from error

    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    try:
        profile.get_device().as_playback().set_real_time(False)
    except RuntimeError:
        # The pipeline is live from start() onwards, and the caller has no handle on it until
        # this function returns -- so anything that throws in between has to stop it here or the
        # bag stays open for the life of the process.
        pipeline.stop()
        raise
    return pipeline, profile, rs


def export_bag(bag_path, export_dir, clouds: bool = True, cloud_stride: int = 1,
               force: bool = False, verbose: bool = True) -> Path:
    """Unpack a bag into colour PNGs, 16-bit depth PNGs, .ply clouds and meta.json.

    Depth is aligned to colour first, as rosbags/README.txt prescribes ("not aligned (align to
    color in replay)"), so the exported depth PNG is already on the colour grid and pass 2 needs
    no alignment and no playback device at all.

    Returns the export directory. An existing complete export is reused unless `force`, or
    unless it holds fewer point clouds than this call asked for.
    """
    export_dir = Path(export_dir)
    cloud_stride = check_cloud_stride(cloud_stride)

    if is_exported(export_dir) and not force:
        meta = read_meta(export_dir)
        shortfall = reuse_shortfall(meta, clouds, cloud_stride)
        if shortfall is None:
            if verbose:
                existing = ("no point clouds" if not meta.get("clouds")
                            else f"point clouds every {meta.get('cloud_stride', 1)} frames")
                print(f"reusing the export in {export_dir} ({len(meta['frames'])} frames, "
                      f"{existing}); --force re-unpacks it")
            return export_dir
        if verbose:
            print(f"re-unpacking {export_dir}: {shortfall}")

    pipeline, profile, rs = open_bag(bag_path)

    try:
        device = profile.get_device()
        playback = device.as_playback()
        depth_scale = device.first_depth_sensor().get_depth_scale()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        stream_fps = float(color_profile.fps())

        if verbose:
            expected = bag_frame_count(bag_path)
            print(f"unpacking {Path(bag_path).name}: {playback.get_duration().total_seconds():.1f} s"
                  + (f", {expected} frames" if expected else "")
                  + f" at {stream_fps:.0f} Hz -> {export_dir}")
            if expected:
                print(f"  {export_estimate(expected, clouds, cloud_stride)}"
                      + (f", clouds every {cloud_stride} frames" if clouds and cloud_stride > 1
                         else "" if clouds else ", no point clouds"))

        for name in (COLOR_DIR, DEPTH_DIR) + ((CLOUD_DIR,) if clouds else ()):
            (export_dir / name).mkdir(parents=True, exist_ok=True)

        align = rs.align(rs.stream.color)
        pointcloud = rs.pointcloud() if clouds else None

        records: list[dict] = []
        color_intrinsics = None

        while True:
            # try_wait_for_frames rather than wait_for_frames: with repeat_playback off, the end
            # of the bag arrives as a False here instead of a RuntimeError several frames deep.
            ok, frames = pipeline.try_wait_for_frames(5000)
            if not ok:
                break

            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            index = len(records)
            if color_intrinsics is None:
                color_intrinsics = rs.video_stream_profile(
                    color_frame.get_profile()).get_intrinsics()

            color_name = f"{COLOR_DIR}/{index:06d}.png"
            depth_name = f"{DEPTH_DIR}/{index:06d}.png"
            write_image(export_dir / color_name, np.asanyarray(color_frame.get_data()))
            # A 16-bit PNG, so the raw Z16 units survive; scaling to metres here would throw the
            # precision away and make the file depend on depth_scale.
            write_image(export_dir / depth_name, np.asanyarray(depth_frame.get_data()))

            cloud_name = None
            if clouds and index % cloud_stride == 0:
                cloud_name = f"{CLOUD_DIR}/{index:06d}.ply"
                # map_to before calculate, so the cloud carries the colour frame's texture.
                pointcloud.map_to(color_frame)
                points = pointcloud.calculate(depth_frame)
                points.export_to_ply(str(export_dir / cloud_name), color_frame)
                # export_to_ply reports nothing, so check the file landed. A cloud is tens of MB
                # and is the first thing to fail when the disk fills.
                if not (export_dir / cloud_name).exists():
                    raise SystemExit(
                        f"could not write {export_dir / cloud_name}. The usual cause is a full "
                        f"disk -- the clouds are ~30 MB a frame. Re-run with --no-clouds or a "
                        f"larger --cloud-stride, and --force.")

            records.append({
                "index": index,
                # Milliseconds in the bag's own domain. Pass 2 paces off the differences, so the
                # absolute epoch does not matter.
                "timestamp_ms": color_frame.get_timestamp(),
                "frame_number": color_frame.get_frame_number(),
                "color": color_name,
                "depth": depth_name,
                "cloud": cloud_name,
            })

            if verbose and index % 25 == 0:
                print(f"  frame {index}", end="\r", flush=True)

        if not records:
            raise SystemExit(f"{bag_path} yielded no complete colour+depth frame pairs")

        meta = {
            "bag": str(Path(bag_path).resolve()),
            "bag_name": Path(bag_path).stem,
            "stream_fps": stream_fps,
            "duration_s": playback.get_duration().total_seconds(),
            "depth_scale": depth_scale,
            "color_intrinsics": intrinsics_to_dict(color_intrinsics),
            "depth_aligned_to_color": True,
            "clouds": bool(clouds),
            "cloud_stride": int(cloud_stride),
            # Written last and checked by is_exported, so an export killed partway is re-done
            # rather than half-read.
            "complete": True,
            "frames": records,
        }
        (export_dir / META_NAME).write_text(json.dumps(meta, indent=2))

        if verbose:
            print(f"  unpacked {len(records)} frames to {export_dir}")
        return export_dir
    finally:
        pipeline.stop()


def default_export_dir(bag_path, root="rosbags/export") -> Path:
    """Where a bag unpacks to by default: <root>/<bag stem>/."""
    return Path(root) / Path(bag_path).stem


# =============================================================================
# 3. PASS 2 -- SERVE THE EXPORTED FRAMES
# =============================================================================
class ExportedDepth:
    """An exported depth PNG, standing in for rs.depth_frame.

    The loop asks a depth frame for get_distance(x, y) per pixel, and -- when it is measuring an
    object's distance over its whole mask rather than at one point -- for metres() once per frame.
    Satisfying those two is enough for segmentation_realsense.measure() to be reused unchanged.

    `scale` is how much the colour frame was shrunk for the model: x and y arrive in the scaled
    frame's coordinates and are mapped back onto the native depth array, which is kept at full
    resolution so no depth precision is lost to resampling.
    """

    def __init__(self, depth: np.ndarray, depth_scale: float, scale: float = COLOR_SCALE):
        self.depth = depth
        self.depth_scale = depth_scale
        self.scale = scale

    def get_distance(self, x: int, y: int) -> float:
        """Metres at a pixel of the scaled colour frame, or 0.0 where there is no depth.

        Out-of-range pixels are clamped rather than raising. The real pyrealsense2 raises here,
        and a box centre one pixel past the edge used to take whole runs down -- see the clipping
        in YoloDetector._to_detections and test_run_live's edge-box test. A replay must not
        reintroduce that failure mode by a different route.
        """
        height, width = self.depth.shape[:2]
        px = min(max(int(round(x / self.scale)), 0), width - 1)
        py = min(max(int(round(y / self.scale)), 0), height - 1)
        # 0 in a Z16 frame means "no reading here", which is what librealsense reports as 0.0 m.
        return float(self.depth[py, px]) * self.depth_scale

    def metres(self) -> np.ndarray:
        """The whole depth map in metres, at the *colour* frame's resolution.

        For measuring an object's distance over its whole mask: the mask is the colour frame's
        size, so the depth has to be too before it can be indexed by it. Done once per frame
        rather than per detection, which is why it is a method on the frame and not on the mask.

        **Built with get_distance's own index arithmetic, not with cv2.resize.** Not because a
        resize is necessarily wrong -- cv2.INTER_NEAREST happens to use floor(x / scale), which
        at COLOR_SCALE = 0.5 lands on the same pixel as get_distance's round(x / scale) for every
        column -- but because that agreement is a coincidence of this scale and this flag.
        INTER_NEAREST_EXACT uses floor((x + 0.5) / scale) and disagrees on 99.98% of pixels, and
        at other scales INTER_NEAREST disagrees too (0.3 and 0.75 were both checked). Having the
        per-pixel path and the per-mask path differ would mean the same object measured two
        slightly different ways depending on whether it had a mask -- invisible in the picture and
        present in every row of the CSV. Doing the indexing by hand makes the agreement
        structural rather than something to re-verify whenever OpenCV changes a default.
        """
        height, width = self.depth.shape[:2]
        out_h = int(round(height * self.scale))
        out_w = int(round(width * self.scale))
        if (out_h, out_w) == (height, width):
            return self.depth.astype(np.float64) * self.depth_scale

        # np.rint and Python's round both break ties to even, so these land on the same pixel
        # get_distance would have picked.
        rows = np.clip(np.rint(np.arange(out_h) / self.scale).astype(int), 0, height - 1)
        columns = np.clip(np.rint(np.arange(out_w) / self.scale).astype(int), 0, width - 1)
        # float64, not float32: get_distance's own product is float64, and at float32 the two
        # disagree in the seventh digit -- which would make "the two paths agree" a claim that is
        # almost true, and so not a claim worth making.
        return self.depth[np.ix_(rows, columns)].astype(np.float64) * self.depth_scale


class ExportedFrameSource:
    """An unpacked bag, as a frame source for segmentation_realsense.run_frames.

    Same interface as RealSenseSource: start(), grab(), explain_stall(), stop(), plus `rs` and
    `stream_fps`.

    Paced in real time off the recorded timestamps, so the replay behaves like the camera it came
    from: a frame whose moment has already passed when the loop comes back for it is skipped, the
    way the camera drops frames a slow model cannot keep up with. grab() returns None for a
    skipped frame, which the loop already accounts for by carrying its time into the next logged
    row.
    """

    def __init__(self, export_dir, scale: float = COLOR_SCALE, real_time: bool = True,
                 preload: bool = True, sleep=time.sleep, clock=time.perf_counter,
                 verbose: bool = True):
        self.export_dir = Path(export_dir)
        self.meta = read_meta(self.export_dir)
        self.frames = self.meta["frames"]
        self.scale = scale
        self.real_time = real_time
        self.preload = preload
        self.verbose = verbose
        self._decoded: dict | None = None
        # Injectable so the tests can drive the pacing without sleeping through a whole bag.
        self._sleep = sleep
        self._clock = clock

        self.stream_fps = float(self.meta.get("stream_fps") or 0.0)
        self.depth_scale = float(self.meta["depth_scale"])
        self.bag_name = self.meta.get("bag_name") or self.export_dir.name

        self.rs = None
        self.intrinsics = None
        self.index = 0          # frames taken off the table, shown or skipped
        self.shown = 0          # frames actually handed to the detector
        self.skipped = 0        # frames whose moment had passed before the loop came back
        self.rebased = False    # whether the warm-up resync below has happened yet
        self._t0 = None
        self._first_timestamp_ms = float(self.frames[0]["timestamp_ms"]) if self.frames else 0.0

        # In the constructor rather than in start(), which run_frames calls after main() has
        # already opened the RunLogger: decoding there would put its several seconds into the
        # log's elapsed_s and shift the whole x-axis of the figure.
        if self.preload:
            self._decode_all()

    # --- the interface run_frames uses ---------------------------------------
    @property
    def frame_shape(self) -> tuple[int, int]:
        """(height, width) of the colour frame the detector will be handed."""
        native = self.meta["color_intrinsics"]
        return (int(round(native["height"] * self.scale)),
                int(round(native["width"] * self.scale)))

    def describe(self) -> str:
        native = self.meta["color_intrinsics"]
        height, width = self.frame_shape
        return (f"bag {self.bag_name}, {len(self.frames)} frames at {self.stream_fps:.0f} Hz, "
                f"colour {native['width']}x{native['height']} -> {width}x{height}"
                + ("" if self.real_time else ", paced by the model rather than the clock"))

    def start(self) -> None:
        # Imported here, not at module scope, for the same reason RealSenseSource does it: the
        # tests and the --plot-from path must work without a RealSense runtime.
        import pyrealsense2 as rs

        self.rs = rs
        self.intrinsics = scaled_intrinsics(self.meta, self.scale)
        self.index = 0
        self.shown = 0
        self.skipped = 0
        self.rebased = False
        self._t0 = self._clock()

    # --- decoding up front ----------------------------------------------------
    def decoded_bytes(self) -> int:
        """How much RAM holding every frame decoded would take."""
        native = self.meta["color_intrinsics"]
        height, width = native["height"], native["width"]
        per_frame = (int(round(height * self.scale)) * int(round(width * self.scale)) * 3
                     + height * width * 2)
        return per_frame * len(self.frames)

    def _decode_all(self) -> None:
        """Decode every frame before the run starts, if it fits in RAM.

        Not an optimisation but a correctness matter for the metrics. A 1280x720 colour PNG takes
        ~13 ms to decode and its depth map ~5 ms, which on these clips is as much as the model
        itself; paying that inside the loop would put libpng's throughput into loop_ms and the
        skipped-frame count, where it would look like the model failing to keep up. A real camera
        hands over decoded frames, so decoding here first is the faithful arrangement, not a
        flattering one. inference_ms was never affected either way.
        """
        needed = self.decoded_bytes()
        if needed > PRELOAD_LIMIT_BYTES:
            # A long recording would not fit. Decoding per frame is then the only option, so say
            # what that does to the numbers rather than silently degrading them.
            self._decoded = None
            if self.verbose:
                print(f"  not preloading: {needed / 2**30:.1f} GiB of decoded frames exceeds the "
                      f"{PRELOAD_LIMIT_BYTES / 2**30:.0f} GiB limit. Frames will be decoded "
                      f"inside the loop, which adds ~20 ms to loop_ms and may skip frames that "
                      f"the model itself was fast enough for. inference_ms is unaffected.")
            return

        if self.verbose:
            print(f"  decoding {len(self.frames)} frames up front "
                  f"({needed / 2**20:.0f} MiB), so the loop measures the model and not libpng")
        # Keyed on the record's own `index`, not on position in the list. export_bag numbers
        # contiguously from 0 so the two agree today, but a meta.json holding a subrange of a
        # bag's frames -- a hand-trimmed export, say -- would make a positional cache serve the
        # wrong frame for every record, silently, while the lazy path stayed correct.
        self._decoded = {record["index"]: self._read(record) for record in self.frames}

    def _frame(self, record):
        """A frame's (scaled colour, native depth), from the preloaded cache or straight off disk."""
        if self._decoded is not None:
            return self._decoded[record["index"]]
        return self._read(record)

    def grab(self):
        """The next frame due, or None if the one that was due has been missed.

        StopIteration at the end of the bag: the loop treats it as a clean finish, and using the
        builtin keeps this module from having to import anything back out of the loop's module.
        """
        if self.index >= len(self.frames):
            raise StopIteration(f"the end of {self.bag_name}")

        record = self.frames[self.index]
        self.index += 1

        if self.real_time and not self._wait_for(record):
            self.skipped += 1
            return None

        self.shown += 1
        return self._load(record)

    def explain_stall(self, error: RuntimeError) -> str | None:
        """Nothing stalls in a replay: a RuntimeError here is a real fault, so never swallow it."""
        return None

    def stop(self) -> None:
        pass

    # --- pacing and loading ---------------------------------------------------
    def due_at(self, record) -> float:
        """Seconds from the start of the run at which this frame should be shown."""
        return (float(record["timestamp_ms"]) - self._first_timestamp_ms) / 1000.0

    def _wait_for(self, record) -> bool:
        """Sleep until this frame is due. False if its moment has already gone.

        The first frame is never skipped: its due time is 0 and the run starts at 0, so there is
        nothing to be late for, and dropping it would lose the warm-up frame the metrics rely on
        being frame 0.
        """
        if self.shown == 1 and not self.rebased:
            # Start the clock again after the warm-up frame. That frame pays for the CUDA
            # context, cuDNN's autotuning and any lazy kernel compilation -- 1.7 s on these
            # checkpoints, which is 26 frame periods of a 15 Hz bag. Without this, a fifth of a
            # 10 s clip is skipped before the model has run at its real speed even once, and a
            # model with a slower warm-up is scored on fewer frames than a faster one, which
            # makes the detection counts incomparable.
            #
            # benchmark.WARMUP_FRAMES already excludes that frame from every statistic, so
            # excluding it from the pacing too is the same decision applied consistently. It
            # only ever fires once, so genuine slowness later is still skipped, and the row is
            # still logged -- plot_run's "warm-up frame, off scale" annotation holds.
            #
            # The new baseline puts this frame one whole period away rather than due now, so it
            # waits for the stream exactly as every later frame does. Due-now would hand it a
            # loop_ms with no stream wait in it at all, and its fps -- up to 38 against a 15 Hz
            # cap -- would be the first thing plotted on every figure.
            self.rebased = True
            self._t0 = self._clock() + self._frame_period() - self.due_at(record)

        late_by = (self._clock() - self._t0) - self.due_at(record)
        if late_by > 0:
            # One frame period of slack: being a hair late on an otherwise keeping-up run should
            # not drop frames, and the camera likewise only drops when it has no buffer left.
            return late_by <= self._frame_period()
        self._sleep(-late_by)
        return True

    def _frame_period(self) -> float:
        return 1.0 / self.stream_fps if self.stream_fps > 0 else 0.0

    def _read(self, record):
        """(colour image scaled for the model, native depth array), decoded off disk."""
        color_path = self.export_dir / record["color"]
        depth_path = self.export_dir / record["depth"]

        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color is None:
            raise SystemExit(f"cannot read the exported colour frame {color_path}")
        # IMREAD_UNCHANGED, or OpenCV hands back an 8-bit 3-channel version of a 16-bit depth map
        # and every distance comes out wrong by a factor of 256.
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise SystemExit(f"cannot read the exported depth frame {depth_path}")
        if depth.dtype != np.uint16:
            raise SystemExit(
                f"{depth_path} is {depth.dtype}, not the uint16 a Z16 depth map must be; "
                f"re-unpack the bag with --force")

        if self.scale != 1.0:
            width = int(round(color.shape[1] * self.scale))
            height = int(round(color.shape[0] * self.scale))
            # INTER_AREA is the right filter for shrinking -- it averages the pixels it drops
            # rather than sampling one of them, which is what the training-time letterbox does.
            color = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)

        return color, depth

    def _load(self, record):
        """(colour image scaled for the model, depth frame stand-in, scaled intrinsics)."""
        color, depth = self._frame(record)
        return color, ExportedDepth(depth, self.depth_scale, self.scale), self.intrinsics
