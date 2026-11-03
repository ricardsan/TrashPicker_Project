"""Draw the architecture overview of the robot/ pick client, as plots_for_text/architecture.png.

The layout is written out by hand and imports nothing from robot/. Run it with any interpreter that
has matplotlib:

    python plots_for_text/architecture_diagram.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "architecture.png"

WIDTH, HEIGHT = 240.0, 120.0

INK = "#1f2933"
EDGE = "#3e4c59"
LINK = "#c2410c"
STYLES = {
    "module": dict(face="#cfe2f7", edge="#2f6cb0", text=INK, round=True),
    "class": dict(face="#eef5fc", edge="#2f6cb0", text=INK, round=False),
    "reused": dict(face="#e4e7eb", edge="#7b8794", text=INK, round=False),
    "blackbox": dict(face="#1f2933", edge="#1f2933", text="#ffffff", round=True),
    "file": dict(face="#fdf3d0", edge="#b08900", text=INK, round=False),
}


def box(ax, cx, cy, w, h, title, sub=None, kind="module", title_size=11.5):
    style = STYLES[kind]
    x, y = cx - w / 2, cy - h / 2
    if style["round"]:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.2",
                               facecolor=style["face"], edgecolor=style["edge"], linewidth=1.5,
                               zorder=3)
    else:
        patch = Rectangle((x, y), w, h, facecolor=style["face"], edgecolor=style["edge"],
                          linewidth=1.5, zorder=3)
    ax.add_patch(patch)
    # Black boxes are named in words, and everything else by its name in the code.
    family = "DejaVu Sans" if kind in ("blackbox", "file") else "DejaVu Sans Mono"
    weight = "bold"
    title_y = cy + (h * 0.17 if sub else 0.0)
    ax.text(cx, title_y, title, ha="center", va="center", color=style["text"], family=family,
            fontsize=title_size, fontweight=weight, zorder=4)
    if sub:
        ax.text(cx, cy - h * 0.2, sub, ha="center", va="center", color=style["text"],
                family="DejaVu Sans", fontsize=8.8, zorder=4)


def diamond(ax, x, y, angle, filled, color=EDGE):
    """A UML diamond with its point at (x, y), lying along `angle` (radians, pointing along the line)."""
    length, half = 3.4, 1.25
    ca, sa = math.cos(angle), math.sin(angle)
    points = [(0.0, 0.0), (length / 2, half), (length, 0.0), (length / 2, -half)]
    rotated = [(x + px * ca - py * sa, y + px * sa + py * ca) for px, py in points]
    ax.add_patch(Polygon(rotated, closed=True, facecolor=color if filled else "#ffffff",
                         edgecolor=color, linewidth=1.3, zorder=5))
    return x + length * ca, y + length * sa


def arrowhead(ax, x, y, angle, color=EDGE, size=2.6, width=1.1):
    """A filled head with its tip at (x, y), pointing along `angle`."""
    ca, sa = math.cos(angle), math.sin(angle)
    base_x, base_y = x - size * ca, y - size * sa
    points = [(x, y), (base_x - width * sa, base_y + width * ca), (base_x + width * sa, base_y - width * ca)]
    ax.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor=color, linewidth=0.5,
                         zorder=5))
    return base_x, base_y


def connect(ax, points, label=None, label_at=None, start=None, head=True, dashed=False,
            color=EDGE, linewidth=1.5, ha="center"):
    """A line through `points`. `start` is None, 'composite' or 'reference': a diamond at the start."""
    points = [tuple(point) for point in points]
    if start:
        (x0, y0), (x1, y1) = points[0], points[1]
        points[0] = diamond(ax, x0, y0, math.atan2(y1 - y0, x1 - x0), start == "composite", color)
    if head:
        (x0, y0), (x1, y1) = points[-2], points[-1]
        points[-1] = arrowhead(ax, x1, y1, math.atan2(y1 - y0, x1 - x0), color)
    ax.plot([p[0] for p in points], [p[1] for p in points], color=color, linewidth=linewidth,
            linestyle=(0, (4, 3)) if dashed else "-", solid_capstyle="butt", zorder=2)
    if label:
        ax.text(*label_at, label, ha=ha, va="center", color=INK, family="DejaVu Sans", fontsize=8.8,
                zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.92))


def legend(ax, x, y):
    ax.text(x, y, "Relationships", color=INK, family="DejaVu Sans", fontsize=9.5, fontweight="bold",
            va="center")
    rows = [
        ("talks to", dict()),
        ("composite: creates and owns", dict(start="composite", head=False)),
        ("reference: uses what it is given", dict(start="reference", head=False)),
        ("file read", dict(dashed=True)),
        ("link to the robot controller", dict(color=LINK, linewidth=2.8)),
    ]
    for index, (text, options) in enumerate(rows):
        row_y = y - 4.6 * (index + 1)
        connect(ax, [(x, row_y), (x + 9.0, row_y)], **options)
        ax.text(x + 11.0, row_y, text, color=INK, family="DejaVu Sans", fontsize=8.6, va="center")

    key_y = y - 4.6 * (len(rows) + 1) - 1.5
    ax.text(x, key_y, "Components", color=INK, family="DejaVu Sans", fontsize=9.5, fontweight="bold",
            va="center")
    keys = [("module", "module in robot/"), ("class", "class in robot/"),
            ("reused", "reused from camera/"), ("blackbox", "black box"), ("file", "external file")]
    for index, (kind, text) in enumerate(keys):
        row_y = key_y - 4.6 * (index + 1)
        style = STYLES[kind]
        ax.add_patch(Rectangle((x, row_y - 1.5), 9.0, 3.0, facecolor=style["face"],
                               edgecolor=style["edge"], linewidth=1.2, zorder=3))
        ax.text(x + 11.0, row_y, text, color=INK, family="DejaVu Sans", fontsize=8.6, va="center")

    ax.text(x, key_y - 4.6 * (len(keys) + 1), "No inheritance among these.", color=INK,
            family="DejaVu Sans", fontsize=8.6, fontstyle="italic", va="center")


def main() -> None:
    fig, ax = plt.subplots(figsize=(16.0, 16.0 * HEIGHT / WIDTH))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    # Components.
    box(ax, 140, 110, 50, 13, "robot_trash_picker", "entry point: the run order")
    box(ax, 52, 110, 32, 13, "extrinsics", "camera to robot base")
    box(ax, 14, 110, 22, 13, "calibration file", "T_base_cam", kind="file", title_size=9.5)

    box(ax, 80, 80, 34, 16, "YoloDetector", "segmentation model", kind="reused")
    box(ax, 75, 56, 22, 10, "best.pt", kind="file", title_size=10)
    box(ax, 140, 80, 44, 16, "Frame selection and\noperator input", "black box", kind="blackbox",
        title_size=10.5)
    box(ax, 196, 80, 40, 14, "pick_sequence", "pick steps, drop locations")

    box(ax, 110, 48, 36, 16, "perception", "masks to targets")
    box(ax, 146, 48, 22, 14, "Snapshot", "chosen frame", kind="class", title_size=10.5)
    box(ax, 196, 54, 44, 14, "ArtCameraClient", "commands and replies", kind="class")
    box(ax, 196, 37, 24, 6, "TCP socket", kind="class", title_size=9)
    box(ax, 196, 12, 44, 16, "ROBOT CONTROLLER", "black box", kind="blackbox", title_size=11.5)

    box(ax, 110, 16, 36, 14, "Target", "position, axis, class", kind="class")
    box(ax, 62, 16, 28, 14, "Detection", "mask, box, class", kind="reused", title_size=10.5)

    # Talks to, and who creates what.
    connect(ax, [(115, 110), (68, 110)], "loads the transform", (91.5, 113.2))
    connect(ax, [(120, 103.5), (120, 94), (80, 94), (80, 88)], "creates", (100, 97), start="composite")
    connect(ax, [(140, 103.5), (140, 88)], "gets the Snapshot;\ncreates the frame source and keys",
            (140, 95.7), start="composite")
    connect(ax, [(160, 103.5), (160, 96), (196, 96), (196, 87)], "runs the picks", (178, 99))
    connect(ax, [(165, 112), (232, 112), (232, 54), (218, 54)], "creates, owns;\nsends it home",
            (222.5, 105), start="composite")

    connect(ax, [(118, 80), (97, 80)], "runs a\nframe", (107.5, 84.8), start="reference")
    connect(ax, [(176, 83), (162, 83)], "asks the\noperator", (169, 89.3))
    connect(ax, [(125, 72), (125, 56)], "measures\nthe masks", (115, 64))
    connect(ax, [(146, 72), (146, 55)], "returns", (153.5, 63.5))
    connect(ax, [(92, 44), (45, 44), (45, 103.5)], "camera to base", (68, 47.2))

    # File reads.
    connect(ax, [(36, 110), (25, 110)], "reads", (30.5, 113.2), dashed=True)
    connect(ax, [(75, 72), (75, 61)], "reads", (82, 66.5), dashed=True)

    # Inside the data: who holds what.
    connect(ax, [(146, 41), (146, 19), (128, 19)], "list of", (152.5, 30), start="composite",
            head=False)
    connect(ax, [(110, 40), (110, 23)], "creates", (103, 31.5))
    connect(ax, [(92, 16), (76, 16)], "holds", (84, 19.5), start="reference", head=False)
    connect(ax, [(176, 76), (167, 76), (167, 13), (128, 13)], "reads", (167, 30),
            start="reference", head=False)

    # The motion side.
    connect(ax, [(196, 73), (196, 61)], "home, grip, move", (208.5, 67), start="reference")
    connect(ax, [(196, 47), (196, 40)], start="composite", head=False)

    # The communication with the robot controller.
    connect(ax, [(191, 34), (191, 20)], color=LINK, linewidth=2.8)
    connect(ax, [(201, 20), (201, 34)], color=LINK, linewidth=2.8)
    ax.text(189, 27, "commands", ha="right", va="center", color=LINK, family="DejaVu Sans",
            fontsize=9.5, fontweight="bold", zorder=6)
    ax.text(203, 27, "replies", ha="left", va="center", color=LINK, family="DejaVu Sans",
            fontsize=9.5, fontweight="bold", zorder=6)

    legend(ax, 4, 92)

    fig.savefig(OUTPUT, dpi=200, facecolor="white")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
