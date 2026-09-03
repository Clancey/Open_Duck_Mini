#!/usr/bin/env python3
"""Offscreen MuJoCo previews of the Open Duck Mini animation library.

*Why this exists.* Every clip in ``experiments/animation/clips/`` and the
episodic training reference were validated **numerically** but never **watched**.
That gap shipped a reference whose angular-velocity axes were transposed and whose
lower body barely moved — invisible to every bound check, obvious the instant you
see it. This renders each clip through the actual simulation model (the *physical*
view that matters for sim2real) so a human can review the whole library.

What it produces (under ``renders/mujoco/``):

* ``<clip>.mp4`` — a side-by-side of two fixed 3/4 views (front-left + front-right)
  so both yaw and roll are visible and every clip is framed identically for
  side-by-side comparison. Each frame is labelled with the clip name and frame
  index.
* ``contact_sheet_3q_left.png`` / ``contact_sheet_3q_right.png`` — grids of a
  representative mid-motion frame from every clip, from each 3/4 angle, so the
  owner can scan the whole library at once.

Design notes:

* **Kinematic playback.** We set ``qpos`` directly and call ``mj_forward`` — no
  physics integration. This shows the *authored intent* projected onto the real
  model geometry, which is what we want to eyeball. (A policy-in-the-loop render
  is a separate, heavier concern; this is the reviewable baseline.)
* **Fixed camera.** A single ``mjvCamera`` (fixed lookat/distance/elevation, two
  azimuths) is reused for every clip and every frame, so clips are directly
  comparable.
* ``.duckanim`` clips carry only the 16 joint angles; the root is pinned at the
  ``home`` keyframe. The 59-float references additionally carry root pose, which
  we honour (converting the stored XYZW quaternion to MuJoCo's WXYZ).

Run with a Python that has ``mujoco`` + ``imageio`` + ``PIL`` (e.g. the project's
render venv):

    PY=/path/to/venv/bin/python
    $PY experiments/animation/render_clips.py --out /path/to/renders
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from open_duck_anim.joint_order import to_hw14  # noqa: E402

DEFAULT_XML = (
    "/Users/clancey/.copilot/session-state/9d7d4839-8a6c-44d0-8b98-328aebd93579/"
    "files/upstream/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/"
    "scene_flat_terrain.xml"
)
DEFAULT_CLIPS = os.path.join(os.path.dirname(__file__), "clips")

# Camera: two 3/4 views (front-left, front-right) at a fixed, comparable framing.
CAM_LOOKAT = (0.0, 0.0, 0.15)
CAM_DISTANCE = 0.95
CAM_ELEVATION = -10.0
CAM_AZIMUTHS = {"3q_left": 135.0, "3q_right": 45.0}

RENDER_W = 480
RENDER_H = 480


def _load_font(size: int):
    from PIL import ImageFont

    for name in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _label(img: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    """Return a copy of ``img`` (H,W,3 uint8) with a text banner drawn top-left."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    font = _load_font(20)
    small = _load_font(14)
    # translucent banner
    banner_h = 30
    draw.rectangle([0, 0, im.width, banner_h], fill=(0, 0, 0))
    draw.text((6, 5), text, fill=(255, 255, 255), font=font)
    if sub:
        draw.text((6, im.height - 20), sub, fill=(200, 200, 200), font=small)
    return np.asarray(im)


def _clip_trajectory(path: str):
    """Return ``(qpos_frames (N,21), fps, name, kind)`` for a clip file.

    ``kind`` is ``"duckanim"`` (16 joints, pinned root) or ``"ref59"`` (full
    59-float frame with its own root pose).
    """
    with open(path) as fh:
        data = json.load(fh)
    name = os.path.splitext(os.path.basename(path))[0]

    home_root_pos = np.array([0.0, 0.0, 0.15])
    home_root_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

    if "joints" in data and isinstance(data["joints"], dict):
        # .duckanim: 16 joints per frame, pinned root.
        fps = float(data.get("fps", 50))
        frames16 = np.asarray(data["joints"]["frames"], dtype=np.float64)
        n = frames16.shape[0]
        qpos = np.zeros((n, 21))
        qpos[:, 0:3] = home_root_pos
        qpos[:, 3:7] = home_root_quat_wxyz
        qpos[:, 7:21] = to_hw14(frames16)
        return qpos, fps, name, "duckanim"

    if "Frames" in data:
        # 59-float reference: honour the stored root pose.
        fps = float(data.get("FPS", data.get("fps", 50)))
        fr = np.asarray(data["Frames"], dtype=np.float64)
        n = fr.shape[0]
        qpos = np.zeros((n, 21))
        qpos[:, 0:3] = fr[:, 0:3]
        # stored root_quat is XYZW (scipy); MuJoCo qpos wants WXYZ.
        q_xyzw = fr[:, 3:7]
        qpos[:, 3] = q_xyzw[:, 3]
        qpos[:, 4:7] = q_xyzw[:, 0:3]
        qpos[:, 7:21] = to_hw14(fr[:, 7:23])
        return qpos, fps, name, "ref59"

    raise ValueError("unrecognised clip format: %s" % path)


def _make_camera(azimuth: float):
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = CAM_LOOKAT
    cam.distance = CAM_DISTANCE
    cam.elevation = CAM_ELEVATION
    cam.azimuth = azimuth
    return cam


def render_library(xml_path: str, clip_paths, out_dir: str, extra_refs=None):
    import mujoco
    import imageio.v2 as imageio

    os.makedirs(out_dir, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    cameras = {k: _make_camera(v) for k, v in CAM_AZIMUTHS.items()}

    all_paths = list(clip_paths) + list(extra_refs or [])
    # representative (mid-motion) frames for the contact sheets, per angle.
    sheet_frames = {k: [] for k in CAM_AZIMUTHS}
    sheet_names = []

    for idx, path in enumerate(sorted(all_paths)):
        try:
            qpos, fps, name, kind = _clip_trajectory(path)
        except Exception as exc:  # pragma: no cover - defensive
            print("  SKIP %s: %s" % (path, exc))
            continue
        n = qpos.shape[0]
        print("[%2d/%2d] %-26s %-9s %3d frames @ %2.0f fps"
              % (idx + 1, len(all_paths), name, kind, n, fps))

        mp4_path = os.path.join(out_dir, name + ".mp4")
        writer = imageio.get_writer(mp4_path, fps=int(round(fps)), macro_block_size=None)

        # pick the frame of maximum motion (largest joint deviation from frame 0)
        dev = np.linalg.norm(qpos[:, 7:21] - qpos[0, 7:21], axis=1)
        rep = int(np.argmax(dev)) if n > 1 else 0

        for i in range(n):
            data.qpos[:] = qpos[i]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            panels = []
            for angle_name, cam in cameras.items():
                renderer.update_scene(data, camera=cam)
                img = renderer.render()
                if i == rep:
                    sheet_frames[angle_name].append(
                        _label(img.copy(), name, "%s | frame %d/%d" % (kind, i, n - 1))
                    )
                panels.append(img)
            combined = np.concatenate(panels, axis=1)
            combined = _label(combined, name, "%s | frame %d/%d | %.0ffps"
                              % (kind, i, n - 1, fps))
            writer.append_data(combined)
        writer.close()
        sheet_names.append(name)

    # --- contact sheets -------------------------------------------------------
    for angle_name, frames in sheet_frames.items():
        if not frames:
            continue
        sheet = _contact_sheet(frames)
        sheet_path = os.path.join(out_dir, "contact_sheet_%s.png" % angle_name)
        imageio.imwrite(sheet_path, sheet)
        print("wrote", sheet_path)

    return sheet_names


def _contact_sheet(frames, cols: int = 6, pad: int = 4):
    """Tile labelled thumbnails into a single grid image."""
    from PIL import Image

    thumbs = []
    tw, th = 240, 240
    for f in frames:
        im = Image.fromarray(f).resize((tw, th))
        thumbs.append(np.asarray(im))
    n = len(thumbs)
    rows = (n + cols - 1) // cols
    grid = np.full(
        (rows * th + (rows + 1) * pad, cols * tw + (cols + 1) * pad, 3),
        30,
        dtype=np.uint8,
    )
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        y = pad + r * (th + pad)
        x = pad + c * (tw + pad)
        grid[y:y + th, x:x + tw] = t
    return grid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--clips", default=DEFAULT_CLIPS)
    ap.add_argument("--out", required=True, help="output directory for renders")
    ap.add_argument(
        "--ref",
        action="append",
        default=[],
        help="extra 59-float reference JSON(s) to include (repeatable)",
    )
    args = ap.parse_args()

    clip_paths = sorted(glob.glob(os.path.join(args.clips, "*.duckanim")))
    if not clip_paths:
        raise SystemExit("no .duckanim clips found under %s" % args.clips)
    print("rendering %d clips + %d reference(s) -> %s"
          % (len(clip_paths), len(args.ref), args.out))
    render_library(args.xml, clip_paths, args.out, extra_refs=args.ref)


if __name__ == "__main__":
    main()
