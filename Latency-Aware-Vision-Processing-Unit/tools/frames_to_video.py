"""Turn a folder of frames (e.g. a DriveCache scenario folder) into an MP4 the UNO Q app can play.

    python tools/frames_to_video.py "DriveCache/Road Scenario 2/YOLO inference of frames 20 min apart"
    python tools/frames_to_video.py --all DriveCache --out videos          # one MP4 per frame folder
    python tools/frames_to_video.py FOLDER --width 960 --fps 30 -o drive.mp4

Defaults are chosen for the board: 640 px wide (camera.py downsizes everything to 640 anyway, and a
1080p H.264 stream is too heavy for the UNO Q's CPU to decode in real time), 30 fps (DriveCache capture
rate), H.264 + yuv420p + a keyframe every second (cheap seeking/looping).

Frame numbers are read from the file names (frame_000123.png -> 123), so gaps are detected:
  gap <= --max-gap frames : the previous frame is repeated, so time in the video stays real time
                            (matters for every speed / TTC estimate downstream)
  larger gap              : --gaps split (default) starts a new video file (_part2, _part3 ...)
                            --gaps join  just continues (time jumps - speeds will spike there)

Git LFS: DriveCache stores the PNGs in Git LFS. If you cloned without LFS you only have 130-byte pointer
files; the tool detects that and tells you what to run, e.g.
    git lfs install
    git lfs pull --include="Road Scenario 2/YOLO inference of frames 20 min apart/**"

Encoder: system ffmpeg -> ffmpeg from `pip install imageio-ffmpeg` -> OpenCV VideoWriter (mp4v) as last resort.
"""
import argparse, concurrent.futures as cf, os, re, shutil, subprocess, sys
from pathlib import Path

import cv2
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
LFS_MAGIC = b"version https://git-lfs"


def frame_no(p):
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else None


def list_frames(folder):
    files = [p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS and p.is_file()]
    numbered = [(frame_no(p), p) for p in files]
    if all(n is not None for n, _ in numbered):
        numbered.sort(key=lambda t: (t[0], t[1].name))
    else:                                            # no numbers: plain name order, assume no gaps
        numbered = list(enumerate(sorted(files)))
    return numbered


def is_lfs_pointer(p):
    try:
        if p.stat().st_size > 1024:
            return False
        with open(p, "rb") as f:
            return f.read(len(LFS_MAGIC)) == LFS_MAGIC
    except OSError:
        return False


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def segments(numbered, max_gap, mode):
    """-> list of segments; each segment is a list of (path, repeat) where repeat >= 1."""
    segs, cur, prev = [], [], None
    for n, p in numbered:
        if prev is not None:
            gap = n - prev - 1
            if gap > max_gap and mode == "split":
                segs.append(cur); cur = []
            elif 0 < gap <= max_gap and cur:
                path, rep = cur[-1]
                cur[-1] = (path, rep + gap)          # hold the previous frame through the gap
        cur.append((p, 1))
        prev = n
    if cur:
        segs.append(cur)
    return segs


class Writer:
    def __init__(self, path, w, h, fps, crf, ffmpeg, preset):
        self.path, self.proc, self.cv = path, None, None
        if ffmpeg:
            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
                   "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                   "-g", str(int(round(fps))), "-movflags", "+faststart", str(path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            for cc in ("avc1", "mp4v"):
                vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*cc), fps, (w, h))
                if vw.isOpened():
                    self.cv = vw; break
            if self.cv is None:
                raise RuntimeError("no video encoder available: install ffmpeg or `pip install imageio-ffmpeg`")

    def write(self, img):
        if self.proc: self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
        else: self.cv.write(img)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError("ffmpeg failed")
        else:
            self.cv.release()


def load(path, size):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    if size and (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def convert(folder, out, a, ffmpeg):
    numbered = list_frames(folder)
    if a.start is not None: numbered = [t for t in numbered if t[0] >= a.start]
    if a.end is not None: numbered = [t for t in numbered if t[0] <= a.end]
    if not numbered:
        print(f"  [skip] no frames in {folder}"); return []
    stubs = sum(is_lfs_pointer(p) for _, p in numbered[:50])
    if stubs:
        rel, top = Path(folder).resolve(), None
        for d in [rel, *rel.parents]:                # path inside the git clone, which is what LFS wants
            if (d / ".git").exists():
                top = d; break
        rel = rel.relative_to(top).as_posix() if top else Path(folder).as_posix()
        print(f"  [skip] {folder}: frames are Git LFS pointer files, not images. In the DriveCache clone run:\n"
              f'           git lfs install && git lfs pull --include="{rel}/**"'
              + (f"   (inside {top})" if top else ""))
        return []
    first = cv2.imread(str(numbered[0][1]))
    h0, w0 = first.shape[:2]
    w = a.width or w0
    w -= w % 2
    h = int(round(h0 * w / w0)); h -= h % 2
    size = (w, h)
    segs = segments(numbered, a.max_gap, a.gaps)
    outs = []
    with cf.ThreadPoolExecutor(max(1, min(8, (os.cpu_count() or 2)))) as pool:
        for k, seg in enumerate(segs):
            path = Path(out) if len(segs) == 1 else Path(out).with_name(f"{Path(out).stem}_part{k+1}{Path(out).suffix}")
            path.parent.mkdir(parents=True, exist_ok=True)
            wr = Writer(path, w, h, a.fps, a.crf, ffmpeg, a.preset)
            n_out = 0
            for img, (p, rep) in zip(pool.map(lambda t: load(t[0], size), seg), seg):
                for _ in range(rep):
                    wr.write(img); n_out += 1
            wr.close()
            held = n_out - len(seg)
            print(f"  -> {path}  {len(seg)} frames"
                  + (f" (+{held} held to fill gaps)" if held else "")
                  + f", {w}x{h} @ {a.fps:g} fps = {n_out/a.fps:.1f} s, {path.stat().st_size/1e6:.1f} MB")
            outs.append(path)
    if len(segs) > 1:
        print(f"  note: {len(segs)-1} gap(s) larger than {a.max_gap} frames -> split into {len(segs)} files")
    return outs


def frame_dirs(root, min_frames):
    for d, _, files in os.walk(root):
        if ".git" in Path(d).parts:
            continue
        if sum(Path(f).suffix.lower() in EXTS for f in files) >= min_frames:
            yield Path(d)


def slug(rel):
    s = re.sub(r"YOLO inference of frames ?", "", rel.as_posix(), flags=re.I)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s or "video"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folders", nargs="*", help="folder(s) with frames")
    ap.add_argument("--all", metavar="ROOT", help="convert every folder under ROOT that holds frames")
    ap.add_argument("-o", "--out", help="output .mp4 (single folder) or output directory (--all / several folders)")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate of the capture (DriveCache: 30)")
    ap.add_argument("--width", type=int, default=640, help="output width, 0 = keep original (default 640 = what the board uses)")
    ap.add_argument("--crf", type=int, default=20, help="H.264 quality, lower = better/larger (18-26 sensible)")
    ap.add_argument("--preset", default="veryfast", help="x264 preset (veryfast is plenty for 640p)")
    ap.add_argument("--max-gap", type=int, default=5, help="fill gaps up to this many missing frames by holding")
    ap.add_argument("--gaps", choices=["split", "join"], default="split", help="what to do with larger gaps")
    ap.add_argument("--start", type=int, help="first frame number to include")
    ap.add_argument("--end", type=int, help="last frame number to include")
    ap.add_argument("--min-frames", type=int, default=10, help="--all: ignore folders with fewer frames")
    a = ap.parse_args()

    jobs = []
    if a.all:
        root = Path(a.all)
        outdir = Path(a.out or "videos")
        for d in sorted(frame_dirs(root, a.min_frames)):
            jobs.append((d, outdir / f"{slug(d.relative_to(root))}.mp4"))
    for f in a.folders:
        f = Path(f)
        if len(a.folders) == 1 and not a.all and a.out and a.out.lower().endswith(".mp4"):
            jobs.append((f, Path(a.out)))
        else:
            outdir = Path(a.out) if a.out else f.parent
            jobs.append((f, outdir / f"{slug(Path(f.parent.name) / f.name)}.mp4"))
    if not jobs:
        ap.error("give a frame folder or --all ROOT")

    ffmpeg = find_ffmpeg()
    print("encoder:", ffmpeg or "OpenCV VideoWriter (install ffmpeg or imageio-ffmpeg for H.264)")
    made = []
    for folder, out in jobs:
        print(f"{folder}")
        made += convert(folder, out, a, ffmpeg)
    print(f"\n{len(made)} video(s) written.")
    if made:
        print("Play on the board: upload through the web page, or copy and set SOURCE=file VIDEO_PATH=<file>.")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
