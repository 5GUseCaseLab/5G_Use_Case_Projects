"""
WiFeye demo-video assets for Clipchamp
=====================================
Renders everything the demo video needs at 1920x1080, already animated where it helps, so the video is
assembled by dragging files onto a Clipchamp timeline (no animating by hand):

  video_assets/clips/      MP4 clips with the motion baked in (builds, cross-fades, CSI sweep, spotlights)
  video_assets/cards/      still cards (PNG) - also the last frame of every clip
  video_assets/overlays/   transparent full-frame PNGs to lay over your own footage (already positioned)
  video_assets/README.md   timeline: what goes where, for how long
  video_assets/narration_script.md   voice-over, timed to the clips

All numbers come from the saved experiment outputs through make_poster.load() (the same as the poster).

  python make_video_assets.py                  everything
  python make_video_assets.py --tally 5 6      extra 'live run' result card (5 of 6 correct)
  python make_video_assets.py --capture        re-capture the dashboard screenshot (replay, alarm muted)
Needs matplotlib + pillow; the MP4 clips need imageio-ffmpeg (pip install imageio-ffmpeg).
"""

import os
import sys
import math
import json
import time
import argparse
import subprocess
import threading
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import make_poster as mp

W, H, SS, FPS, XF = 1920, 1080, 2, 30, 0.5
OUT = "video_assets"
FD = r"C:\Windows\Fonts"
FONTS = {"reg": "segoeui.ttf", "light": "segoeuil.ttf", "semi": "seguisb.ttf", "bold": "segoeuib.ttf",
         "black": "seguibl.ttf", "italic": "segoeuii.ttf", "icon": "SegoeIcons.ttf", "mono": "consola.ttf"}

# palette: poster + dashboard tokens
INK, INK2, MUTED, LINE, BG = "#0f172a", "#475569", "#94a3b8", "#d3dcea", "#f3f6fb"
HA, HB_, YEL, HMUTED = mp.HEADER_A, mp.HEADER_B, mp.ON_HEADER, mp.HEADER_MUTED
STAGE = mp.STAGE_C
ALERT, OKC, GREEN, AMBER, SLATE, EDGE = "#dc2626", "#0f766e", "#16a34a", "#d97706", "#64748b", "#94a3b8"
CLS = {"fall": "#dc2626", "walk": "#2563eb", "posture": "#7c3aed", "hand": "#d97706", "empty_room": "#64748b"}
# Segoe Fluent icon code points (checked visually)
IC = {"person": 0xE77B, "camera": 0xE722, "battery": 0xE850, "wifi": 0xE701, "tower": 0xEC05, "x": 0xE711,
      "check": 0xE8FB, "warn": 0xE7BA, "home": 0xE80F, "walk": 0xE805, "router": 0xE704, "chip": 0xE964,
      "laptop": 0xE7F8, "bars": 0xE908, "grid": 0xE80A, "bell": 0xEA8F, "info": 0xE946}


def rgb(c, a=None):
    if isinstance(c, tuple):
        return c if a is None else c[:3] + (a,)
    h = c.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255 if a is None else a)


def mix(c, bg="#ffffff", a=0.14):
    """Opaque equivalent of colour c at opacity a over bg."""
    c1, c2 = rgb(c), rgb(bg)
    return tuple(int(round(c1[i] * a + c2[i] * (1 - a))) for i in range(3)) + (255,)


_FC = {}


def font(kind, px):
    key = (kind, int(px))
    if key not in _FC:
        path = os.path.join(FD, FONTS[kind])
        if not os.path.exists(path):
            path = os.path.join(FD, FONTS["reg"])
        _FC[key] = ImageFont.truetype(path, int(px))
    return _FC[key]


def tw(s, size, kind="reg"):
    return font(kind, size * SS).getlength(s) / SS


# ---------------------------------------------------------------- drawing (2x supersampled, 1080p coordinates)
class Canvas:
    def __init__(self, bg=BG, transparent=False):
        self.im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0) if transparent else rgb(bg))
        self.stack = []

    @property
    def tgt(self):
        return self.stack[-1][0] if self.stack else self.im

    def begin(self, opacity=1.0):
        """Everything drawn until end() is composited as one group at this opacity."""
        self.stack.append((Image.new("RGBA", self.im.size, (0, 0, 0, 0)), opacity))

    def end(self):
        layer, op = self.stack.pop()
        if op <= 0:
            return
        if op < 1.0:
            layer.putalpha(layer.getchannel("A").point(lambda v: int(v * op)))
        self.tgt.alpha_composite(layer)

    def _paint(self, bbox, fn, color, blur=0):
        pad = int(blur * 3 * SS) + 6
        x0, y0 = int(bbox[0] * SS) - pad, int(bbox[1] * SS) - pad
        x1, y1 = int(math.ceil(bbox[2] * SS)) + pad, int(math.ceil(bbox[3] * SS)) + pad
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(self.im.width, x1), min(self.im.height, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        m = Image.new("L", (x1 - x0, y1 - y0), 0)
        fn(ImageDraw.Draw(m), lambda x, y: (x * SS - x0, y * SS - y0))
        if blur:
            m = m.filter(ImageFilter.GaussianBlur(blur * SS))
        c = rgb(color)
        if c[3] < 255:
            m = m.point(lambda v: v * c[3] // 255)
        layer = Image.new("RGBA", m.size, c[:3] + (255,))
        layer.putalpha(m)
        self.tgt.alpha_composite(layer.crop((cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0)), dest=(cx0, cy0))

    def rrect(self, box, r=18, fill=None, outline=None, width=2, shadow=None):
        x0, y0, x1, y1 = box
        if shadow:                                   # (colour, alpha, blur, dy)
            col, al, bl, dy = shadow
            self._paint((x0, y0 + dy, x1, y1 + dy),
                        lambda d, P: d.rounded_rectangle([*P(x0, y0 + dy), *P(x1, y1 + dy)], radius=r * SS, fill=255),
                        rgb(col, int(255 * al)), blur=bl)
        if fill:
            self._paint(box, lambda d, P: d.rounded_rectangle([*P(x0, y0), *P(x1, y1)], radius=r * SS, fill=255), fill)
        if outline:
            self._paint(box, lambda d, P: d.rounded_rectangle([*P(x0, y0), *P(x1, y1)], radius=r * SS, outline=255,
                                                              width=max(1, int(round(width * SS)))), outline)

    def circle(self, cx, cy, r, fill=None, outline=None, width=2):
        b = (cx - r, cy - r, cx + r, cy + r)
        if fill:
            self._paint(b, lambda d, P: d.ellipse([*P(b[0], b[1]), *P(b[2], b[3])], fill=255), fill)
        if outline:
            self._paint(b, lambda d, P: d.ellipse([*P(b[0], b[1]), *P(b[2], b[3])], outline=255,
                                                  width=max(1, int(round(width * SS)))), outline)

    def poly(self, pts, fill=None, outline=None, width=2):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        b = (min(xs), min(ys), max(xs), max(ys))
        if fill:
            self._paint(b, lambda d, P: d.polygon([P(x, y) for x, y in pts], fill=255), fill)
        if outline:
            self._paint(b, lambda d, P: d.line([P(x, y) for x, y in pts + [pts[0]]], fill=255,
                                               width=max(1, int(round(width * SS))), joint="curve"), outline)

    def line(self, pts, color, width=3):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        b = (min(xs) - width, min(ys) - width, max(xs) + width, max(ys) + width)
        self._paint(b, lambda d, P: d.line([P(x, y) for x, y in pts], fill=255, width=max(1, int(round(width * SS))),
                                           joint="curve"), color)

    def arc(self, cx, cy, r, start, end, color, width=6):
        b = (cx - r, cy - r, cx + r, cy + r)
        self._paint(b, lambda d, P: d.arc([*P(b[0], b[1]), *P(b[2], b[3])], start, end, fill=255,
                                          width=max(1, int(round(width * SS)))), color)

    def arrow(self, pts, color, width=4, head=18):
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        ang = math.atan2(yb - ya, xb - xa)
        base = (xb - head * math.cos(ang), yb - head * math.sin(ang))
        hw = head * 0.6
        left = (base[0] + hw * math.sin(ang), base[1] - hw * math.cos(ang))
        right = (base[0] - hw * math.sin(ang), base[1] + hw * math.cos(ang))
        self.line(list(pts[:-1]) + [base], color, width)
        self.poly([(xb, yb), left, right], fill=color)

    def text(self, x, y, s, size, kind="reg", color=INK, anchor="la", opacity=1.0):
        if not s:
            return (x, y, x, y)
        f = font(kind, size * SS)
        bb = f.getbbox(s, anchor=anchor)
        box = (x + bb[0] / SS, y + bb[1] / SS, x + bb[2] / SS, y + bb[3] / SS)
        c = rgb(color)
        c = c[:3] + (int(c[3] * opacity),)
        self._paint(box, lambda d, P: d.text(P(x, y), s, font=f, fill=255, anchor=anchor), c)
        return box

    def icon(self, x, y, cp, size, color, anchor="mm"):
        return self.text(x, y, chr(cp), size, "icon", color, anchor)

    def wrap(self, s, size, kind, width):
        f = font(kind, size * SS)
        lines = []
        for block in s.split("\n"):
            cur = ""
            for wd in block.split(" "):
                t = (cur + " " + wd).strip()
                if f.getlength(t) / SS <= width or not cur:
                    cur = t
                else:
                    lines.append(cur)
                    cur = wd
            lines.append(cur)
        return lines

    def para(self, x, y, s, size, kind="reg", color=INK2, width=600, lh=1.32, align="l"):
        lines = self.wrap(s, size, kind, width)
        anchor = {"l": "la", "c": "ma", "r": "ra"}[align]
        for i, ln in enumerate(lines):
            self.text(x, y + i * size * lh, ln, size, kind, color, anchor)
        return len(lines) * size * lh

    def rich(self, x, y, parts, align="c"):
        """parts: [(text, size, kind, colour)], one baseline at y."""
        widths = [tw(t, s, k) for t, s, k, _ in parts]
        xx = x - sum(widths) / 2 if align == "c" else (x - sum(widths) if align == "r" else x)
        for (t, s, k, col), wd in zip(parts, widths):
            self.text(xx, y, t, s, k, col, anchor="ls")
            xx += wd
        return sum(widths)

    def gradient(self, box, c0, c1, r=0, angle=0.0):
        x0, y0, x1, y1 = box
        w, h = int(round((x1 - x0) * SS)), int(round((y1 - y0) * SS))
        gw, gh = max(2, w // 4), max(2, h // 4)
        xs = np.linspace(0, 1, gw, dtype=np.float32)[None, :]
        ys = np.linspace(0, 1, gh, dtype=np.float32)[:, None]
        t = xs * math.cos(angle) + ys * math.sin(angle)
        t = (t - t.min()) / (t.max() - t.min() + 1e-9)
        a, b = np.array(rgb(c0)[:3], np.float32), np.array(rgb(c1)[:3], np.float32)
        g = Image.fromarray((a + (b - a) * t[..., None]).astype(np.uint8), "RGB").resize((w, h), Image.BILINEAR)
        g = g.convert("RGBA")
        if r:
            m = Image.new("L", (w, h), 0)
            ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=int(r * SS), fill=255)
            g.putalpha(m)
        self.tgt.alpha_composite(g, dest=(int(round(x0 * SS)), int(round(y0 * SS))))

    def image(self, img, box, r=0, resample=Image.LANCZOS):
        x0, y0, x1, y1 = box
        w, h = int(round((x1 - x0) * SS)), int(round((y1 - y0) * SS))
        im = img.convert("RGBA").resize((w, h), resample)
        if r:
            m = Image.new("L", (w, h), 0)
            ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=int(r * SS), fill=255)
            a = Image.fromarray(np.minimum(np.asarray(im.getchannel("A")), np.asarray(m)))
            im.putalpha(a)
        self.tgt.alpha_composite(im, dest=(int(round(x0 * SS)), int(round(y0 * SS))))

    def dim_except(self, box, alpha=0.62, r=18):
        m = Image.new("L", self.im.size, int(255 * alpha))
        x0, y0, x1, y1 = box
        ImageDraw.Draw(m).rounded_rectangle([x0 * SS, y0 * SS, x1 * SS, y1 * SS], radius=r * SS, fill=0)
        layer = Image.new("RGBA", self.im.size, (8, 12, 28, 255))
        layer.putalpha(m)
        self.tgt.alpha_composite(layer)

    def final(self):
        return self.im.resize((W, H), Image.LANCZOS)


def save(img, path, rgba=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    (img if rgba else img.convert("RGB")).save(path, optimize=True)
    return path


# ---------------------------------------------------------------- shared pieces
HB = 150


def section(title, kicker=None, footer=True):
    c = Canvas(BG)
    c.gradient((0, 0, W, HB), HA, HB_)
    if kicker:
        c.text(80, 52, kicker.upper(), 24, "semi", YEL, anchor="ls")
        c.text(80, 116, title, 58, "bold", "#ffffff", anchor="ls")
    else:
        c.text(80, HB / 2, title, 58, "bold", "#ffffff", anchor="lm")
    c.text(W - 80, 100, "WiFeye", 46, "black", "#ffffff", anchor="rs")
    if footer:
        c.text(80, H - 16, "WiFeye  ·  EE 4032 5G Wireless Networks  ·  IIT Guwahati", 18, "reg", MUTED, anchor="ls")
    return c


def card(c, box, r=24, outline=LINE, fill="#ffffff", shadow=True):
    c.rrect(box, r, fill=fill, outline=outline, width=2,
            shadow=("#0f172a", 0.10, 14, 6) if shadow else None)


def fbox(c, x, y, w, h, title, body, color, tsize=32, bsize=25, glow=False, tint=0.12, lw=3):
    if glow:
        c.rrect((x, y, x + w, y + h), 20, shadow=(color, 0.55, 16, 0))
    c.rrect((x, y, x + w, y + h), 20, fill=mix(color, "#ffffff", tint), outline=color, width=lw)
    lines = body.split("\n") if body else []
    total = tsize * 1.22 + len(lines) * bsize * 1.3
    yt = y + h / 2 - total / 2 + 2
    c.text(x + w / 2, yt, title, tsize, "bold", color, anchor="ma")
    for i, ln in enumerate(lines):
        c.text(x + w / 2, yt + tsize * 1.22 + i * bsize * 1.3, ln, bsize, "reg", INK, anchor="ma")


def pill(c, cx, y, text, color, size=26, h=46, back=BG, pad=26):
    w = tw(text, size, "bold") + 2 * pad
    if back:
        c.rrect((cx - w / 2, y, cx + w / 2, y + h), h / 2, fill=back)
    c.rrect((cx - w / 2, y, cx + w / 2, y + h), h / 2, fill=mix(color, back or "#ffffff", 0.18),
            outline=mix(color, "#ffffff", 0.8), width=2)
    c.text(cx, y + h / 2, text, size, "bold", color, anchor="mm")


def icon_disc(c, cx, cy, r, cp, color, tint=0.14, size=None):
    c.circle(cx, cy, r, fill=mix(color, "#ffffff", tint), outline=color, width=3)
    c.icon(cx, cy, cp, size or r * 1.05, color)


def mpl_image(draw, w, h):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100 * SS, facecolor="#ffffff")
    draw(fig)
    fig.canvas.draw()
    im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy())
    plt.close(fig)
    return im


# ---------------------------------------------------------------- clips
def _writer(path):
    import imageio_ffmpeg
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = imageio_ffmpeg.write_frames(path, (W, H), fps=FPS, codec="libx264", pix_fmt_in="rgb24",
                                    pix_fmt_out="yuv420p", macro_block_size=8, quality=None,
                                    ffmpeg_log_level="error",
                                    output_params=["-crf", "18", "-preset", "medium", "-movflags", "+faststart"])
    w.send(None)
    return w


def write_clip(path, items, xf=XF):
    """items: [(PIL image, seconds)]; each item cross-fades into the next during its last xf seconds."""
    w = _writer(path)
    arrs = [np.asarray(im.convert("RGB")) for im, _ in items]
    n_total = 0
    for i, (a, (_, dur)) in enumerate(zip(arrs, items)):
        n = int(round(dur * FPS))
        nx = int(round(xf * FPS)) if i + 1 < len(arrs) else 0
        b = a.tobytes()
        for _ in range(max(0, n - nx)):
            w.send(b)
        if nx:
            a0, a1 = a.astype(np.float32), arrs[i + 1].astype(np.float32)
            for k in range(nx):
                t = (k + 1) / (nx + 1)
                t = t * t * (3 - 2 * t)
                w.send((a0 + (a1 - a0) * t).astype(np.uint8).tobytes())
        n_total += n
    w.close()
    return n_total / FPS


def write_iter(path, frames):
    w = _writer(path)
    n = 0
    for f in frames:
        w.send(np.ascontiguousarray(f).tobytes())
        n += 1
    w.close()
    return n / FPS


# ---------------------------------------------------------------- 01 hook + brand overlays
def overlay_pill(c, cx, cy, parts, size=60, pad_x=46, h=None, fill=(15, 23, 42, 200), align="c"):
    width = sum(tw(t, size, k) for t, k, _ in parts) + 2 * pad_x
    h = h or size * 1.9
    x0 = cx - width / 2 if align == "c" else cx
    c.rrect((x0, cy - h / 2, x0 + width, cy + h / 2), h / 2, fill=fill, shadow=("#000000", 0.25, 10, 4))
    c.rich(x0 + width / 2, cy + size * 0.36, [(t, size, k, col) for t, k, col in parts])
    return x0, x0 + width


def hook_overlay():
    c = Canvas(transparent=True)
    overlay_pill(c, W / 2, 900, [("No camera.  No wearable.  ", "bold", "#ffffff"), ("Just Wi-Fi.", "bold", YEL)],
                 size=62)
    return c.final()


def brand_overlay():
    c = Canvas(transparent=True)
    c.rrect((W - 250, H - 86, W - 30, H - 28), 29, fill=(15, 23, 42, 170))
    c.text(W - 140, H - 57, "WiFeye", 32, "black", "#ffffff", anchor="mm")
    return c.final()


# ---------------------------------------------------------------- 02 title
def title_frames(cfg):
    def base():
        c = Canvas()
        c.gradient((0, 0, W, H), HA, HB_, angle=0.35)
        for i, r in enumerate((300, 460, 620, 780, 940)):          # Wi-Fi wave motif
            c.arc(W / 2, H + 140, r, 220, 320, (255, 255, 255, int(255 * (0.075 - 0.011 * i))), width=12)
        logo = cfg.get("logo")
        if logo and os.path.exists(logo):
            c.circle(W / 2, 168, 84, fill="#ffffff")
            c.image(Image.open(logo), (W / 2 - 71, 168 - 71, W / 2 + 71, 168 + 71))
        return c

    def add_title(c):
        c.text(W / 2, 470, "WiFeye", 176, "black", "#ffffff", anchor="ms")

    def add_sub(c):
        c.text(W / 2, 562, "Device-free fall & activity detection with two ESP32 Wi-Fi receivers", 44, "reg",
               HMUTED, anchor="ms")

    def add_rest(c):
        c.rrect((W / 2 - 70, 610, W / 2 + 70, 616), 3, fill=YEL)
        c.text(W / 2, 690, "EE 4032  ·  5G Wireless Networks", 38, "italic", YEL, anchor="ms")
        c.text(W / 2, 776, cfg["authors"].replace("  ·  ", "   ·   "), 40, "semi", "#ffffff", anchor="ms")
        c.text(W / 2, 834, "Department of EEE, Indian Institute of Technology Guwahati", 30, "reg", HMUTED,
               anchor="ms")

    out = []
    c = base(); out.append(c.final())
    c = base(); add_title(c); out.append(c.final())
    c = base(); add_title(c); add_sub(c); out.append(c.final())
    c = base(); add_title(c); add_sub(c); add_rest(c); out.append(c.final())
    return out


# ---------------------------------------------------------------- 03 problem
def problem_frames():
    cards = [("Falls go unseen", ALERT, IC["person"], ("!", AMBER),
              "WHO: the 2nd leading cause of accidental-injury deaths. Worst when nobody sees."),
             ("Cameras", SLATE, IC["camera"], ("x", ALERT),
              "Privacy: nobody wants one at home, least of all in a bathroom."),
             ("Wearables", SLATE, IC["battery"], ("x", ALERT),
              "Help only if people remember to wear and charge them."),
             ("Wi-Fi", OKC, IC["wifi"], ("check", GREEN),
              "Already in every home, and a moving body changes its signal.")]
    cw_, g = 390, 40
    x0 = (W - (4 * cw_ + 3 * g)) / 2

    def draw_card(c, i):
        title, col, cp, (badge, bcol), body = cards[i]
        x = x0 + i * (cw_ + g)
        card(c, (x, 205, x + cw_, 715))
        cx = x + cw_ / 2
        icon_disc(c, cx, 330, 80, cp, col, size=84)
        c.circle(cx + 60, 270, 28, fill=bcol, outline="#ffffff", width=4)
        if badge == "!":
            c.text(cx + 60, 270, "!", 36, "black", "#ffffff", anchor="mm")
        else:
            c.icon(cx + 60, 271, IC[badge], 26, "#ffffff")
        c.text(cx, 470, title, 40, "bold", INK, anchor="ms")
        c.para(cx, 505, body, 30, "reg", INK2, width=cw_ - 50, align="c")

    def banner(c):
        c.gradient((230, 790, W - 230, 900), HA, HB_, r=55)
        c.icon(330, 845, IC["tower"], 50, YEL)
        c.rich(W / 2 + 30, 861, [("WiFeye:  ", 44, "black", "#ffffff"), ("sense with Wi-Fi", 44, "bold", "#ffffff"),
                                 ("   ·   ", 44, "reg", HMUTED), ("alert over 5G", 44, "bold", YEL)])

    out = []
    for k in range(6):
        c = section("Why WiFeye?", "The problem")
        for i in range(min(k, 4)):
            draw_card(c, i)
        if k == 5:
            banner(c)
        if k >= 1:
            c.text(W - 80, H - 16, "Source: WHO, Falls fact sheet (2021)", 18, "reg", MUTED, anchor="rs")
        out.append(c.final())
    return out


# ---------------------------------------------------------------- 04 CSI explainer (real data)
def motion_map(trial, rx="rx1", w=20, hop=5):
    import live_demo_engine as L
    _, pk = L.load_trial_packets(trial)
    t0 = max(v[0][0] for v in pk.values())
    t1 = min(v[0][-1] for v in pk.values())
    ts, A, R_ = pk[rx]
    trel, X, _ = L.preprocess(ts, A, t0, t1, R_)
    st = np.arange(0, len(X) - w + 1, hop)
    return trel[st + w // 2], np.array([X[s:s + w].std(0) for s in st]).T


CSI_ROWS = [("empty", "6_empty_room_t05", "Empty room", "calm: almost nothing changes", CLS["empty_room"], IC["home"]),
            ("walk", "5_walk_t13", "Walking", "steady stripes while moving", CLS["walk"], IC["walk"]),
            ("fall", "1_fall_t06", "Fall", "one sharp burst, then stillness", CLS["fall"], IC["warn"])]
CSI_HX0, CSI_HX1 = 612, 1828


def csi_rows_y():
    return [(240 + i * 262 + 12, 240 + i * 262 + 235 - 12) for i in range(3)]


def csi_images():
    import pandas as pd
    maps = {k: motion_map(t) for k, t, *_ in CSI_ROWS}
    maps = {k: (t[2:-2], M[:, 2:-2]) for k, (t, M) in maps.items()}          # drop window edge columns
    base = np.median(maps["empty"][1], axis=1, keepdims=True)
    vals = {k: np.log2(M / base) for k, (_, M) in maps.items()}
    floor = 0.9
    vmax = np.percentile(np.concatenate([v.ravel() for v in vals.values()]), 99.5)
    disp = {k: np.clip((v - floor) / (vmax - floor), 0, 1) for k, v in vals.items()}
    tf, tl = maps["fall"][0][0], maps["fall"][0][-1]
    xt = lambda s: CSI_HX0 + (s - tf) / (tl - tf) * (CSI_HX1 - CSI_HX0)
    # mark the fall where it is visible: the contiguous burst around the strongest motion column
    e = np.convolve(disp["fall"].mean(0), np.ones(9) / 9, mode="same")
    pk = int(np.argmax(e)); lo = hi = pk
    while lo > 0 and e[lo - 1] > 0.45 * e[pk]:
        lo -= 1
    while hi < len(e) - 1 and e[hi + 1] > 0.45 * e[pk]:
        hi += 1
    n = disp["fall"].shape[1]
    xa_b = CSI_HX0 + lo / n * (CSI_HX1 - CSI_HX0) - 6
    xb_b = CSI_HX0 + (hi + 1) / n * (CSI_HX1 - CSI_HX0) + 6

    def frame(full):
        c = section("What the Wi-Fi sees", "How it works")
        c.rich(80, 210, [("Every Wi-Fi packet carries ", 30, "reg", INK2), ("CSI", 30, "bold", INK),
                         (": the strength of 52 sub-frequencies. A moving body disturbs it.", 30, "reg", INK2)],
               align="l")
        c.text(1840, 210, "RX1 · darker = more motion · played ≈3× faster", 22, "reg", MUTED, anchor="rs")
        for i, (key, trial, name, sub, col, cp) in enumerate(CSI_ROWS):
            ry0 = 240 + i * 262
            mid = ry0 + 235 / 2
            icon_disc(c, 140, mid, 52, cp, col, size=54)
            c.text(220, mid - 6, name, 40, "bold", col, anchor="ls")
            c.text(220, mid + 34, sub, 24, "reg", INK2, anchor="ls")
            card(c, (600, ry0, 1840, ry0 + 235), r=16)
            if full:
                rgba = (mp.SEQMAP(disp[key][::-1]) * 255).astype(np.uint8)
                c.image(Image.fromarray(rgba, "RGBA"), (CSI_HX0, ry0 + 12, CSI_HX1, ry0 + 223), resample=Image.BILINEAR)
                if key == "fall":
                    c.rrect((xa_b, ry0 + 14, xb_b, ry0 + 221), 8, outline=ALERT, width=4)
                    c.rrect((xa_b, ry0 - 22, xa_b + 72, ry0 + 12), 8, fill=ALERT)
                    c.text(xa_b + 36, ry0 - 5, "fall", 22, "bold", "#ffffff", anchor="mm")
        for s in (1, 3, 5, 7, 9):
            c.text(xt(s), 1032, f"{s} s", 22, "reg", MUTED, anchor="ms")
        return c.final()

    return frame(False), frame(True)


def csi_clip_frames(base, full, total=24.0, start=1.0, sweep=3.5, gap=0.5):
    b, f = np.asarray(base.convert("RGB")), np.asarray(full.convert("RGB"))
    rows = csi_rows_y()
    ph = np.array(rgb(HA)[:3], np.uint8)
    for k in range(int(total * FPS)):
        t = k / FPS
        a = b.copy()
        for i, (y0, y1) in enumerate(rows):
            ts = start + i * (sweep + gap)
            frac = min(1.0, max(0.0, (t - ts) / sweep))
            if frac <= 0:
                continue
            xr = int(CSI_HX0 + frac * (CSI_HX1 - CSI_HX0))
            a[y0:y1, CSI_HX0:xr] = f[y0:y1, CSI_HX0:xr]
            if frac < 1.0:
                a[y0 - 6:y1 + 6, max(CSI_HX0, xr - 2):xr + 2] = ph
        yield a


# ---------------------------------------------------------------- 05 setup
def setup_card():
    c = section("The setup", "Hardware")
    card(c, (70, 200, 1160, 830))
    fw, fh = 1040, int(1040 / (10 / 5.6))

    def draw(fig):
        ax = fig.add_axes([0, 0, 1, 1])
        old = mp.K
        mp.K = fw / 100 / 10                     # inches per diagram unit (keeps obox text layout right)
        mp.diagram_setup_v4(ax, 27.0)
        mp.K = old
    c.image(mpl_image(draw, fw, fh), (95, 225, 95 + fw, 225 + fh))
    rows = [(IC["router"], STAGE["Clean"], "Router (transmitter)", "TP-Link · 2.4 GHz · channel 6"),
            (IC["chip"], STAGE["Acquire"], "Two receivers", "ESP32-WROOM-32, 60–90° apart"),
            (IC["laptop"], SLATE, "Two laptops", "A runs WiFeye · B relays RX2 over Wi-Fi"),
            (IC["bars"], OKC, "Traffic", "~70 CSI packets per second per ESP32"),
            (IC["grid"], STAGE["Features"], "Dataset", "509 trials × 10 s · 5 people + empty room")]
    for i, (cp, col, k, v) in enumerate(rows):
        y = 245 + i * 122
        icon_disc(c, 1250, y + 40, 40, cp, col, size=40)
        c.text(1312, y + 30, k, 32, "bold", INK, anchor="ls")
        c.text(1312, y + 70, v, 26, "reg", INK2, anchor="ls")
    c.rrect((70, 870, 1850, 1020), 24, fill=mix(HA, "#ffffff", 0.06), outline=mix(HA, "#ffffff", 0.35), width=2)
    c.icon(130, 945, IC["info"], 40, HA)
    c.para(180, 902, "Laptop B reads RX2 and relays it over the router's Wi-Fi to Laptop A, which runs the whole "
                     "pipeline every second and shows the dashboard.", 32, "reg", INK, width=1620)
    return c.final()


def label_overlay(cp, col, title, sub):
    c = Canvas(transparent=True)
    w = max(tw(title, 40, "bold"), tw(sub, 28, "reg")) + 170
    x0, y0, y1 = 70, 880, 1010
    c.rrect((x0, y0, x0 + w, y1), 26, fill=(255, 255, 255, 240), shadow=("#000000", 0.30, 12, 5))
    c.rrect((x0, y0, x0 + 14, y1), 7, fill=col)
    icon_disc(c, x0 + 78, (y0 + y1) / 2, 40, cp, col, size=40)
    c.text(x0 + 138, y0 + 58, title, 40, "bold", INK, anchor="ls")
    c.text(x0 + 138, y0 + 100, sub, 28, "reg", INK2, anchor="ls")
    return c.final()


# ---------------------------------------------------------------- 06 pipeline (flowchart, one stage at a time)
P_ORDER = ["Acquire", "Clean", "Condition", "Localise", "Features", "Classify", "Decide"]
P_CAPTION = [
    "Seven steps run once every second, the same for each receiver.",
    "Two ESP32 boards report CSI for every Wi-Fi packet: about 70 per second each.",
    "Raw captures mixed ~8 transmitters: keep only the router's packets to each board.",
    "Resample to a steady 100 Hz, remove spikes, smooth below 10 Hz, centre every subcarrier.",
    "Find when the event happened, trusting the receiver that saw it more clearly.",
    "Describe the event: shape, strength and timing (49 per receiver) + 42 vetted CSI features.",
    "One random forest asks “fall?”, another “which activity?”; both receivers are averaged.",
    "If P(fall) ≥ τ, raise the FALL alarm; otherwise report the most likely activity.",
    "End to end: a label 3.5–5 s after each movement, once the whole event has been seen."]


def pipeline_frame(active):
    c = section("How WiFeye works", "The pipeline")
    X0, CW = 70, 380
    g1 = (1780 - 4 * CW) / 3
    g2 = 86
    col = {s: X0 + i * (CW + g1) for i, s in enumerate(P_ORDER[:4])}
    col.update({"Features": X0, "Classify": X0 + CW + g2, "Decide": X0 + 2 * (CW + g2)})
    DW = 1780 - 2 * (CW + g2)
    yP1, yB1, yE1 = 178, 232, 500
    yP2, yB2, yE2 = 596, 650, 918
    hp = (yE1 - yB1 - 22) / 2

    def op(k):
        if active == 0:
            return 0.16
        if active == 8 or k == active:
            return 1.0
        return 0.45 if k < active else 0.10

    def stage_pill(s, y):
        x = col[s]
        w_ = DW if s == "Decide" else CW
        pill(c, x + w_ / 2, y, s, STAGE[s], size=26, h=44)

    for k, s in enumerate(P_ORDER, start=1):
        glow = active == k
        c.begin(op(k))
        sc = STAGE[s]
        if s in ("Acquire", "Clean", "Condition", "Localise"):
            stage_pill(s, yP1)
            x = col[s]
            if s == "Acquire":
                fbox(c, x, yB1, CW, hp, "ESP32 RX1", "CSI from every packet", sc, 30, 24, glow)
                fbox(c, x, yB1 + hp + 22, CW, hp, "ESP32 RX2", "~70 packets per second", sc, 30, 24, glow)
            elif s == "Clean":
                fbox(c, x, yB1, CW, yE1 - yB1, "Packet filter", "keep router → ESP32\npackets (≈80%)\nkeep 52 of 64 subcarriers",
                     sc, 32, 25, glow)
                for yy in (yB1 + hp / 2, yB1 + hp * 1.5 + 22):
                    c.arrow([(col["Acquire"] + CW + 6, yy), (x - 6, (yB1 + yE1) / 2 + (yy - (yB1 + yE1) / 2) * 0.35)],
                            "#475569", 4)
            elif s == "Condition":
                fbox(c, x, yB1, CW, yE1 - yB1, "Clean-up", "100 Hz grid\nHampel filter (spikes)\n10 Hz low-pass\nmedian removal",
                     sc, 32, 25, glow)
                c.arrow([(col["Clean"] + CW + 6, (yB1 + yE1) / 2), (x - 6, (yB1 + yE1) / 2)], "#475569", 4)
            else:
                fbox(c, x, yB1, CW, yE1 - yB1, "Fused localiser", "weights each receiver\nby how clearly it saw\nthe burst → onset,\npeak and offset",
                     sc, 32, 25, glow)
                c.arrow([(col["Condition"] + CW + 6, (yB1 + yE1) / 2), (x - 6, (yB1 + yE1) / 2)], "#475569", 4)
        elif s == "Features":
            stage_pill(s, yP2)
            x = col[s]
            fbox(c, x, yB2, CW, hp, "Event + window", "49 features per receiver", sc, 30, 24, glow)
            fbox(c, x, yB2 + hp + 22, CW, hp, "CSI domain", "42 vetted features", sc, 30, 24, glow)
            lx = col["Localise"] + CW / 2
            fx = x + CW / 2
            ym = (yE1 + yP2) / 2
            c.arrow([(lx, yE1 + 6), (lx, ym), (fx, ym), (fx, yP2 - 6)], "#475569", 4)
        elif s == "Classify":
            stage_pill(s, yP2)
            x = col[s]
            fbox(c, x, yB2, CW, hp, "Fall head", "random forest per receiver", sc, 30, 24, glow)
            fbox(c, x, yB2 + hp + 22, CW, hp, "Activity head", "random forest per receiver", sc, 30, 24, glow)
            fa, fb = yB2 + hp / 2, yB2 + hp * 1.5 + 22
            c.arrow([(col["Features"] + CW + 6, fa), (x - 6, fa)], "#475569", 4)
            c.arrow([(col["Features"] + CW + 6, fb), (x - 6, fb)], "#475569", 4)
            c.arrow([(col["Features"] + CW + 6, fa + 22), (x - 6, fb - 22)], "#475569", 3, head=15)
        else:                                        # Decide
            stage_pill(s, yP2)
            x = col[s]
            ym = (yB2 + yE2) / 2
            dcx, dh = x + 250, 124
            if glow:
                c.poly([(dcx, ym - dh), (dcx + dh, ym), (dcx, ym + dh), (dcx - dh, ym)], fill=rgb(sc, 120))
            c.poly([(dcx, ym - dh), (dcx + dh, ym), (dcx, ym + dh), (dcx - dh, ym)], fill=mix(sc, "#ffffff", 0.12),
                   outline=sc, width=3)
            c.text(dcx, ym - 6, "P(fall)", 30, "bold", INK, anchor="ms")
            c.text(dcx, ym + 32, "≥ τ ?", 30, "bold", INK, anchor="ms")
            fa, fb = yB2 + hp / 2, yB2 + hp * 1.5 + 22
            cx_r = col["Classify"] + CW + 6
            c.arrow([(cx_r, fa), (dcx - dh - 6, ym - 14)], "#475569", 4)
            c.arrow([(cx_r, fb), (dcx - dh - 6, ym + 14)], "#475569", 4)
            lx = (cx_r + dcx - dh) / 2
            c.text(lx, yB2 + 6, "late fusion", 22, "semi", INK2, anchor="ma")
            bx, bw = dcx + dh + 96, x + DW - (dcx + dh + 96)
            fbox(c, bx, yB2, bw, hp, "FALL ALERT", "alarm + beep", ALERT, 34, 24, glow, tint=0.22, lw=4)
            fbox(c, bx, yB2 + hp + 22, bw, hp, "Activity", "walk · posture · hand · empty", OKC, 32, 24, glow)
            ex = dcx + dh + 40
            c.line([(dcx + dh + 4, ym), (ex, ym)], "#475569", 4)
            c.arrow([(ex, ym), (ex, fa), (bx - 6, fa)], ALERT, 4)
            c.arrow([(ex, ym), (ex, fb), (bx - 6, fb)], OKC, 4)
            c.text(ex - 12, (ym + fa) / 2, "yes", 24, "bold", ALERT, anchor="rm")
            c.text(ex - 12, (ym + fb) / 2, "no", 24, "bold", OKC, anchor="rm")
        c.end()

    # caption bar
    c.rrect((70, 950, 1850, 1040), 22, fill="#ffffff", outline=LINE, width=2, shadow=("#0f172a", 0.08, 10, 4))
    cap = P_CAPTION[active]
    if 1 <= active <= 7:
        s = P_ORDER[active - 1]
        c.circle(122, 995, 28, fill=STAGE[s])
        c.text(122, 995, str(active), 30, "black", "#ffffff", anchor="mm")
        size = 31
        while tw(s + "  ", size, "bold") + tw(cap, size, "reg") > 1640 and size > 22:
            size -= 1
        c.rich(170, 1006, [(s + "  ", size, "bold", STAGE[s]), (cap, size, "reg", INK)], align="l")
    else:
        c.icon(122, 995, IC["info"], 40, HA)
        c.text(170, 1006, cap, 31, "semi", INK, anchor="ls")
    return c.final()


# ---------------------------------------------------------------- 07 dashboard tour (real screenshot + spotlights)
TOUR = [((20, 110, 1323, 327), "What WiFeye detects now, and how long after the movement it decided", "below"),
        ((1340, 110, 1899, 327), "Fall risk: P(fall) against the alarm threshold τ", "below"),
        ((20, 345, 1323, 703), "Each receiver: packets per second and its live CSI waterfall", "below"),
        ((1340, 345, 1899, 703), "How sure the model is about each class", "below"),
        ((20, 720, 1323, 1032), "Motion over the last 12 s, in multiples of the quiet level", "above"),
        ((1340, 720, 1899, 1032), "Every decision: time, class, P(fall) and receivers used", "above")]


def tour_frames(shot):
    shot = shot.convert("RGB").resize((W, H), Image.LANCZOS)
    out = []
    c = Canvas(); c.image(shot, (0, 0, W, H)); out.append(c.final())
    for i, (box, label, where) in enumerate(TOUR, start=1):
        c = Canvas()
        c.image(shot, (0, 0, W, H))
        c.dim_except(box)
        x0, y0, x1, y1 = box
        c.rrect((x0 - 5, y0 - 5, x1 + 5, y1 + 5), 22, outline=YEL, width=5)
        size, h = 30, 66
        pw = tw(label, size, "semi") + 120
        cx = min(max((x0 + x1) / 2, pw / 2 + 30), W - pw / 2 - 30)
        py = y1 + 22 if where == "below" else y0 - 22 - h
        c.rrect((cx - pw / 2, py, cx + pw / 2, py + h), h / 2, fill=(15, 23, 42, 240), outline=YEL, width=2,
                shadow=("#000000", 0.35, 10, 4))
        c.circle(cx - pw / 2 + 40, py + h / 2, 22, fill=YEL)
        c.text(cx - pw / 2 + 40, py + h / 2, str(i), 26, "black", INK, anchor="mm")
        c.text(cx - pw / 2 + 76, py + h / 2, label, size, "semi", "#ffffff", anchor="lm")
        out.append(c.final())
    return out


def capture_dashboard(path, trial_seq=("6_empty_room_t05", "5_walk_t13", "5_sit_t02", "3_hand_t16", "1_fall_t06"),
                      at_s=8.0, port=8061):
    """Replay recorded trials into the real dashboard (alarm muted) and screenshot it with headless Edge."""
    import joblib
    import live_demo_engine as L
    import ui_dashboard
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    prof = os.path.abspath(os.path.join(OUT, "_edge_profile"))
    bundle = joblib.load("data/models/safesense_live_bundle.pkl")
    feed = ui_dashboard.UiFeed("starting")
    ui_dashboard.start_server(feed, port, open_browser=False)
    procs = []
    for k, tid in enumerate(trial_seq):
        eng = L.LiveEngine(bundle, calibrate_s=0.0, quiet=True)
        feed.attach(eng, f"Replay · recorded trial · {tid}")
        if k == len(trial_seq) - 1:
            threading.Timer(at_s, lambda: procs.append(subprocess.Popen(
                [edge, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1920,1080",
                 "--force-device-scale-factor=1", f"--user-data-dir={prof}", "--no-first-run",
                 f"--screenshot={os.path.abspath(path)}", "--virtual-time-budget=1500", f"http://127.0.0.1:{port}/"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))).start()
        L.replay(eng, tid, speed=1.0, show=False, ui=feed)
        time.sleep(2.0)
    for p in procs:
        p.wait(timeout=30)


# ---------------------------------------------------------------- 08 / 09 overlays for the footage
DOING = [("empty", "Empty room", "No activity", CLS["empty_room"], IC["home"]),
         ("walk", "Walk across", "Walking", CLS["walk"], IC["walk"]),
         ("sit", "Sit down", "Posture change", CLS["posture"], IC["person"]),
         ("stand", "Stand up", "Posture change", CLS["posture"], IC["person"]),
         ("hand", "Wave a hand", "Hand gesture", CLS["hand"], IC["person"]),
         ("fall", "Fall (onto the mattress)", "FALL alarm", CLS["fall"], IC["warn"])]


def doing_overlay(name, expect, col, cp):
    c = Canvas(transparent=True)
    w = max(tw(name, 46, "bold"), tw("expected: " + expect, 25, "semi")) + 150
    x0, y0, y1 = 40, 36, 186
    c.rrect((x0, y0, x0 + w, y1), 28, fill=(15, 23, 42, 215), shadow=("#000000", 0.3, 10, 4))
    c.circle(x0 + 62, (y0 + y1) / 2, 38, fill=col)
    c.icon(x0 + 62, (y0 + y1) / 2, cp, 38, "#ffffff")
    c.text(x0 + 120, y0 + 38, "DOING", 22, "semi", "#cbd5e1", anchor="ls")
    c.text(x0 + 120, y0 + 92, name, 46, "bold", "#ffffff", anchor="ls")
    c.text(x0 + 120, y0 + 130, "expected: " + expect, 25, "semi", mix(col, "#ffffff", 0.45), anchor="ls")
    return c.final()


def note_overlay(parts, cy=985, size=31, cp=None, cp_col=YEL):
    c = Canvas(transparent=True)
    width = sum(tw(t, size, k) for t, k, _ in parts) + (70 if cp else 0) + 80
    h = 80
    x0 = W / 2 - width / 2
    c.rrect((x0, cy - h / 2, x0 + width, cy + h / 2), h / 2, fill=(15, 23, 42, 215), shadow=("#000000", 0.3, 10, 4))
    x = x0 + 40
    if cp:
        c.icon(x + 22, cy, cp, 34, cp_col)
        x += 70
    c.rich(x, cy + size * 0.36, [(t, size, k, col) for t, k, col in parts], align="l")
    return c.final()


def badge_overlay(text, col, x0=None, y=72, dot=True, size=28):
    c = Canvas(transparent=True)
    w = tw(text, size, "bold") + (78 if dot else 48)
    x0 = W - 40 - w if x0 is None else x0
    c.rrect((x0, y - 28, x0 + w, y + 28), 28, fill=(15, 23, 42, 215))
    if dot:
        c.circle(x0 + 32, y, 10, fill=col)
        c.text(x0 + 54, y, text, size, "bold", "#ffffff", anchor="lm")
    else:
        c.text(x0 + w / 2, y, text, size, "bold", "#ffffff", anchor="mm")
    return c.final()


def replay_banner():
    c = Canvas(transparent=True)
    text = "REPLAY  ·  recorded trials of a person the model never saw"
    w = tw(text, 30, "bold") + 110
    y = H - 72                                   # bottom: the dashboard's FALL banner lives at the top centre
    c.rrect((W / 2 - w / 2, y - 32, W / 2 + w / 2, y + 32), 32, fill=rgb(AMBER, 235), shadow=("#000000", 0.3, 10, 4))
    c.icon(W / 2 - w / 2 + 44, y, IC["info"], 30, "#ffffff")
    c.text(W / 2 - w / 2 + 78, y, text, 30, "bold", "#ffffff", anchor="lm")
    return c.final()


def skip_overlay(sec):
    c = Canvas(transparent=True)
    text = f"»  {sec} s skipped"
    w = tw(text, 28, "bold") + 56
    c.rrect((W / 2 - w / 2, 30, W / 2 + w / 2, 86), 28, fill=(15, 23, 42, 200))
    c.text(W / 2, 58, text, 28, "bold", "#ffffff", anchor="mm")
    return c.final()


def tally_card(k, n, runs=1):
    c = section("Live run" if runs == 1 else "Live runs", "Result")
    ok = k == n
    col = OKC if k >= n - 1 else AMBER
    card(c, (360, 230, 1560, 900))
    c.text(W / 2, 560, f"{k} / {n}", 240, "black", col, anchor="ms")
    c.text(W / 2, 660, "activities detected correctly", 46, "bold", INK, anchor="ms")
    how = "one continuous live run" if runs == 1 else f"{runs} live runs, {runs} people"
    c.text(W / 2, 720, how + "  ·  no cuts between an action and its label", 30, "reg", INK2, anchor="ms")
    if not ok:
        c.text(W / 2, 800, "misses are shown as they happened, not edited out", 26, "italic", MUTED, anchor="ms")
    return c.final()


def replay_intro_card():
    c = section("A fair test", "Unseen person")
    card(c, (200, 230, 1720, 900))
    icon_disc(c, W / 2, 360, 80, IC["person"], AMBER, size=84)
    c.circle(W / 2 + 60, 300, 28, fill=ALERT, outline="#ffffff", width=4)
    c.icon(W / 2 + 60, 301, IC["x"], 26, "#ffffff")
    c.text(W / 2, 520, "The model never saw this person", 52, "bold", INK, anchor="ms")
    c.para(W / 2, 560, "WiFeye is retrained without subject 4 (leave-one-subject-out), then subject 4's recorded "
                       "trials are replayed in real time through the same live engine.", 32, "reg", INK2,
           width=1300, align="c")
    c.rrect((W / 2 - 330, 740, W / 2 + 330, 810), 18, fill="#0f172a")
    c.text(W / 2, 775, "run_replay_demo.bat loso 4", 32, "mono", "#e2e8f0", anchor="mm")
    return c.final()


# ---------------------------------------------------------------- 10 results
def results_cards(R):
    adm = R["adres"]["metrics"]["ADOPTED"]
    ops = R["adres"]["fall_operating_points"]
    o3, o10 = ops["0.03"], ops["0.1"]
    res = R["adrx"]["results"]
    tests = {t["b"]: t for t in R["adrx"]["tests"]}
    Lr, r1, r2 = res["LATE (2 ESP32)"], res["RX1 only"], res["RX2 only"]
    cm = R["cm5"]
    rec = cm.diagonal() / cm.sum(1)
    order = np.argsort(-rec)
    fi = mp.CL5.index("fall")
    off = cm[fi].copy(); off[fi] = -1
    conf = {"posture": "posture change (sit/stand)", "walk": "walk", "hand": "hand gesture",
            "empty_room": "empty room"}[mp.CL5[int(np.argmax(off))]]
    p0, p_last = R["progression"][0][1], R["progression"][-1][1]
    best = ", ".join(f"{mp.NICE5[mp.CL5[j]].lower()} {rec[j]:.0%}" for j in order[:3])
    FSC = 22
    cards = [
        dict(title="Cleaning the data mattered most", big=f"{p0:.2f} → {p_last:.2f}", bigc=mp.ACC_INK,
             biglab="fall AUC  (0.5 = chance)",
             take="Every raw capture mixed about eight transmitters. Keeping only the router's packets to each "
                  "board lifted fall detection from chance, more than any change of model.",
             draw=lambda fig: mp.plot_progression(fig.add_axes([0.13, 0.17, 0.84, 0.76]), R, FSC)),
        dict(title=f"Five activities: {adm['accuracy']:.0%} correct", big=f"{adm['accuracy']:.1%}", bigc=mp.ACC_INK,
             biglab="accuracy over 5 classes  (chance: 20%)",
             take=f"Best recognised: {best}. Missed falls are mostly taken for a {conf}.",
             draw=lambda fig: mp.plot_confusion(fig.add_axes([0.16, 0.12, 0.80, 0.84]), R, FSC, key="cm5",
                                                classes=mp.CL5, nice=mp.NICE5)),
        dict(title="Falls: the hardest part", big=f"{o3['sensitivity']:.0%}", bigc=ALERT,
             biglab=f"of a new person's falls caught at {o3['adl_far']:.1%} false alarms",
             take=f"Operating points on the curve: A {o3['adl_far']:.1%} → {o3['sensitivity']:.0%}, "
                  f"B {ops['0.05']['adl_far']:.1%} → {ops['0.05']['sensitivity']:.0%}, "
                  f"C {o10['adl_far']:.1%} → {o10['sensitivity']:.0%} (false alarms → falls caught). Falls vary "
                  f"most between people: more subjects and real falls are the next step.",
             draw=lambda fig: mp.plot_roc_v3(fig.add_axes([0.17, 0.12, 0.68, 0.83]), R, FSC)),
        dict(title="One ESP32 or two?", big=f"{Lr['macro_f1']:.2f} vs {r1['macro_f1']:.2f}", bigc=OKC,
             biglab="macro-F1: two receivers vs the better one alone",
             take=f"About the same overall (p = {tests['RX1']['p_holm']:.2f}), but two spot an empty room better "
                  f"(F1 {Lr['per_class_f1']['empty_room']:.2f} vs {r1['per_class_f1']['empty_room']:.2f}) and keep "
                  f"working if one fails. The weaker receiver alone: {r2['macro_f1']:.2f}.",
             draw=lambda fig: mp.plot_receivers_adopted(fig.add_axes([0.08, 0.16, 0.38, 0.72]),
                                                        fig.add_axes([0.58, 0.16, 0.38, 0.72]), R, FSC))]
    out = []
    for i, d in enumerate(cards, start=1):
        c = section("Results", "On people the model never saw")
        card(c, (70, 190, 1100, 1030))
        c.image(mpl_image(d["draw"], 990, 800), (90, 210, 1080, 1010))
        x, wd = 1160, 690
        c.text(x, 240, f"RESULT {i} OF 4", 24, "semi", INK2, anchor="ls")
        y = 268
        y += c.para(x, y, d["title"], 46, "bold", INK, width=wd, lh=1.18) + 40
        c.text(x, y + 100, d["big"], 104 if len(d["big"]) < 9 else 84, "black", d["bigc"], anchor="ls")
        y += 128
        y += c.para(x, y, d["biglab"], 28, "semi", INK2, width=wd) + 34
        c.rrect((x, y, x + 90, y + 6), 3, fill=d["bigc"])
        y += 30
        c.para(x, y, d["take"], 31, "reg", INK, width=wd, lh=1.36)
        c.para(x, 968, "Leave-one-subject-out: trained on 4 people, tested on the 5th, repeated for all 5.", 21,
               "reg", MUTED, width=wd)
        out.append(c.final())
    return out


# ---------------------------------------------------------------- 11 future: edge + 5G (build)
def future_frames():
    def frame(step):
        c = section("Next: WiFeye on the edge + 5G", "Future work  ·  proposed")
        # step 1: sensors + Arduino UNO Q
        c.begin(1.0 if step >= 1 else 0.0)
        fbox(c, 80, 330, 250, 120, "ESP32", "RX1", STAGE["Acquire"], 32, 26)
        fbox(c, 80, 600, 250, 120, "ESP32", "RX2", STAGE["Acquire"], 32, 26)
        c.rrect((420, 205, 1150, 900), 28, fill=mix(HA, BG, 0.05), outline=HA, width=3)
        pill(c, 610, 182, "Arduino UNO Q", HA, size=28, h=48)
        c.arrow([(336, 390), (414, 390)], "#475569", 4)
        c.arrow([(336, 660), (414, 660)], "#475569", 4)
        c.text(375, 372, "UART", 20, "semi", INK2, anchor="ms")
        c.text(375, 642, "Wi-Fi", 20, "semi", INK2, anchor="ms")
        fbox(c, 460, 260, 650, 110, "STM32U585 MCU", "capture · filter · timestamps", STAGE["Clean"], 30, 24)
        c.arrow([(785, 374), (785, 418)], "#475569", 4)
        c.text(805, 402, "bridge", 20, "semi", INK2, anchor="ls")
        c.rrect((460, 422, 1110, 762), 20, fill=mix(EDGE, "#ffffff", 0.10), outline=EDGE, width=2)
        c.text(785, 462, "QRB2210 · Linux: the same pipeline", 28, "bold", INK, anchor="ms")
        chips = [("Clean-up", "Condition"), ("Localiser", "Localise"), ("Features", "Features"),
                 ("Two heads", "Classify")]
        for i, (t, s) in enumerate(chips):
            xx = 480 + (i % 2) * 315
            yy = 490 + (i // 2) * 134
            fbox(c, xx, yy, 295, 116, t, "", STAGE[s], 30, 24)
        fbox(c, 460, 790, 315, 82, "Buzzer + LED", "", ALERT, 27, 22)
        fbox(c, 795, 790, 315, 82, "Sleep when quiet", "", SLATE, 27, 22)
        c.end()
        # step 2: 5G modem + server
        c.begin(1.0 if step >= 2 else 0.0)
        fbox(c, 1260, 215, 580, 110, "5G modem", "", HA, 34, 24, tint=0.2)
        c.arrow([(1156, 592), (1205, 592), (1205, 270), (1254, 270)], "#475569", 4)
        c.text(1190, 440, "USB", 20, "semi", INK2, anchor="rs")
        c.arrow([(1550, 330), (1550, 404)], HA, 5)
        c.text(1572, 376, "5G NR", 24, "bold", HA, anchor="ls")
        fbox(c, 1260, 408, 580, 110, "Edge / cloud server", "", SLATE, 32, 24)
        c.end()
        # step 3: applications + 5G service classes
        c.begin(1.0 if step >= 3 else 0.0)
        fbox(c, 1260, 600, 580, 120, "Fall alert", "URLLC → carer's phone", ALERT, 34, 26, tint=0.2)
        fbox(c, 1260, 780, 580, 120, "Activity log", "mMTC → care dashboard", OKC, 34, 26)
        c.arrow([(1550, 522), (1550, 594)], ALERT, 4)
        c.arrow([(1844, 463), (1880, 463), (1880, 840), (1846, 840)], OKC, 4)
        c.rrect((80, 935, 940, 1035), 22, fill=mix(ALERT, "#ffffff", 0.08), outline=mix(ALERT, "#ffffff", 0.5), width=2)
        c.text(110, 975, "URLLC: urgent fall alarms", 30, "bold", ALERT, anchor="ls")
        c.text(110, 1015, "IMT-2020 targets: 1 ms latency, 99.999% reliability", 24, "reg", INK2, anchor="ls")
        c.rrect((980, 935, 1840, 1035), 22, fill=mix(OKC, "#ffffff", 0.08), outline=mix(OKC, "#ffffff", 0.5), width=2)
        c.text(1010, 975, "mMTC: routine activity logs", 30, "bold", OKC, anchor="ls")
        c.text(1010, 1015, "many low-power sensors: up to 10⁶ devices per km²", 24, "reg", INK2, anchor="ls")
        c.end()
        return c.final()
    return [frame(1), frame(2), frame(3)]


# ---------------------------------------------------------------- 12 limits, 13 credits
def limits_card(R):
    o3 = R["adres"]["fall_operating_points"]["0.03"]
    c = section("What we showed, and its limits", "Conclusion")
    cols = [("What we showed", OKC, IC["check"],
             ["A router + two ESP32s detect falls and daily activities: no camera, nothing to wear.",
              "Cleaning the CSI mattered more than the choice of model.",
              "It runs live, with a label 3.5–5 s after each movement."]),
            ("Limits & next steps", AMBER, IC["warn"],
             ["One room, 5 people, staged falls onto a mattress.",
              f"Falls of a new person: {o3['sensitivity']:.0%} caught at {o3['adl_far']:.1%} false alarms.",
              "Next: more people and rooms, an Arduino UNO Q edge node, 5G alerts."])]
    for i, (head, col, cp, items) in enumerate(cols):
        x0 = 80 + i * 900
        card(c, (x0, 215, x0 + 860, 990))
        c.rrect((x0 + 28, 211, x0 + 832, 221), 5, fill=col)
        c.text(x0 + 50, 318, head, 50, "bold", col, anchor="ls")
        y = 392
        for it in items:
            c.circle(x0 + 82, y + 27, 30, fill=mix(col, "#ffffff", 0.16), outline=col, width=3)
            c.icon(x0 + 82, y + 28, cp, 30, col)
            hgt = c.para(x0 + 138, y, it, 38, "reg", INK, width=680, lh=1.3)
            y += max(hgt, 66) + 58
    return c.final()


def credits_card(cfg):
    c = Canvas()
    c.gradient((0, 0, W, H), HA, HB_, angle=0.35)
    for i, r in enumerate((300, 460, 620, 780, 940)):
        c.arc(W / 2, H + 140, r, 220, 320, (255, 255, 255, int(255 * (0.075 - 0.011 * i))), width=12)
    logo = cfg.get("logo")
    if logo and os.path.exists(logo):
        c.circle(W / 2, 170, 86, fill="#ffffff")
        c.image(Image.open(logo), (W / 2 - 72, 170 - 72, W / 2 + 72, 170 + 72))
    c.text(W / 2, 400, "Thank you", 130, "black", "#ffffff", anchor="ms")
    c.text(W / 2, 500, cfg["authors"].replace("  ·  ", "   ·   "), 42, "semi", "#ffffff", anchor="ms")
    c.text(W / 2, 566, "EE 4032  ·  5G Wireless Networks  ·  IIT Guwahati", 34, "italic", YEL, anchor="ms")
    c.para(W / 2, 640, cfg.get("acknowledgement", ""), 30, "reg", HMUTED, width=1300, align="c")
    c.text(W / 2, 1000, "Code, poster and video: WiFeye_Device-Free_Fall_and_Activity_Detection_with_Two_ESP32_WiFi_Receivers",
           22, "reg", HMUTED, anchor="ms")
    return c.final()


# ---------------------------------------------------------------- narration + timeline
def write_docs(R, durations):
    adm = R["adres"]["metrics"]["ADOPTED"]
    o3, o10 = R["adres"]["fall_operating_points"]["0.03"], R["adres"]["fall_operating_points"]["0.1"]
    res = R["adrx"]["results"]
    Lr, r1 = res["LATE (2 ESP32)"], res["RX1 only"]
    p0, p_last = R["progression"][0][1], R["progression"][-1][1]
    d = durations
    segs = [
        ("1. Hook", "~8 s", "your fall clip + `overlays/01_hook_text.png`",
         "*(no voice: let the beep play)*"),
        ("2. Title", f"{d['02']:.0f} s", "`clips/02_title.mp4`",
         "This is WiFeye: device-free fall and activity detection with two ESP32 Wi-Fi receivers. "
         "Our project for EE 4032, 5G Wireless Networks."),
        ("3. Why WiFeye", f"{d['03']:.0f} s", "`clips/03_problem.mp4`",
         "Falls are the second leading cause of accidental-injury deaths worldwide, and they are most dangerous "
         "when someone lives alone and nobody sees them. Cameras could watch, but nobody wants a camera at home, "
         "least of all in a bathroom. Wearables help only if people remember to wear and charge them. Wi-Fi is "
         "already in every home, and a moving body changes the Wi-Fi signal. So WiFeye senses with Wi-Fi, and "
         "alerts over 5G."),
        ("4. What the Wi-Fi sees", f"{d['04']:.0f} s", "`clips/04_csi.mp4`",
         "Every Wi-Fi packet carries channel state information, or CSI: the strength of 52 sub-frequencies. "
         "This is one receiver over ten seconds. In an empty room it stays calm. Walking leaves stripes the whole "
         "time. A fall is one sharp burst, and then stillness. WiFeye learns to tell these patterns apart."),
        ("5. The setup", "8 s card + ~15 s room footage", "`cards/05_setup.png`, then your phone pan with "
         "`overlays/05_label_*.png`",
         "Our setup: one TP-Link router on channel 6, and two ESP32 receivers placed sixty to ninety degrees "
         "apart, with the person two to three metres away. Laptop A reads RX1 and runs WiFeye; Laptop B reads RX2 "
         "and relays it to Laptop A over the same Wi-Fi."),
        ("6. The pipeline", f"{d['06']:.0f} s", "`clips/06_pipeline.mp4`",
         "WiFeye runs seven steps every second, for each receiver. Acquire: the boards report CSI for about seventy "
         "packets a second. Clean: every raw capture mixed about eight transmitters, so we keep only the router's "
         "packets to each board. Condition: resample to one hundred hertz, remove spikes and smooth. Localise: find "
         "when the event happened, trusting the receiver that saw it more clearly. Features: describe the event. "
         "Classify: one random forest asks whether it was a fall, another which activity it was. Decide: if the "
         "fall probability crosses the threshold, raise the alarm."),
        ("7. The dashboard", f"{d['07']:.0f} s", "`clips/07_dashboard_tour.mp4`",
         "This is the live dashboard. On top, what WiFeye detects now, and how long after the movement it decided. "
         "Next to it, the fall risk against the alarm threshold. Each receiver shows its packet rate and a live CSI "
         "waterfall. Then the class probabilities, the motion over the last twelve seconds, and every decision."),
        ("8. Live demo", "~75 s", "your side-by-side + `overlays/08_doing_*.png`, `08_live_badge.png`, "
         "`08_note_delay.png`, then `cards/08_tally_*.png`",
         "Here it is live. The empty room... walking... sitting down and standing up, which both count as a posture "
         "change... a hand wave... and a fall. *(once, early on:)* Each label appears about four seconds after the "
         "movement, because WiFeye waits three and a half seconds to see the whole event before it decides."),
        ("9. Robustness + fair test", "~20 s", "`overlays/09_rx2_unplugged.png` on your unplug clip; "
         "`cards/09_replay_intro.png` (3 s), then the replay footage with `overlays/09_replay_banner.png`",
         "If a receiver drops out (here we unplug RX2), WiFeye carries on with RX1 alone. And for a fair test, "
         "this is a person the model never saw during training, replayed through the same live engine."),
        ("10. Results", f"{d['10']:.0f} s", "`clips/10_results.mp4`",
         f"All results are on people the model never saw. Cleaning the data took fall detection from chance, an "
         f"AUC of {p0:.2f}, to {p_last:.2f} for the final system. Across five classes WiFeye is right "
         f"{adm['accuracy']:.0%} of the time, where chance is 20%. Falls of new people are the hardest part: at about "
         f"{o3['adl_far']:.0%} false alarms it catches {o3['sensitivity']:.0%} of them, and {o10['sensitivity']:.0%} "
         f"at about {o10['adl_far']:.0%}. Two receivers match the better single one overall, macro-F1 "
         f"{Lr['macro_f1']:.2f} versus {r1['macro_f1']:.2f}, but spot an empty room more reliably and keep the system "
         f"running if one fails."),
        ("11. Future: edge + 5G", f"{d['11']:.0f} s", "`clips/11_future.mp4`",
         "Next, we want to run the whole pipeline on an Arduino UNO Q at the edge, with a local buzzer, and send "
         "alerts over 5G: urgent fall alarms over URLLC, and routine activity logs over mMTC."),
        ("12. Conclusion", "8 s", "`cards/12_limits.png`",
         "In short: a router and two ESP32s can sense falls and activities with no camera and nothing to wear. "
         "Our limits are one room, five people and staged falls, so more people, rooms and real falls come next."),
        ("13. Credits", "6 s", "`cards/13_credits.png`", "Thank you."),
    ]
    words = sum(len(s[3].split()) for s in segs)
    with open(os.path.join(OUT, "narration_script.md"), "w", encoding="utf-8") as f:
        f.write("# WiFeye demo video: narration script\n\n")
        f.write(f"About {words} words, roughly {words / 150:.1f} minutes at a calm 150 words per minute. Numbers are "
                "the saved leave-one-subject-out results (same as the poster). Read it once aloud against the clips "
                "and trim words rather than speeding up.\n\n")
        for name, dur, asset, line in segs:
            f.write(f"## {name}  ({dur})\n\n**On screen:** {asset}\n\n{line}\n\n")
    return segs


def write_readme(durations, tally_variants):
    d = durations
    rows = [
        ("1", "Hook", "your fall clip (5–8 s)", "overlays/01_hook_text.png on top"),
        ("2", "Title", f"clips/02_title.mp4 ({d['02']:.0f} s)", ""),
        ("3", "Why WiFeye", f"clips/03_problem.mp4 ({d['03']:.0f} s)", ""),
        ("4", "What the Wi-Fi sees", f"clips/04_csi.mp4 ({d['04']:.0f} s)", "real CSI from recorded trials"),
        ("5", "Setup", "cards/05_setup.png (8 s), then your phone pan (~15 s)",
         "overlays/05_label_router / rx1 / rx2 / laptopA / laptopB.png over the pan, ~3 s each"),
        ("6", "Pipeline", f"clips/06_pipeline.mp4 ({d['06']:.0f} s)", "one stage lights up at a time"),
        ("7", "Dashboard tour", f"clips/07_dashboard_tour.mp4 ({d['07']:.0f} s)", "spotlights on each panel"),
        ("8", "Live demo", "your side-by-side (~75 s)",
         "overlays/08_doing_*.png (one per activity), 08_live_badge.png (whole section), 08_note_delay.png "
         "(once), 08_skip_*.png where you cut waiting time; end with cards/08_tally_K_of_N.png (4 s)"),
        ("9", "Robustness + fair test", "your unplug clip, cards/09_replay_intro.png (3 s), replay footage",
         "overlays/09_rx2_unplugged.png on the unplug clip; overlays/09_replay_banner.png on the replay"),
        ("10", "Results", f"clips/10_results.mp4 ({d['10']:.0f} s)", "4 result cards"),
        ("11", "Future: edge + 5G", f"clips/11_future.mp4 ({d['11']:.0f} s)", "builds in 3 steps"),
        ("12", "Conclusion", "cards/12_limits.png (8 s)", ""),
        ("13", "Credits", "cards/13_credits.png (6 s)", ""),
    ]
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write("# WiFeye demo video: assets for Clipchamp\n\n")
        f.write("Everything is 1920×1080. The animated parts are already rendered as MP4 clips, so in Clipchamp you "
                "only **drag, trim and stack**; nothing needs animating. Regenerate with "
                "`python make_video_assets.py`.\n\n")
        f.write("## Timeline (about 5 minutes)\n\n| # | Section | Main track | On top (overlay track) |\n"
                "|---|---|---|---|\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")
        f.write("\n## In Clipchamp\n\n"
                "1. **Create a new video**, set the ratio to **16:9**, then drag the whole `video_assets` folder and "
                "your recordings into *Your media*.\n"
                "2. Drag the main-track items onto the timeline in the order above. Clips already contain their "
                "animation; cards (PNG) default to a few seconds: drag their right edge to the length in the table.\n"
                "3. **Overlays:** drag a PNG from `overlays/` onto the timeline *above* the footage. They are "
                "full-frame and transparent, already positioned (captions top-left over the camera half, notes at the "
                "bottom). If Clipchamp shrinks one, choose *Fill* in the toolbar so it covers the frame exactly. "
                "Drag its edges to set when it shows.\n"
                "4. **Transitions (optional, one drag):** *Transitions* tab → **Cross fade** → drop it between two "
                "sections. Inside the clips the fades are already done.\n"
                "5. **Voice-over:** record it with *Record & create → Audio* (or import your phone recording) on a "
                "track below; `narration_script.md` is timed to these clips. Keep the live beep from the demo footage.\n"
                "6. **Export** at 1080p. If the repository rejects files over 100 MB, compress with HandBrake "
                "(preset *Fast 1080p30*, RF ≈ 24) or link an unlisted upload in `Demo_Video/README.md`.\n\n")
        f.write("## Files\n\n- `clips/`: 02_title, 03_problem, 04_csi, 06_pipeline, 07_dashboard_tour, 10_results, "
                "11_future (MP4, H.264).\n- `cards/`: still versions (last frame of each clip, each pipeline stage, each "
                "result), 05_setup, 08_tally_*, 09_replay_intro, 12_limits, 13_credits.\n- `overlays/`: transparent "
                "PNGs for your footage.\n")
        f.write(f"- Tally cards made: {', '.join(f'{k}/{n}' for k, n in tally_variants)}. For another score: "
                "`python make_video_assets.py --tally K N`.\n")
        f.write("\nHonesty notes for the live section: use one continuous run, do not cut an action and a label from "
                "different takes, mark shortened waits with an `08_skip_*` overlay, and keep the ~4 s delay visible: "
                "it is how the system works.\n")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="WiFeye demo-video assets")
    ap.add_argument("--tally", nargs=2, type=int, action="append", metavar=("K", "N"),
                    help="extra live-run result card, e.g. --tally 5 6")
    ap.add_argument("--runs", type=int, default=1, help="number of live runs the --tally cards cover (default 1)")
    ap.add_argument("--capture", action="store_true", help="re-capture the dashboard screenshot first")
    ap.add_argument("--no-clips", action="store_true", help="stills only (skip MP4 encoding)")
    args = ap.parse_args()
    cfg = json.load(open("poster_config.json", encoding="utf-8"))
    for sub in ("clips", "cards", "overlays", "src"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    shot_path = os.path.join(OUT, "src", "dashboard_fall.png")
    if args.capture or not os.path.exists(shot_path):
        print("  capturing the dashboard (replay, alarm muted) ...", flush=True)
        capture_dashboard(shot_path)
    print("  loading results ...", flush=True)
    R = mp.load()
    D = {}
    clips = not args.no_clips
    J = lambda *p: os.path.join(OUT, *p)

    print("  overlays ...", flush=True)
    save(hook_overlay(), J("overlays", "01_hook_text.png"), rgba=True)
    save(brand_overlay(), J("overlays", "00_corner_brand.png"), rgba=True)
    labels = [("router", IC["router"], STAGE["Clean"], "Router  ·  transmitter", "TP-Link · 2.4 GHz · channel 6"),
              ("rx1", IC["chip"], STAGE["Acquire"], "ESP32 RX1", "receiver 1 · read by Laptop A"),
              ("rx2", IC["chip"], STAGE["Acquire"], "ESP32 RX2", "receiver 2 · read by Laptop B"),
              ("laptopA", IC["laptop"], SLATE, "Laptop A", "runs WiFeye and the dashboard"),
              ("laptopB", IC["laptop"], SLATE, "Laptop B", "relays RX2 over Wi-Fi")]
    for key, cp, col, t, s in labels:
        save(label_overlay(cp, col, t, s), J("overlays", f"05_label_{key}.png"), rgba=True)
    for i, (key, name, expect, col, cp) in enumerate(DOING, start=1):
        save(doing_overlay(name, expect, col, cp), J("overlays", f"08_doing_{i}_{key}.png"), rgba=True)
    save(badge_overlay("LIVE  ·  real time", ALERT), J("overlays", "08_live_badge.png"), rgba=True)
    save(note_overlay([("Labels appear ", "reg", "#ffffff"), ("~4 s after", "bold", YEL),
                       (" each movement: WiFeye waits 3.5 s to see the whole event.", "reg", "#ffffff")], cp=IC["info"]),
         J("overlays", "08_note_delay.png"), rgba=True)
    for s in (5, 10, 15, 20):
        save(skip_overlay(s), J("overlays", f"08_skip_{s:02d}s.png"), rgba=True)
    save(note_overlay([("RX2 unplugged  →  ", "bold", YEL), ("WiFeye keeps running on RX1 alone", "reg", "#ffffff")],
                      cp=IC["warn"], cp_col=YEL), J("overlays", "09_rx2_unplugged.png"), rgba=True)
    save(replay_banner(), J("overlays", "09_replay_banner.png"), rgba=True)

    print("  cards ...", flush=True)
    tallies = [(6, 6), (5, 6), (4, 6)]
    for k, n in tallies:
        save(tally_card(k, n), J("cards", f"08_tally_{k}_of_{n}.png"))
    for k, n in (tuple(t) for t in (args.tally or [])):
        save(tally_card(k, n, args.runs), J("cards", f"08_tally_{k}_of_{n}.png"))
        tallies.append((k, n))
    save(replay_intro_card(), J("cards", "09_replay_intro.png"))
    save(setup_card(), J("cards", "05_setup.png"))
    save(limits_card(R), J("cards", "12_limits.png"))
    save(credits_card(cfg), J("cards", "13_credits.png"))

    print("  02 title ...", flush=True)
    tf = title_frames(cfg)
    save(tf[-1], J("cards", "02_title.png"))
    if clips:
        D["02"] = write_clip(J("clips", "02_title.mp4"), [(tf[0], 0.9), (tf[1], 1.6), (tf[2], 1.5), (tf[3], 4.5)],
                             xf=0.6)
    print("  03 problem ...", flush=True)
    pf = problem_frames()
    save(pf[-1], J("cards", "03_problem.png"))
    if clips:
        D["03"] = write_clip(J("clips", "03_problem.mp4"),
                             [(pf[0], 1.2), (pf[1], 7.0), (pf[2], 6.0), (pf[3], 5.5), (pf[4], 5.5), (pf[5], 6.5)])
    print("  04 CSI ...", flush=True)
    cb, cf = csi_images()
    save(cf, J("cards", "04_csi.png"))
    if clips:
        D["04"] = write_iter(J("clips", "04_csi.mp4"), csi_clip_frames(cb, cf))
    print("  06 pipeline ...", flush=True)
    pp = [pipeline_frame(k) for k in range(9)]
    for k, im in enumerate(pp):
        save(im, J("cards", f"06_pipeline_{k}.png"))
    if clips:
        D["06"] = write_clip(J("clips", "06_pipeline.mp4"), [(pp[0], 3.0)] + [(im, 5.0) for im in pp[1:8]] + [(pp[8], 5.0)])
    print("  07 dashboard tour ...", flush=True)
    tfr = tour_frames(Image.open(shot_path))
    save(tfr[0], J("cards", "07_dashboard.png"))
    if clips:
        D["07"] = write_clip(J("clips", "07_dashboard_tour.mp4"), [(tfr[0], 2.5)] + [(im, 3.2) for im in tfr[1:]],
                             xf=0.4)
    print("  10 results ...", flush=True)
    rc = results_cards(R)
    for i, im in enumerate(rc, start=1):
        save(im, J("cards", f"10_result_{i}.png"))
    if clips:
        D["10"] = write_clip(J("clips", "10_results.mp4"), [(im, 7.5) for im in rc])
    print("  11 future ...", flush=True)
    ff = future_frames()
    save(ff[-1], J("cards", "11_future.png"))
    if clips:
        D["11"] = write_clip(J("clips", "11_future.mp4"), [(ff[0], 6.0), (ff[1], 5.5), (ff[2], 7.5)])
    defaults = {"02": 8.5, "03": 31.7, "04": 24.0, "06": 43.0, "07": 21.7, "10": 30.0, "11": 19.0}
    for k, v in defaults.items():
        D.setdefault(k, v)
    write_docs(R, D)
    write_readme(D, tallies)
    print("  done:", {k: round(v, 1) for k, v in D.items()}, flush=True)


if __name__ == "__main__":
    main()
