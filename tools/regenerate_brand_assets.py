"""Regenerates the app's launcher icon, notification icon, and wordmark
from brand/logo.png.

Usage:
    python -m tools.regenerate_brand_assets

What this does NOT do: figure out where the tooth / soundwave-arcs /
wordmark sit inside a *new* brand/logo.png. The crop boxes below are pixel
coordinates specific to the current 1500x1500 source file's layout (tooth
top-left, "BEX" monogram center, wordmark bottom, all on a flat #10162B
background). If brand/logo.png is replaced with a differently-composed
file, re-derive these boxes first (inspect the file, find the bounding
box of each element you want — see the "ใส่โลโก้จริงแทน placeholder"
session in project history for the exact process used the first time)
and edit the constants below; the rest of the pipeline (alpha-keying,
safe-zone fitting, per-density export) stays valid.

Requires: pillow (already a project dependency via pyproject.toml).
"""

import os

from PIL import Image, ImageFilter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(REPO_ROOT, "brand", "logo.png")
RES = os.path.join(REPO_ROOT, "android", "app", "src", "main", "res")

BG = (16, 22, 43)  # flat background color sampled from brand/logo.png

# Pixel crop boxes within brand/logo.png (1500x1500), specific to its
# current layout. See module docstring before editing brand/logo.png.
TOOTH_BOX = (300, 555, 575, 975)
WAVES_BOX = (985, 565, 1230, 995)
WORDMARK_BOX = (442, 970, 1166, 1128)

# Small rectangular patches (in WAVES_BOX / WORDMARK_BOX local coords,
# i.e. absolute minus the box's own origin) where a thin decorative
# connector line (that threads tooth -> letters -> wordmark underline in
# the original artwork) bleeds into an otherwise-clean crop. Erased by
# zeroing alpha before trim(). Empty list = nothing to patch.
WAVES_ERASE_PATCHES = [(985 - 985 - 2, 928 - 565, 1105 - 985, 936 - 565)]
WORDMARK_ERASE_PATCHES = [(1135 - 442, 970 - 970, 1170 - 442, 998 - 970)]

# Launcher (adaptive icon) foreground: mdpi..xxxhdpi at the 108dp-per-dp
# adaptive-icon canvas scale (NOT the 48dp legacy launcher scale).
LAUNCHER_SIZES = {
    "mipmap-mdpi": 108,
    "mipmap-hdpi": 162,
    "mipmap-xhdpi": 216,
    "mipmap-xxhdpi": 324,
    "mipmap-xxxhdpi": 432,
}
LAUNCHER_SUPERSAMPLE = 864
LAUNCHER_SAFE_RADIUS_FRACTION = 0.33  # 66% safe-zone diameter per Android's adaptive icon spec
LAUNCHER_MARGIN_FRACTION = 0.96  # shrink slightly inside the safe zone
LAUNCHER_GAP = 26  # px gap between tooth and waves at supersample scale

# Notification icon: single-color (white) alpha silhouette, per density.
NOTIF_SIZES = {
    "drawable-mdpi": 24,
    "drawable-hdpi": 36,
    "drawable-xhdpi": 48,
    "drawable-xxhdpi": 72,
    "drawable-xxxhdpi": 96,
}
NOTIF_SUPERSAMPLE = 384
NOTIF_MARGIN_FRACTION = 0.14
NOTIF_ARC_WIDTH_FRACTION = 0.34  # fraction of WAVES_BOX width kept, to isolate just the innermost arc
NOTIF_ARC_CROP_WIDTH = 46  # then hard-cropped to this width to drop the 2nd arc's fragment
NOTIF_DILATE_PX = 9  # thickens thin outline strokes so they survive downsampling to 24px
NOTIF_BLUR_PX = 1.2


def to_rgba_alpha(crop: Image.Image, bg: tuple, soft: int = 40) -> Image.Image:
    """Alpha = color distance from bg, clamped. Preserves anti-aliased edges."""
    crop = crop.convert("RGB")
    w, h = crop.size
    out = Image.new("RGBA", (w, h))
    px_in = crop.load()
    px_out = out.load()
    for y in range(h):
        for x in range(w):
            r, g, b = px_in[x, y]
            dist = abs(r - bg[0]) + abs(g - bg[1]) + abs(b - bg[2])
            a = max(0, min(255, int(dist * 255 / soft)))
            px_out[x, y] = (r, g, b, a)
    return out


def erase_patches(img: Image.Image, patches: list) -> Image.Image:
    px = img.load()
    w, h = img.size
    for x0, y0, x1, y1 in patches:
        for y in range(max(0, y0), min(h, y1)):
            for x in range(max(0, x0), min(w, x1)):
                r, g, b, _a = px[x, y]
                px[x, y] = (r, g, b, 0)
    return img


def trim(img: Image.Image) -> Image.Image:
    bbox = img.getbbox()
    return img.crop(bbox) if bbox else img


def whiten(img: Image.Image) -> Image.Image:
    """Keep only alpha (shape); force RGB to white for a pure silhouette."""
    a = img.split()[-1]
    out = Image.new("RGBA", img.size, (255, 255, 255, 0))
    out.putalpha(a)
    return out


def build_launcher_foreground(tooth: Image.Image, waves: Image.Image) -> Image.Image:
    canvas_size = LAUNCHER_SUPERSAMPLE
    safe_radius = LAUNCHER_SAFE_RADIUS_FRACTION * canvas_size
    target = safe_radius * LAUNCHER_MARGIN_FRACTION

    def layout(scale: float):
        tw, th = int(tooth.width * scale), int(tooth.height * scale)
        ww, wh = int(waves.width * scale), int(waves.height * scale)
        t_r = tooth.resize((tw, th), Image.LANCZOS)
        w_r = waves.resize((ww, wh), Image.LANCZOS)
        gap = int(LAUNCHER_GAP * scale)
        group_w, group_h = tw + gap + ww, max(th, wh)
        gx0 = (canvas_size - group_w) // 2
        gy_t = (canvas_size - group_h) // 2 + (group_h - th) // 2
        gy_w = (canvas_size - group_h) // 2 + (group_h - wh) // 2
        canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
        canvas.alpha_composite(t_r, (gx0, gy_t))
        canvas.alpha_composite(w_r, (gx0 + tw + gap, gy_w))
        return canvas, t_r, (gx0, gy_t), w_r, (gx0 + tw + gap, gy_w)

    def max_radius(t_r, t_off, w_r, w_off):
        center = (canvas_size / 2, canvas_size / 2)
        m = 0.0
        for img, (ox, oy) in ((t_r, t_off), (w_r, w_off)):
            a = img.split()[-1]
            px = a.load()
            for y in range(0, img.height, 2):
                for x in range(0, img.width, 2):
                    if px[x, y] > 16:
                        d = ((ox + x - center[0]) ** 2 + (oy + y - center[1]) ** 2) ** 0.5
                        m = max(m, d)
        return m

    lo, hi = 0.1, 3.0
    for _ in range(25):
        mid = (lo + hi) / 2
        _, t_r, t_off, w_r, w_off = layout(mid)
        if max_radius(t_r, t_off, w_r, w_off) > target:
            hi = mid
        else:
            lo = mid
    canvas, *_ = layout(lo)
    return canvas


def build_notification_icon(tooth: Image.Image, waves: Image.Image) -> Image.Image:
    inner = waves.crop((0, 0, int(waves.width * NOTIF_ARC_WIDTH_FRACTION), waves.height))
    inner = trim(inner.crop((0, 0, NOTIF_ARC_CROP_WIDTH, inner.height)))

    tooth_w, arc_w = whiten(tooth), whiten(inner)
    canvas_size = NOTIF_SUPERSAMPLE
    avail = canvas_size * (1 - 2 * NOTIF_MARGIN_FRACTION)
    gap = int(0.05 * canvas_size)

    group_w_raw = tooth_w.width + gap + arc_w.width
    group_h_raw = max(tooth_w.height, arc_w.height)
    scale = min(avail / group_w_raw, avail / group_h_raw)

    tw, th = int(tooth_w.width * scale), int(tooth_w.height * scale)
    aw, ah = int(arc_w.width * scale), int(arc_w.height * scale)
    t_r = tooth_w.resize((tw, th), Image.LANCZOS)
    a_r = arc_w.resize((aw, ah), Image.LANCZOS)

    group_w, group_h = tw + gap + aw, max(th, ah)
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 0))
    gx0 = (canvas_size - group_w) // 2
    gy_t = (canvas_size - group_h) // 2 + (group_h - th) // 2
    gy_a = (canvas_size - group_h) // 2 + (group_h - ah) // 2
    canvas.alpha_composite(t_r, (gx0, gy_t))
    canvas.alpha_composite(a_r, (gx0 + tw + gap, gy_a))

    a = canvas.split()[-1]
    a = a.filter(ImageFilter.MaxFilter(NOTIF_DILATE_PX)).filter(ImageFilter.GaussianBlur(NOTIF_BLUR_PX))
    out = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    out.putalpha(a)
    return out


def main() -> None:
    im = Image.open(SOURCE).convert("RGB")

    tooth = trim(to_rgba_alpha(im.crop(TOOTH_BOX), BG))
    waves_raw = to_rgba_alpha(im.crop(WAVES_BOX), BG)
    waves = trim(erase_patches(waves_raw, WAVES_ERASE_PATCHES))
    wordmark_raw = to_rgba_alpha(im.crop(WORDMARK_BOX), BG)
    wordmark = trim(erase_patches(wordmark_raw, WORDMARK_ERASE_PATCHES))

    # Launcher (adaptive icon foreground): tooth + waves, safe-zone verified.
    fg_master = build_launcher_foreground(tooth, waves)
    for folder, size in LAUNCHER_SIZES.items():
        d = os.path.join(RES, folder)
        os.makedirs(d, exist_ok=True)
        fg_master.resize((size, size), Image.LANCZOS).save(os.path.join(d, "ic_launcher_foreground.png"))
    print(f"launcher foreground: {len(LAUNCHER_SIZES)} densities written")

    # Notification icon: tooth + single innermost arc, thickened, monochrome.
    notif_master = build_notification_icon(tooth, waves)
    for folder, size in NOTIF_SIZES.items():
        d = os.path.join(RES, folder)
        os.makedirs(d, exist_ok=True)
        notif_master.resize((size, size), Image.LANCZOS).save(os.path.join(d, "ic_notification.png"))
    print(f"notification icon: {len(NOTIF_SIZES)} densities written")

    # Wordmark: cropped straight from the source, single asset.
    wordmark.save(os.path.join(RES, "drawable", "logo_wordmark.png"))
    print("wordmark: drawable/logo_wordmark.png written")


if __name__ == "__main__":
    main()
