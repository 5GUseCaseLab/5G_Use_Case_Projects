import cv2
import numpy as np

FRAME_W, FRAME_H = 640, 480


def crop_to_aspect_ratio(image, width=FRAME_W, height=FRAME_H):
    """Center-crop to width:height, then resize. No-op if already that size."""
    h, w = image.shape[:2]
    if (w, h) == (width, height):
        return image
    desired = width / height
    if w / h > desired:
        new_w = int(desired * h)
        off = (w - new_w) // 2
        image = image[:, off:off + new_w]
    else:
        new_h = int(w / desired)
        off = (h - new_h) // 2
        image = image[off:off + new_h, :]
    return cv2.resize(image, (width, height))


def get_darkest_area(gray):
    """Center of the darkest 20x20 block (sparsely sampled), vectorized.
    Same result as the original nested loops: 20px border ignored, blocks every
    10px, every 5th pixel sampled inside each block."""
    ignore, step, area, skip = 20, 10, 20, 5
    h, w = gray.shape
    ys = np.arange(ignore, h - ignore, step)
    xs = np.arange(ignore, w - ignore, step)
    if len(ys) == 0 or len(xs) == 0:
        return (w // 2, h // 2)

    sub = gray[::skip, ::skip].astype(np.float32)
    n = area // skip                                   # 4 samples per axis
    sums = cv2.boxFilter(sub, -1, (n, n), anchor=(0, 0), normalize=False,
                         borderType=cv2.BORDER_REPLICATE)
    block = sums[ys // skip][:, xs // skip]
    iy, ix = np.unravel_index(np.argmin(block), block.shape)
    return (int(xs[ix] + area // 2), int(ys[iy] + area // 2))


def threshold_in_square(gray, level, center, size=250):
    """Inverse binary threshold, zeroed outside a square around `center`."""
    _, th = cv2.threshold(gray, level, 255, cv2.THRESH_BINARY_INV)
    x, y = center
    half = size // 2
    y0, y1 = max(0, y - half), min(gray.shape[0], y + half)
    x0, x1 = max(0, x - half), min(gray.shape[1], x + half)
    out = np.zeros_like(th)
    out[y0:y1, x0:x1] = th[y0:y1, x0:x1]
    return out


def largest_round_contour(contours, min_area, max_ratio):
    """Largest contour with area >= min_area whose bbox aspect ratio <= max_ratio."""
    best, best_area = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area <= best_area:
            continue
        _, _, w, h = cv2.boundingRect(c)
        if max(w, h) / min(w, h) <= max_ratio:
            best, best_area = c, area
    return best


def ellipse_fit_score(binary, contour):
    """Combined goodness of an ellipse fit: fill * thick_px^2 * thin_ratio."""
    if len(contour) < 5:
        return 0.0
    ellipse = cv2.fitEllipse(contour)
    shape = binary.shape

    filled = np.zeros(shape, np.uint8)
    cv2.ellipse(filled, ellipse, 255, -1)
    area = cv2.countNonZero(filled)
    if area == 0:
        return 0.0
    fill = cv2.countNonZero(cv2.bitwise_and(binary, filled)) / area

    border = np.zeros(shape, np.uint8)
    cv2.drawContours(border, [contour], -1, 255, 1)
    total_border = cv2.countNonZero(border)
    if total_border == 0:
        return 0.0

    thick = np.zeros(shape, np.uint8)
    thin = np.zeros(shape, np.uint8)
    cv2.ellipse(thick, ellipse, 255, 10)
    cv2.ellipse(thin, ellipse, 255, 4)
    thick_px = cv2.countNonZero(cv2.bitwise_and(border, thick))
    thin_ratio = cv2.countNonZero(cv2.bitwise_and(border, thin)) / total_border
    return fill * thick_px * thick_px * thin_ratio


def refine_contour(contour):
    """Keep contour points whose local bend faces the centroid (vectorized,
    same selection rule and edge handling as the original loop)."""
    pts = contour.reshape(-1, 2).astype(np.int64)
    n = len(pts)
    spacing = n // 25
    if spacing < 1:
        return None

    idx = np.arange(n)
    prev = np.where((idx - spacing >= 0)[:, None], pts[idx - spacing], pts[-spacing])
    nxt = np.where((idx + spacing < n)[:, None], pts[(idx + spacing) % n], pts[spacing])

    to_centroid = pts.mean(axis=0) - pts
    bend = ((prev - pts) + (nxt - pts)) / 2.0
    keep = (to_centroid * bend).sum(axis=1) >= np.cos(np.radians(60))

    kept = pts[keep]
    if len(kept) < 5:
        return None
    return kept.astype(np.int32).reshape(-1, 1, 2)


def process_frame(frame):
    """Return the pupil ellipse ((cx, cy), (axis1, axis2), angle), or None if not found."""
    frame = crop_to_aspect_ratio(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    dx, dy = get_darkest_area(gray)
    darkest = int(gray[dy, dx])

    kernel = np.ones((5, 5), np.uint8)
    best_score, best_contour = 0.0, None

    for offset in (25, 15, 5):                         # relaxed, medium, strict
        binary = threshold_in_square(gray, darkest + offset, (dx, dy))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = largest_round_contour(contours, 1000, 3)
        if contour is None or len(contour) <= 5:
            continue
        score = ellipse_fit_score(dilated, contour)
        if score > best_score:
            best_score, best_contour = score, contour

    if best_contour is None:
        return None
    refined = refine_contour(best_contour)
    if refined is None:
        return None
    return cv2.fitEllipse(refined)


def detect_pupil_features(frame):
    """6-D normalized feature vector for model training, or None."""
    ellipse = process_frame(frame)
    if ellipse is None:
        return None
    (cx, cy), (axis1, axis2), angle = ellipse
    if axis1 <= 0 or axis2 <= 0:
        return None
    return np.array([
        cx / FRAME_W, cy / FRAME_H,
        axis1 / FRAME_W, axis2 / FRAME_H,
        np.sin(np.radians(angle)), np.cos(np.radians(angle)),
    ], dtype=np.float64)


if __name__ == "__main__":
    # Headless smoke test / benchmark: python -m detection.pupil <cam index or video path>
    import sys, time
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    cap = cv2.VideoCapture(int(arg) if arg.isdigit() else arg)
    times, found = [], 0
    for _ in range(100):
        ok, f = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        e = process_frame(f)
        times.append((time.perf_counter() - t0) * 1000)
        found += e is not None
    cap.release()
    if times:
        print(f"{len(times)} frames, detected {found}, avg {np.mean(times):.1f} ms, worst {max(times):.1f} ms")
        print("last pupil center:", None if e is None else tuple(int(v) for v in e[0]))