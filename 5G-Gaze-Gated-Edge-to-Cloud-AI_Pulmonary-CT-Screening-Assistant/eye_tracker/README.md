# Dual-Camera Pupil Detection & Gaze Dwell-Time Heatmap Pipeline

An OpenCV-based eye-tracking and visual attention mapping pipeline in Python. The system detects and fits a 2D ellipse to the dark pupil region from an eye-facing camera stream (`pupil.py`), maps pupil center coordinates to scene coordinates via a calibrated regression model, and accumulates temporal fixation density into Gaussian-smoothed heatmap overlays (`heatmap.py`).

## System Architecture & Script Workflow

The codebase is organized into two primary modules that operate sequentially during live capture or offline video evaluation:

| Module | Primary Role | Key Inputs | Key Outputs |
| :--- | :--- | :--- | :--- |
| `pupil.py` | Multi-threshold pupil segmentation, contour filtering, angle optimization, and ellipse fitting | Eye camera stream (`cv2.VideoCapture`) or `.mp4`/`.avi` video file | Fitted pupil rotated rectangle `((cx, cy), (axis1, axis2), angle)` or 6D normalized feature vector |
| `heatmap.py` | Dual-camera synchronization, gaze coordinate prediction, temporal dwell integration, and heatmap rendering | Live Eye Camera (`EYE_CAM_ID = 2`), Frontal Camera (`FRONT_CAM_ID = 1`), and `gaze_model.joblib` | Timestamped raw heatmap (`_heat.png`), alpha-blended scene overlay (`_overlay.png`), scene frame (`_frame.png`), and raw dwell grid (`_seconds.npy`) |

### 1. Pupil Detection Workflow (`pupil.py`)

Each raw frame captured from the eye camera or video file passes through a six-stage geometric computer vision pipeline inside `process_frame()` and `process_frames()`:

1. **Aspect-Ratio Center-Cropping & Standardization (`crop_to_aspect_ratio`)**:
   - Compares the incoming frame's aspect ratio (`current_width / current_height`) against the target `4:3` ratio (`640x480`).
   - Symmetrically crops excess horizontal or vertical margins around the optical center and resizes the frame to exactly `640x480` pixels so all downstream spatial thresholds and gaze coordinates remain resolution-independent.
2. **Coarse Darkest-Patch Localization (`get_darkest_area`)**:
   - Converts the standardized BGR frame to grayscale (`cv2.cvtColor`) and searches for the darkest `20x20 px` square region (`searchArea = 20`) while ignoring a `20 px` border (`ignoreBounds = 20`).
   - Uses two-level sparse sampling—stepping the block origin by `imageSkipSize = 10 px` across the image and summing pixel intensities every `internalSkipSize = 5 px` inside each candidate block—to locate the pupil interior (`darkest_point`) with minimal CPU overhead. Defaults to the frame center `(320, 240)` if no valid patch is found.
3. **Multi-Level Binary Thresholding & ROI Masking (`apply_binary_threshold`, `mask_outside_square`)**:
   - Samples the baseline grayscale intensity at `darkest_point` (`darkest_pixel_value`) and generates three inverted binary masks (`cv2.THRESH_BINARY_INV`) using progressive intensity tolerances:
     - **Strict (`+5`)**: Isolates only the deepest dark core of the pupil.
     - **Medium (`+15`)**: Captures standard pupil boundaries under moderate illumination.
     - **Relaxed (`+25`)**: Accommodates dimmer contrast or mild corneal haze.
   - Masks out all pixels outside a `250x250 px` bounding square centered on `darkest_point` (`mask_outside_square`) to eliminate dark eyelashes, eyebrow shadows, and camera housing artifacts near the frame edges.
4. **Morphological Dilation & Contour Filtering (`filter_contours_by_area_and_return_largest`)**:
   - Dilates each thresholded binary mask twice (`iterations = 2`) using a `5x5` rectangular ones kernel to bridge small gaps caused by specular corneal reflections (glints).
   - Extracts external contours (`cv2.RETR_EXTERNAL`, `cv2.CHAIN_APPROX_SIMPLE`) and discards noise by retaining only the single largest contour whose area is at least `pixel_thresh = 1000 px²` and whose bounding-rectangle aspect ratio (`max(w, h) / min(w, h)`) does not exceed `ratio_thresh = 3`.
5. **Candidate Ellipse Goodness Scoring (`check_ellipse_goodness`, `check_contour_pixels`)**:
   - Fits a preliminary ellipse (`cv2.fitEllipse`, requiring `> 5` contour points) to the winning contour from each of the three threshold levels and evaluates fit quality using a composite metric:
     $$\text{final\_goodness} = \text{fill\_ratio} \times (\text{overlap\_thick})^2 \times \text{ratio\_under\_ellipse}$$
   - `fill_ratio` (`current_goodness[0]`): Fraction of pixels inside the filled ellipse mask that are active (`255`) in the dilated binary image.
   - `overlap_thick` (`total_pixels[0]`): Count of contour border pixels falling within a `10 px`-thickened ellipse perimeter mask.
   - `ratio_under_ellipse` (`total_pixels[1]`): Fraction of total contour border pixels falling within a tighter `4 px`-thickened ellipse perimeter mask.
   - The threshold level (`relaxed`, `medium`, or `strict`) achieving the highest positive `final_goodness` score is selected.
6. **Angle-Based Contour Pruning & Final Fit (`optimize_contours_by_angle`)**:
   - Refines the winning contour to remove non-elliptical dents caused by corneal glints or upper-eyelid occlusion. For every contour vertex `i`, it computes vectors `vec1` and `vec2` to neighboring points spaced `len(all_contours) // 25` indices away, along with the vector toward the contour centroid (`vec_to_centroid`).
   - Retains only vertices where the directional dot product satisfies `np.dot(vec_to_centroid, (vec1 + vec2) / 2) >= cos(60°)` (`0.5`), ensuring local curvature faces inward toward the pupil center.
   - Fits the final rotated ellipse `((cx, cy), (axis1, axis2), angle)` to the pruned contour (`>= 5` points). When called via `detect_pupil_features()`, converts this ellipse into a 6-element normalized feature vector:
     `[cx / 640, cy / 480, axis1 / 640, axis2 / 480, sin(angle), cos(angle)]`.

### 2. Gaze Mapping & Heatmap Generation Workflow (`heatmap.py`)

`heatmap.py` orchestrates synchronized capture from the eye camera and frontal scene camera to build a time-integrated visual attention map:

1. **Dual-Stream Initialization (`open_camera`, `read_frame`)**:
   - Opens the eye camera (`EYE_CAM_ID = 2`) and frontal scene camera (`FRONT_CAM_ID = 1`) via `cv2.CAP_ANY`, requesting `640x480` resolution at `15 FPS`, and center-crops both streams with `crop_to_aspect_ratio()`.
2. **Real-Time Gaze Projection (`detect_pupil`, `predict_gaze`)**:
   - Extracts the integer pupil center `(center_x, center_y)` from `process_frame(eye_frame)[0]` and passes it as a `(1, 2)` float64 feature array into `gaze_model.joblib` (`model.predict(features)[0]`) to obtain predicted scene coordinates `(fx, fy)`.
3. **Windowed Dwell-Time Integration (`main`)**:
   - Pressing `g` resets a `480x640` float32 `dwell` matrix and starts a `WINDOW_SECONDS = 22` second timer using `time.perf_counter()`.
   - On every iteration with a valid in-bounds gaze prediction (`0 <= xi < 640`, `0 <= yi < 480`), the elapsed frame interval `dt = min(now - previous_time, MAX_DT)` (clamped at `MAX_DT = 0.2 s` to prevent lag spikes from inflating a single pixel) is added to `dwell[yi, xi]` and `valid_time`, while a green live gaze marker (`r = 8 px`) and red countdown timer are drawn on the `frontal` preview window.
4. **Gaussian Kernel Density Estimation & Alpha Blending (`make_heatmap`, `save_heatmap`)**:
   - Once `elapsed >= 22 s`, the sparse dwell grid is convolved with a 2D Gaussian filter (`cv2.GaussianBlur(dwell, (0, 0), sigma=25)`), normalized by its peak value (`blurred / blurred.max()`), scaled to `0–255` (`uint8`), and colorized with `cv2.COLORMAP_JET`.
   - Computes a smooth per-pixel opacity channel `alpha = clip(norm * 1.5, 0, 1) * 0.7` and alpha-blends `heat_color` onto the final frontal scene frame (`last_front`).
   - Saves four timestamped files (`gaze_heatmap_<YYYYMMDD_HHMMSS>_*`): `_heat.png`, `_overlay.png`, `_frame.png`, and `_seconds.npy`.

## Results Shown (Sample Output Analysis)

The images below show the captured output from experimental run `20261026_115039` (`gaze_heatmap_20261026_115039_*`), illustrating how raw temporal fixation data is transformed into an interpretable scene overlay.

### Raw Gaussian-Smoothed Dwell Heatmap (`_heat.png`)



- **Spatial Distribution**: Rendered on a `640x480` canvas using `cv2.COLORMAP_JET` after Gaussian smoothing (`BLUR_SIGMA_PX = 25`).
- **Fixation Hotspots (Dark Red / Orange-Yellow)**: Two dominant high-dwell fixation basins appear in the central-left column—an upper horizontal elliptical cluster (`x ≈ 240–300, y ≈ 110–140`) and a larger elongated lower cluster (`x ≈ 210–300, y ≈ 300–380`), accompanied by a secondary orange fixation node to the lower-left (`x ≈ 110–140, y ≈ 290–320`).
- **Saccadic Bridges & Unvisited Zones (Cyan / Deep Blue)**: Cyan and green bridges connect the upper, middle-left, and lower fixation zones along the scanpath, while the right third of the field (`x > 400`) remains uniform deep blue (`0.0 s` dwell time).

### Scene-Aligned Alpha Overlay (`_overlay.jpg`)



- **Scene Context**: The frontal camera (`640x480`) is angled upward toward an indoor laboratory/office ceiling featuring four recessed rectangular LED light panels, an office chair mesh headrest in the upper-right foreground, and a partition wall with a presentation poster (`IoT-BASED ... PARKING SYSTEM`) on the right.
- **Selective Opacity Blending**: Because `make_heatmap()` scales alpha as `clip(norm * 1.5, 0, 1) * 0.7`, zero-dwell regions (`alpha = 0`) leave the right-hand partition, poster text, and chair mesh completely unoccluded, while peak fixation zones reach up to `70%` JET colormap opacity.
- **Visual Attention Interpretation**: The two primary red fixation hotspots fall along the central ceiling corridor between the left and right pairs of recessed light fixtures, with secondary yellow/cyan gaze clusters aligning over the lower-left and upper-left ceiling light panels and the far-left structural beam.

## Reproducibility Guide

### 1. Software Dependencies & Environment Setup

Use Python 3.9+ and install the required scientific and computer vision packages:

```bash
pip install opencv-python numpy joblib scikit-learn matplotlib
```

*(Note: `tkinter` is included with standard Python installations on Windows and macOS; on Debian/Ubuntu Linux, install `python3-tk` via `apt` if using the file-dialog selector in `pupil.py`.)*

### 2. Required Directory Structure

`heatmap.py` imports `crop_to_aspect_ratio` and `process_frame` from `detection.pupil` (`from detection.pupil import crop_to_aspect_ratio, process_frame`) and searches for `gaze_model.joblib` in either the script directory or a `./python/` subdirectory. Organize your working directory as follows so imports and model loading succeed without modifying source code:

```text
project_root/
├── heatmap.py
├── gaze_model.joblib
└── detection/
    ├── __init__.py
    └── pupil.py
```

### 3. Hardware & Calibration Prerequisites

1. **Dual USB Cameras (`heatmap.py`)**:
   - Connect both the eye-facing IR/close-up camera and the frontal world-facing camera before launching `heatmap.py`.
   - Verify camera device indices in the configuration block at the top of `heatmap.py`:
     - `EYE_CAM_ID = 2`
     - `FRONT_CAM_ID = 1`
   - If your OS assigns different indices (e.g., `0` and `1`), update `EYE_CAM_ID` and `FRONT_CAM_ID` accordingly.
2. **Pre-Trained Gaze Calibration Model (`gaze_model.joblib`)**:
   - `heatmap.py` requires a serialized `joblib` estimator implementing `.predict(X)` that maps a `(1, 2)` float64 pupil center array `[[center_x, center_y]]` (in `640x480` eye-camera coordinates) to a `(1, 2)` frontal scene coordinate array `[[fx, fy]]` (in `640x480` frontal-camera coordinates).
   - **Feature Dimensionality Note**: `heatmap.py` uses `detect_pupil()`, which extracts the 2D center `(center_x, center_y)` from `process_frame()`. If you train a model using the 6D normalized feature vector from `pupil.detect_pupil_features()` (`[cx/w, cy/h, axis1/w, axis2/h, sin(angle), cos(angle)]`), update `detect_pupil()` in `heatmap.py` to call `detect_pupil_features()` so input dimensions match.

### 4. Known Reproducibility Adjustments in `pupil.py`

When running `pupil.py` standalone across different machines or operating systems, address these hardcoded paths and platform settings:

| Location in `pupil.py` | Current Hardcoded Setting | Recommended Adjustment for Reproducibility |
| :--- | :--- | :--- |
| `process_video()` (line 322) | `cv2.VideoWriter('C:/Storage/Source Videos/output_video.mp4', ...)` | Replace with a relative path such as `'output_video.mp4'` (note: `out.write()` is not called inside the loop, so `output_video.mp4` is initialized as an empty container unless `out.write(frame)` is added). |
| `process_video()` (line 327) | `cv2.VideoCapture(1, cv2.CAP_DSHOW)` (`input_method == 2`) | `cv2.CAP_DSHOW` is Windows DirectShow-specific. On Linux or macOS, replace `cv2.CAP_DSHOW` with `cv2.CAP_ANY` and verify camera index `1`. |
| `select_video()` (line 403) | `video_path = 'C:/Users/sitanshu/Downloads/eye_test.mp4'` | Automatically falls back to a `tkinter` file-selection dialog (`filedialog.askopenfilename`) if the path does not exist. |
| `__main__` block (line 442) | Calls `process_video("", 2)` (live webcam mode) | Uncomment `select_video()` and comment out `process_video("", 2)` to test offline `.mp4`/`.avi` recordings. |

### 5. Running the Scripts & Interactive Controls

#### Run Standalone Pupil Detection (`pupil.py`)

```bash
python detection/pupil.py
```

| Key | Action in `pupil.py` |
| :--- | :--- |
| `SPACE` | Pause / resume video playback |
| `d` | Toggle debug windows (`relaxed`, `medium`, and `strict` thresholds, goodness metrics, and darkest patch overlay) |
| `c` | Save current annotated grayscale frame with `(X, Y)` coordinates to `frame-serial-<N>.jpeg` |
| `t` | Append current pupil center `(x, y),` to `pupil.txt` |
| `q` | Release video/camera handles and quit |

#### Run Live Gaze Heatmap Capture (`heatmap.py`)

```bash
python heatmap.py
```

| Key | Action in `heatmap.py` |
| :--- | :--- |
| `g` | Reset dwell grid and start a `22 s` (`WINDOW_SECONDS`) gaze recording session; automatically exports `_heat.png`, `_overlay.png`, `_frame.png`, and `_seconds.npy` upon completion |
| `q` | Close camera streams and exit |
