# Bloom pre-deployment audit

Date: 2026-11-11. Local Windows/Python 3.11/Vite verification; cloud deployment has not been run.

## API and component checks

- 17 endpoint scenarios returned their expected status: root, health, registration, login, current user, history read/create/delete, recovery question, password reset, login after reset, invalid upload rejection, single scan, hybrid scan, front capture check, three-angle scan, and routine generation. Auth and history checks used a temporary SQLite database.
- A real JPEG of `skimage.data.astronaut()` passed the image quality and MediaPipe face gates and returned HTTP 200 from `/api/analyse`. The response contained three classical concerns, 11 image-view fields, four finite ML signal scores, and an acne detection object. A clean image produced zero acne detections, which is a valid result.
- The same image returned HTTP 200 from `/api/analyse/multi`; all three `angle_results` now include their own `ml` result. `/api/capture-check` accepted the front view. This did not verify the left/right yaw gates with distinct side-view photos.
- The acne detector returned positive regions on local Papules, Pustules, Whiteheads, Blackheads, and Cyst sample images when called directly. Its checkpoint reports the single class `acne`; it does not emit the subclass labels shown in some example payloads. These cropped samples were not used for a full face-gated API scan.
- In-memory instrumentation confirmed one call each to the signal and acne inference functions for a successful single scan. Both models were unloaded before the request and loaded afterward. The model input was byte-for-byte equal to the enhanced RGB image produced by the classical pipeline.
- Missing checkpoint paths returned structured errors and `ml.available: false` without raising into the main pipeline.
- The frontend production build passed. The password visibility and ML results code compiled; browser interaction was not part of this API audit.
- After restarting the local launcher, the live frontend returned HTTP 200. A real POST through its `/api` proxy returned a successful scan with `ml.available: true`; the live health endpoint then reported both lazy-loaded models available.

## Before and after

| Check | Before enhancement or ML | After enhancement or ML |
| --- | --- | --- |
| Image mean RGB value, one example | 114.60 | 120.21 |
| Mean absolute RGB pixel difference | Original image | 14.82 per channel after enhancement |
| Signal model on the same image | Original: structure 0.655, hydration 0.615, sun damage 0.645, elasticity 0.592 | Enhanced: 0.685, 0.645, 0.666, 0.647 |
| ML disabled in memory vs enabled | `ml.available: false`, no signal/acne output | `ml.available: true`, four signals and acne output |
| Classical concern values, image views, skin tone | Baseline run | Exactly equal with ML enabled |
| Example local wall time | 2.49 s with ML disabled | 8.61 s with cold ML loading; not a controlled performance benchmark |

The ML results supplement the classical pipeline. They do not change classical concern scores, image overlays, or the rule-based routine. The frontend displays them separately.

## Fixes made during this audit

- CORS now allows the browser's `Authorization` header and `DELETE` method. Previously, authenticated history GET and DELETE preflights returned HTTP 400 even for an allowed origin; they now return HTTP 200.
- `processing_time_sec` now includes visualization and ML inference. In a real cold scan, the API reported 10.46 s versus 10.51 s measured at the test client; the response also reports separate visualization and ML stage times.
- `/api/health` now distinguishes installed checkpoints from models loaded on first use. In one process it changed from `installed: true, available: false` before a scan to `installed: true, available: true` after the scan.
- Multi-angle responses now expose per-angle ML results, and the frontend angle selector displays the selected angle's ML result.

## Remaining deployment checks

- Run the exact Render build and start commands on the chosen service size. The local audit does not prove that PyTorch, MediaPipe, and both checkpoints fit the cloud memory and disk limits. The ML packages are currently installed by `setup_ml.sh`, rather than pinned in `backend/requirements.txt`.
- Configure a persistent SQLite disk or a durable database, `AUTH_SECRET`, `ALLOWED_ORIGINS`, and the frontend's build-time `VITE_API_URL` before opening the site to users.
- Verify the full production browser flow, including password toggle, cross-origin login/history, valid face upload, and ML display. Test distinct left and right side images for guided capture.
- These tests verify execution and wiring, not clinical accuracy or generalization across skin tones, lighting, cameras, and acne types.
