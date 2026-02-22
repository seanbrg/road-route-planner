import os
import uuid
from pathlib import Path
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, flash
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
# Import the existing API module (do not modify it)
import api

# Monkeypatch GUI calls inside api so they don't block or try to open windows when running on a server
# We keep these no-ops local to the api module only.
api.cv2.imshow = lambda *args, **kwargs: None
api.plt.show = lambda *args, **kwargs: None

# Create app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")

# Development error handler: return full traceback for unhandled exceptions to aid debugging.
import traceback
@app.errorhandler(Exception)
def handle_exception(e):
    tb = traceback.format_exc()
    print('Unhandled exception in request:\n', tb)
    # Return a simple HTML page with the traceback for debugging (development only)
    return f"<h1>Internal Server Error</h1><pre>{tb}</pre>", 500

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "output"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "tif"}

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["OUTPUT_FOLDER"] = str(OUTPUT_FOLDER)

# Map result image filename -> original uploaded unique filename so we can re-annotate
RESULT_MAP = {}
# Map result image filename -> saved skeleton filename (if any)
RESULT_SKELETON = {}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        flash("No file part")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        saved_path = UPLOAD_FOLDER / unique_name
        file.save(str(saved_path))

        # Expose path to api module (api.process_image references image_path global)
        api.image_path = str(saved_path)

        # Create a browser-friendly preview (PNG) so annotate UI works with TIFF/JPEG/others
        from PIL import Image
        preview_failed = False
        try:
            img = Image.open(str(saved_path)).convert("RGB")
            preview_name = f"{unique_name}.preview.png"
            preview_path = UPLOAD_FOLDER / preview_name
            # Optionally resize large images for the preview to speed up browser rendering
            max_preview_dim = 1600
            if max(img.width, img.height) > max_preview_dim:
                scale = max_preview_dim / max(img.width, img.height)
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.LANCZOS)
            img.save(str(preview_path), format="PNG")
        except Exception as e:
            # If preview creation with PIL fails, try OpenCV as a fallback (many environments have cv2)
            try:
                import cv2
                import numpy as np
                img_cv = cv2.imdecode(np.fromfile(str(saved_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_cv is not None:
                    # Convert BGR -> RGB
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    preview_name = f"{unique_name}.preview.png"
                    preview_path = UPLOAD_FOLDER / preview_name
                    # Resize if necessary
                    h, w = img_cv.shape[:2]
                    max_preview_dim = 1600
                    if max(w, h) > max_preview_dim:
                        scale = max_preview_dim / max(w, h)
                        img_cv = cv2.resize(img_cv, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                    # Use imwrite via cv2 to ensure unicode paths on Windows work
                    cv2.imencode('.png', cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))[1].tofile(str(preview_path))
                else:
                    preview_name = unique_name
                    preview_failed = True
            except Exception:
                preview_name = unique_name
                preview_failed = True

        # Render annotation UI so the user can pick two points on the image
        image_url = url_for("uploaded_file", filename=preview_name)
        # Determine original image size so the client can map preview coords back to original coords
        try:
            orig_img = Image.open(str(saved_path))
            orig_w, orig_h = orig_img.size
        except Exception:
            # Fallback: if PIL can't open, leave None (JS will handle missing values)
            orig_w, orig_h = None, None

        return render_template("annotate.html", image_url=image_url, filename=unique_name, preview_failed=preview_failed, orig_w=orig_w or 0, orig_h=orig_h or 0)

    else:
        flash("File type not allowed")
        return redirect(url_for("index"))


# Serve uploaded files for display in the annotate UI
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# Serve files from the repository assets/ directory (e.g., assets/smile.png)
@app.route('/assets/<path:filename>')
def asset_file(filename):
    assets_dir = BASE_DIR / 'assets'
    return send_from_directory(str(assets_dir), filename)


# Serve generated output files (annotated images and path JSON)
@app.route('/output/<path:filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


# Process route: receives filename and two points, runs road_route_extraction, and saves result
@app.route('/process', methods=["POST"])
def process_annotation():
    filename = request.form.get('filename')
    x1 = request.form.get('x1')
    y1 = request.form.get('y1')
    x2 = request.form.get('x2')
    y2 = request.form.get('y2')

    if not filename or None in (x1, y1, x2, y2):
        return render_template("result.html", error="Missing filename or point coordinates", image_url=None)

    try:
        x1 = int(float(x1)); y1 = int(float(y1)); x2 = int(float(x2)); y2 = int(float(y2))
    except ValueError:
        return render_template("result.html", error="Invalid coordinate values", image_url=None)

    saved_path = UPLOAD_FOLDER / filename
    if not saved_path.exists():
        return render_template("result.html", error="Uploaded file not found", image_url=None)

    # Set module-level image_path used elsewhere
    api.image_path = str(saved_path)

    # Call existing function with the two selected points (pass as tuples)
    try:
        # Note: api.road_route_extraction signature is (image_path, point1, point2, model_path=...)
        # Build barriers list from new JSON field (list of [x,y]) and legacy checkbox flags
        import json
        barriers = []
        # parse extra barriers JSON (client sends [[x,y], ...])
        b_json = request.form.get('barriers', '[]')
        try:
            parsed = json.loads(b_json)
            if isinstance(parsed, list):
                for item in parsed:
                    # Expect item to be [y, x] (row, col)
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        try:
                            by = int(float(item[0])); bx = int(float(item[1]))
                            barriers.append((by, bx))
                        except Exception:
                            # skip malformed barrier points
                            continue
        except Exception:
            # ignore JSON parse errors and continue with any legacy flags
            pass

        # legacy single-point barrier flags (point A/B)
        barrierA_flag = request.form.get('barrierA', '0')
        barrierB_flag = request.form.get('barrierB', '0')
        if barrierA_flag == '1':
            barriers.append((y1, x1))
        if barrierB_flag == '1':
            barriers.append((y2, x2))

        # NOTE: internal processing expects coordinates as (y, x) == (row, col)
        use_cached = request.form.get('use_cached', '0')
        print(f"DEBUG: calling road_route_extraction (use_cached={use_cached}) with point1=(y,x)={(y1, x1)}, point2={(y2, x2)}, barriers={barriers}")

        if use_cached == '1':
            # Use cached skeleton and original image (no model run)
            try:
                result_img = api.road_route_extraction_from_skeleton((y1, x1), (y2, x2), barriers=barriers, image_path=str(saved_path))
            except Exception as e:
                return render_template("result.html", error=f"Cached re-annotation failed: {e}", image_url=None)
        else:
            # Full processing: run model then compute annotation
            result_img = api.road_route_extraction(str(saved_path), point1=(y1, x1), point2=(y2, x2), barriers=barriers, model_path=api.MODEL_PATH)
    except Exception as e:
        return render_template("result.html", error=str(e), image_url=None)

    # Save returned image (could be a grayscale mask or a colored BGR/RGB image)


    # API may return (image, path_coords) or just the image. Handle both.
    path_coords = None
    annotated_img = result_img
    if isinstance(result_img, (list, tuple)) and len(result_img) == 2:
        annotated_img, path_coords = result_img

    # If the annotated image is None, that means no path was found. Let the user re-select points on the same image
    if annotated_img is None:
        # Build preview name created at upload time (unique_name.preview.png)
        preview_name = f"{filename}.preview.png"
        preview_path = UPLOAD_FOLDER / preview_name
        preview_failed = False
        if not preview_path.exists():
            # fallback to original uploaded file (may be displayable)
            preview_name = filename
            preview_failed = True

        # Determine original image size so the client can map preview coords back to original coords
        try:
            from PIL import Image as PilImage
            orig_img = PilImage.open(str(saved_path))
            orig_w, orig_h = orig_img.size
        except Exception:
            orig_w, orig_h = 0, 0

        image_url = url_for('uploaded_file', filename=preview_name)
        # Render annotate template with use_cached so the server will reuse the cached skeleton
        return render_template('annotate.html', image_url=image_url, filename=filename, preview_failed=preview_failed, orig_w=orig_w or 0, orig_h=orig_h or 0, use_cached=True, no_route=True)

    try:
        arr = np.array(annotated_img)
    except Exception as e:
        return render_template("result.html", error=f"Unable to convert result to array: {e}", image_url=None)

    out_filename = f"result_{filename.rsplit('.',1)[0]}.png"
    out_path = OUTPUT_FOLDER / out_filename

    # Handle different array shapes/dtypes
    try:
        if arr.ndim == 2:
            # grayscale mask: ensure 0-255 uint8
            out_arr = (arr.astype(bool).astype(np.uint8) * 255).astype(np.uint8)
            Image.fromarray(out_arr).save(str(out_path))
        elif arr.ndim == 3 and arr.shape[2] == 3:
            # Color image; assume OpenCV BGR -> convert to RGB for PIL
            try:
                import cv2
                rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            except Exception:
                # If conversion fails, assume image is already RGB
                rgb = arr
            Image.fromarray(rgb).save(str(out_path))
        else:
            # Unknown shape: try to normalize and save
            norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
            Image.fromarray((norm * 255).astype(np.uint8)).save(str(out_path))
    except Exception as e:
        return render_template("result.html", error=f"Failed saving output image: {e}", image_url=None)

    # Record mapping so user can re-annotate this result and reuse the original upload/skeleton
    RESULT_MAP[out_filename] = filename

    # Save path coordinates JSON (client expects [x,y] pairs)
    path_json_name = None
    try:
        if path_coords:
            import json
            path_list = [[int(p[1]), int(p[0])] for p in path_coords]
            path_json_name = out_filename + '.path.json'
            path_json_path = OUTPUT_FOLDER / path_json_name
            with open(path_json_path, 'w', encoding='utf-8') as f:
                json.dump(path_list, f)
    except Exception:
        path_json_name = None

    # Save skeleton image if it's cached for this uploaded image_path
    skeleton_filename = None
    try:
        cache_key = str(saved_path)
        cached = getattr(api, 'SKELETON_CACHE', {}).get(cache_key)
        if cached is not None:
            sk = cached[0]
            # convert skeleton to uint8 image (0/255)
            import numpy as _np
            from PIL import Image as _PilImage
            sk_arr = _np.array(sk)
            # normalize boolean or 0/1/255 -> 0/255
            sk_bin = (_np.asarray(sk_arr) > 0).astype(_np.uint8) * 255
            skeleton_filename = out_filename.rsplit('.', 1)[0] + '.skeleton.png'
            _PilImage.fromarray(sk_bin).save(str(OUTPUT_FOLDER / skeleton_filename))
            # record mapping so show_result can find it
            RESULT_SKELETON[out_filename] = skeleton_filename
    except Exception:
        skeleton_filename = None

    return redirect(url_for("show_result", filename=out_filename))


@app.route('/reannotate/<path:result_filename>')
def reannotate(result_filename):
    # Find original upload filename
    orig = RESULT_MAP.get(result_filename)
    if not orig:
        return render_template("result.html", image_url=None, error="Original upload not found for re-annotation")

    # Preview file name
    preview_name = f"{orig}.preview.png"
    preview_path = UPLOAD_FOLDER / preview_name
    preview_failed = False
    if not preview_path.exists():
        preview_name = orig  # fallback to original file
        preview_failed = True

    # Determine original image size
    try:
        from PIL import Image as PilImage
        orig_img = PilImage.open(str(UPLOAD_FOLDER / orig))
        orig_w, orig_h = orig_img.size
    except Exception:
        orig_w, orig_h = 0, 0

    image_url = url_for('uploaded_file', filename=preview_name)
    # Render annotate template with use_cached flag set so the API will reuse the last skeleton
    return render_template('annotate.html', image_url=image_url, filename=orig, preview_failed=preview_failed, orig_w=orig_w or 0, orig_h=orig_h or 0, use_cached=True)


@app.route('/result/<path:filename>')
def show_result(filename):
    image_url = url_for("output_file", filename=filename)
    # If we know the original upload for this result, expose a re-annotate link
    orig_upload = RESULT_MAP.get(filename)
    reannotate_url = None
    if orig_upload:
        reannotate_url = url_for('reannotate', result_filename=filename)
    # Check for a saved path JSON next to the output image
    path_json_name = f"{filename}.path.json"
    path_url = None
    if (OUTPUT_FOLDER / path_json_name).exists():
        path_url = url_for('output_file', filename=path_json_name)

    # Provide URL to the walker image (assets/smile.png) if present
    walker_url = None
    try:
        walker_path = BASE_DIR / 'assets' / 'smile.png'
        if walker_path.exists():
            walker_url = url_for('asset_file', filename='smile.png')
    except Exception:
        walker_url = None

    # Provide URL to skeleton image if it was saved (look up in RESULT_SKELETON)
    skeleton_url = None
    try:
        candidate = RESULT_SKELETON.get(filename, filename.rsplit('.',1)[0] + '.skeleton.png')
        if candidate and (OUTPUT_FOLDER / candidate).exists():
            skeleton_url = url_for('output_file', filename=candidate)
    except Exception:
        skeleton_url = None

    return render_template("result.html", image_url=image_url, error=None, reannotate_url=reannotate_url, result_filename=filename, path_url=path_url, walker_url=walker_url, skeleton_url=skeleton_url)

if __name__ == "__main__":
    # Run the Flask development server bound to the requested local IP so other machines
    # on the same network can connect. Disable debug when exposing to other hosts.
    app.run(host="0.0.0.0", port=5000, debug=False)
