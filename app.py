from flask import Flask, request, jsonify, send_file
import os
import requests
from io import BytesIO
from PIL import Image
import tempfile
import traceback

# ===================================
# FLASK INITIALIZATION
# ===================================
app = Flask(__name__)

# Lazy load for DeepFace
deepface_model = None
def get_deepface():
    global deepface_model
    if deepface_model is None:
        from deepface import DeepFace
        deepface_model = DeepFace
    return deepface_model


# ===================================
# HEALTH CHECK (Render uses this)
# ===================================
@app.route("/")
def home():
    return jsonify({"status": "Talentify Face API is running ✅"})


# ===================================
# IMAGE PROXY
# (used by find_similar.php to bypass 406 / blocked images)
# ===================================
@app.route("/image_proxy.php")
def image_proxy():
    url = request.args.get("url")
    if not url:
        return jsonify({"error": "Missing 'url' parameter"}), 400

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Talentify Image Proxy)"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()

        img = Image.open(BytesIO(r.content))
        img_format = img.format if img.format else "JPEG"

        img_io = BytesIO()
        img.save(img_io, format=img_format)
        img_io.seek(0)

        return send_file(img_io, mimetype=f"image/{img_format.lower()}")

    except Exception as e:
        return jsonify({
            "error": f"Failed to download image: {str(e)}",
            "trace": traceback.format_exc()
        }), 500


# ===================================
# FIND SIMILAR FACES
# (Compatible with your PHP: find_similar.php)
# ===================================
@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json(force=True)

        target_image_url = data.get("target_image")
        candidate_images = data.get("candidate_images")

        if not target_image_url or not candidate_images:
            return jsonify({"error": "Missing parameters"}), 400

        # Download target image
        target_response = requests.get(
            f"https://talentify.co.in/school/image_proxy.php?url={target_image_url}",
            headers={"User-Agent": "Mozilla/5.0 (Talentify API)"},
            timeout=10
        )
        target_response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as target_file:
            target_file.write(target_response.content)
            target_path = target_file.name

        DeepFace = get_deepface()

        results = []
        for img_url in candidate_images:
            try:
                candidate_response = requests.get(
                    f"https://talentify.co.in/school/image_proxy.php?url={img_url}",
                    headers={"User-Agent": "Mozilla/5.0 (Talentify API)"},
                    timeout=10
                )
                candidate_response.raise_for_status()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as candidate_file:
                    candidate_file.write(candidate_response.content)
                    candidate_path = candidate_file.name

                verification = DeepFace.verify(
                    img1_path=target_path,
                    img2_path=candidate_path,
                    model_name="Facenet",
                    detector_backend="mtcnn",
                    enforce_detection=False
                )

                results.append({
                    "image": img_url,
                    "verified": verification.get("verified", False),
                    "distance": verification.get("distance", None)
                })

            except Exception as e:
                results.append({
                    "image": img_url,
                    "error": str(e)
                })

        os.remove(target_path)
        return jsonify({"results": results})

    except Exception as e:
        return jsonify({
            "error": f"Failed to process: {str(e)}",
            "trace": traceback.format_exc()
        }), 500


# ===================================
# RUN APP (Render uses $PORT)
# ===================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
