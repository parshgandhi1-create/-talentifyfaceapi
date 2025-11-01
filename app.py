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

# Lazy load DeepFace to avoid long cold starts
deepface_model = None
def get_deepface():
    global deepface_model
    if deepface_model is None:
        from deepface import DeepFace
        deepface_model = DeepFace
    return deepface_model


# ===================================
# HEALTH CHECK
# ===================================
@app.route("/")
def home():
    return jsonify({"status": "Talentify Face API is running ✅"})


# ===================================
# IMAGE PROXY
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
# FIND SIMILAR FACES (PHP Compatible)
# ===================================
@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json(force=True)

        school_id = data.get("school_id")
        folder_url = data.get("folder_url")
        image_url = data.get("image_url")

        if not school_id or not folder_url or not image_url:
            return jsonify({"error": "Missing parameters (school_id, folder_url, or image_url)"}), 400

        DeepFace = get_deepface()
        results = []

        # Download target image
        target_response = requests.get(image_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        target_response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as target_file:
            target_file.write(target_response.content)
            target_path = target_file.name

        # Extract base folder (real uploads folder without proxy prefix)
        folder_base = folder_url.replace("https://talentify.co.in/school/image_proxy.php?url=", "")
        r = requests.get(folder_base, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            os.remove(target_path)
            return jsonify({"error": f"Failed to access folder: {r.status_code}"}), 500

        # Detect candidate image URLs (basic HTML parse)
        candidate_images = []
        for line in r.text.splitlines():
            if any(ext in line.lower() for ext in [".jpg", ".jpeg", ".png"]):
                for part in line.split('"'):
                    if any(ext in part.lower() for ext in [".jpg", ".jpeg", ".png"]):
                        img_url = f"https://talentify.co.in/school/image_proxy.php?url={folder_base}/{part}"
                        if img_url != image_url:
                            candidate_images.append(img_url)

        if not candidate_images:
            os.remove(target_path)
            return jsonify({"error": "No candidate images found"}), 404

        # Compare each candidate with the target
        for img_url in candidate_images:
            try:
                candidate_response = requests.get(img_url, timeout=10)
                candidate_response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as candidate_file:
                    candidate_file.write(candidate_response.content)
                    candidate_path = candidate_file.name

                result = DeepFace.verify(
                    img1_path=target_path,
                    img2_path=candidate_path,
                    model_name="Facenet",
                    detector_backend="mtcnn",
                    enforce_detection=False
                )

                similarity = max(0, 1 - result.get("distance", 1.0))
                results.append({
                    "image_url": img_url,
                    "similarity": round(similarity, 2)
                })

                os.remove(candidate_path)

            except Exception as e:
                results.append({
                    "image_url": img_url,
                    "error": str(e)
                })

        os.remove(target_path)
        return jsonify({"similar_images": results})

    except Exception as e:
        return jsonify({
            "error": f"Failed to process: {str(e)}",
            "trace": traceback.format_exc()
        }), 500


# ===================================
# RUN APP
# ===================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
