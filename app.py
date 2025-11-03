# ===================================
# FINAL app.py (Talentify Face Match using DeepFace)
# ===================================

from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

# ✅ Helper to download and prepare images
def load_image_from_url(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return np.array(img)
    except Exception as e:
        print(f"Image load failed for {url}: {e}")
        return None

@app.route('/find_similar', methods=['POST'])
def find_similar():
    try:
        # ✅ 1. Read JSON body
        data = request.get_json(force=True)
        school_id = data.get('school_id')
        folder_url = data.get('folder_url')
        image_url = data.get('image_url')  # same key names as before

        # ✅ 2. Validate
        if not school_id or not folder_url or not image_url:
            return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

        # ✅ 3. Load target image
        target_img = load_image_from_url(image_url)
        if target_img is None:
            return jsonify({"error": "Failed to download target image"}), 400

        # ✅ 4. Get folder image list
        list_url = f"https://talentify.co.in/school/list_images.php?school_id={school_id}"
        list_resp = requests.get(list_url, timeout=15)
        if list_resp.status_code != 200:
            return jsonify({"error": "Failed to get image list"}), 400

        image_files = list_resp.json()
        if not isinstance(image_files, list) or len(image_files) == 0:
            return jsonify({"error": "No candidate images found for comparison"}), 400

        # ✅ 5. Compare using DeepFace
        best_match = None
        best_score = float("inf")
        best_file = None

        for filename in image_files:
            candidate_url = f"https://talentify.co.in/uploads/schools/{school_id}/{filename}"
            candidate_img = load_image_from_url(candidate_url)
            if candidate_img is None:
                continue

            try:
                result = DeepFace.verify(target_img, candidate_img, model_name="VGG-Face", enforce_detection=False)
                distance = result.get("distance", 1.0)
                if distance < best_score:
                    best_score = distance
                    best_match = filename
                    best_file = candidate_url
            except Exception as e:
                print(f"Comparison failed for {filename}: {e}")
                continue

        # ✅ 6. Return result
        if best_match:
            similarity = round((1 - best_score) * 100, 2)
            return jsonify({
                "match_found": True,
                "matched_image": best_match,
                "similarity_score": similarity,
                "image_url": best_file
            })
        else:
            return jsonify({"match_found": False, "message": "No similar faces found"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    return "Talentify Face API running successfully ✅"


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
