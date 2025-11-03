# ===================================
# FINAL app.py (Talentify Face Match)
# ===================================

from flask import Flask, request, jsonify
import requests
import face_recognition
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)

@app.route('/find_similar', methods=['POST'])
def find_similar():
    try:
        # ✅ 1. Read JSON body
        data = request.get_json(force=True)
        school_id = data.get('school_id')
        folder_url = data.get('folder_url')
        image_url = data.get('image_url')  # <-- matches PHP key

        # ✅ 2. Validate parameters
        if not school_id or not folder_url or not image_url:
            return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

        # ✅ 3. Download target image
        target_resp = requests.get(image_url, timeout=15)
        if target_resp.status_code != 200:
            return jsonify({"error": f"Failed to download target image: HTTP {target_resp.status_code}"}), 400

        target_img = face_recognition.load_image_file(BytesIO(target_resp.content))
        target_encodings = face_recognition.face_encodings(target_img)
        if not target_encodings:
            return jsonify({"error": "No face detected in target image"}), 400
        target_encoding = target_encodings[0]

        # ✅ 4. Get candidate image list from PHP
        list_url = f"https://talentify.co.in/school/list_images.php?school_id={school_id}"
        list_resp = requests.get(list_url, timeout=15)
        if list_resp.status_code != 200:
            return jsonify({"error": "Failed to get image list"}), 400

        image_files = list_resp.json()
        if not isinstance(image_files, list) or len(image_files) == 0:
            return jsonify({"error": "No candidate images found for comparison"}), 400

        # ✅ 5. Compare faces
        best_match = None
        best_score = 0.0

        for filename in image_files:
            candidate_url = f"{folder_url}/{filename}"
            try:
                resp = requests.get(candidate_url, timeout=10)
                if resp.status_code != 200:
                    continue

                img = face_recognition.load_image_file(BytesIO(resp.content))
                encodings = face_recognition.face_encodings(img)
                if not encodings:
                    continue

                result = face_recognition.compare_faces([target_encoding], encodings[0])
                distance = face_recognition.face_distance([target_encoding], encodings[0])[0]
                score = (1 - distance) * 100  # Convert to similarity %

                if result[0] and score > best_score:
                    best_score = score
                    best_match = filename

            except Exception:
                continue

        # ✅ 6. Return result
        if best_match:
            return jsonify({
                "match_found": True,
                "matched_image": best_match,
                "similarity_score": round(best_score, 2)
            })
        else:
            return jsonify({"match_found": False, "message": "No similar faces found"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    return "Talentify Face API running successfully ✅"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
