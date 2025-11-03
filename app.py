from flask import Flask, request, jsonify
import requests, os
import face_recognition
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "version": "url_final_render"})

@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json(force=True)
        school_id = data.get("school_id")
        folder_url = data.get("folder_url")
        image_url = data.get("image_url")

        if not all([school_id, folder_url, image_url]):
            return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

        # Download target image
        target_res = requests.get(image_url, timeout=10)
        if target_res.status_code != 200:
            return jsonify({"error": "Failed to download target image"}), 404

        target_img = face_recognition.load_image_file(BytesIO(target_res.content))
        target_encodings = face_recognition.face_encodings(target_img)
        if not target_encodings:
            return jsonify({"error": "No face found in target image"}), 400
        target_encoding = target_encodings[0]

        # Fetch all photos from folder
        import re
        from bs4 import BeautifulSoup

        folder_res = requests.get(folder_url, timeout=10)
        if folder_res.status_code != 200:
            return jsonify({"error": f"Folder not accessible: {folder_url}"}), 404

        soup = BeautifulSoup(folder_res.text, "html.parser")
        img_urls = [
            folder_url + href
            for href in re.findall(r'photo_[^"\'<>]+\.(?:jpg|jpeg|png|webp)', folder_res.text, re.IGNORECASE)
        ]
        if not img_urls:
            return jsonify({"error": f"No images found in {folder_url}"}), 404

        best_match = None
        best_score = 0.0

        for img_url in img_urls:
            try:
                img_res = requests.get(img_url, timeout=10)
                if img_res.status_code != 200:
                    continue
                img = face_recognition.load_image_file(BytesIO(img_res.content))
                enc = face_recognition.face_encodings(img)
                if not enc:
                    continue
                distance = face_recognition.face_distance([target_encoding], enc[0])[0]
                score = 1 - distance  # similarity
                if score > best_score:
                    best_score = score
                    best_match = img_url
            except Exception:
                continue

        if not best_match:
            return jsonify({"error": "No match found"}), 404

        return jsonify({
            "status": "success",
            "best_match": best_match,
            "score": round(float(best_score), 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
