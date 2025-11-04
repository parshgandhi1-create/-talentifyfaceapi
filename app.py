from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2, numpy as np, requests, os, re
from io import BytesIO
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "version": "url_proxy_fix"})

@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json(force=True)
        school_id = data.get("school_id")
        folder_url = data.get("folder_url")
        image_url = data.get("image_url")

        if not all([school_id, folder_url, image_url]):
            return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

        # ✅ Build proxy URL for target image
        proxy_base = "https://talentify.co.in/school/image_proxy.php?url="
        proxy_image_url = proxy_base + image_url

        # ✅ Download via proxy
        target_res = requests.get(proxy_image_url, timeout=10)
        if target_res.status_code != 200:
            return jsonify({"error": "Failed to download target image"}), 404

        # ✅ Analyze target image
        target_img = np.frombuffer(target_res.content, np.uint8)
        target_img = cv2.imdecode(target_img, cv2.IMREAD_COLOR)

        # ✅ List all images in folder via list_images.php
        list_url = f"https://talentify.co.in/school/list_images.php?school_id={school_id}"
        folder_res = requests.get(list_url, timeout=10)
        if folder_res.status_code != 200:
            return jsonify({"error": f"Folder not accessible: {folder_url}"}), 404

        img_files = folder_res.json()
        if not isinstance(img_files, list) or not img_files:
            return jsonify({"error": "No images found in folder"}), 404

        # ✅ Compare images using DeepFace
        best_match = None
        best_score = 0.0

        for file_name in img_files:
            img_url = proxy_base + folder_url + file_name
            try:
                img_res = requests.get(img_url, timeout=10)
                if img_res.status_code != 200:
                    continue

                img_arr = np.frombuffer(img_res.content, np.uint8)
                img_cv = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                if img_cv is None:
                    continue

                result = DeepFace.verify(target_img, img_cv, model_name="VGG-Face", enforce_detection=False)
                score = result.get("similarity", 1 - result.get("distance", 1))
                if score > best_score:
                    best_score = score
                    best_match = folder_url + file_name
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
