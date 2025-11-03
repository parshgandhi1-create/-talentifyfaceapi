from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2, numpy as np, requests, os, re
from io import BytesIO
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "version": "url_final_render"})

@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json(force=True)
        school_id = data.get("school_id")
        folder_url = data.get("folder_url")   # ✅ e.g. https://talentify.co.in/list_images.php?school_id=1
        image_url = data.get("image_url")

        if not all([school_id, folder_url, image_url]):
            return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

        # ✅ Download target image
        target_res = requests.get(image_url, timeout=10)
        if target_res.status_code != 200:
            return jsonify({"error": "Failed to download target image"}), 404

        target_img_path = f"temp_target_{school_id}.jpg"
        with open(target_img_path, "wb") as f:
            f.write(target_res.content)

        # ✅ Fetch list of images from PHP (JSON response)
        folder_res = requests.get(folder_url, timeout=10)
        if folder_res.status_code != 200:
            return jsonify({"error": f"Folder not accessible: {folder_url}"}), 404

        try:
            file_list = folder_res.json()
        except:
            return jsonify({"error": "Invalid JSON received from folder_url"}), 500

        if not file_list:
            return jsonify({"error": f"No images found for school_id {school_id}"}), 404

        # ✅ Build full URLs from filenames
        base_url = f"https://talentify.co.in/uploads/schools/{school_id}/"
        img_urls = [base_url + f for f in file_list]

        # ✅ Compare faces using DeepFace
        best_match = None
        best_score = 0.0

        for img_url in img_urls:
            try:
                result = DeepFace.verify(img1_path=target_img_path, img2_path=img_url, enforce_detection=False)
                if result.get("verified"):
                    score = 1 - result.get("distance", 1.0)
                    if score > best_score:
                        best_score = score
                        best_match = img_url
            except Exception:
                continue

        # ✅ Cleanup temporary file
        if os.path.exists(target_img_path):
            os.remove(target_img_path)

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
